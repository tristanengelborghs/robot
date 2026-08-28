"""Teacher training: PPO on privileged state, with an adaptive curriculum.

    python -m dexhand.train --config configs/smoke.yaml
    python -m dexhand.train --config configs/smoke.yaml env.hand=toy train.n_envs=4

The curriculum widens goal difficulty when the measured success rate clears a
threshold, rather than on a step schedule. A step schedule encodes an assumption
about how fast this run learns; gating on evidence means a small smoke run and
a long GPU run traverse the same curriculum at their own pace, and a run that
never learns never gets goals it cannot reach (which would only add noise to
the diagnosis).

Logging is the sibling project's discipline: every iteration appends one JSON
line with everything needed to reconstruct the curve — no aggregate-only
numbers, no TensorBoard requirement to read a result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dexhand.config import config_hash, load_config
from dexhand.env import DRConfig, EnvConfig, ReorientEnv
from dexhand.networks import ActorCritic
from dexhand.ppo import PPO, RolloutBuffer, RunningNorm
from dexhand.vec import ThreadedVecEnv


def bootstrap_truncations(rew, done, infos, policy, norm_c, gamma: float, device) -> np.ndarray:
    """Fold gamma*V(final_obs) into the reward of time-limit truncations.

    An episode that hits its step limit did not fail; the state it stopped in
    has exactly as much future value as any other. Cutting the bootstrap there
    (which storing the merged done flag does) tells the critic that value goes
    to zero at t=200 — and since no observation carries time-in-episode, the
    critic cannot represent that and instead drags V down everywhere. Folding
    the continuation value into the final reward is the standard fix; the trace
    is still cut at `done` because a new episode really does begin.

    Terminations (drops) get no bootstrap: that state has no future.
    """
    idx = [i for i in range(len(infos)) if done[i] and not infos[i].get("terminated", True)]
    if not idx:
        return rew
    final_c = np.stack([infos[i]["final_obs"]["critic"] for i in idx])
    c_t = torch.as_tensor(norm_c.normalize(final_c), dtype=torch.float32, device=device)
    with torch.no_grad():
        _, _, v = policy.act(c_t, c_t)   # the critic only reads its own input
    rew = rew.copy()
    rew[idx] += gamma * v.cpu().numpy()
    return rew


def make_env_cfg(cfg: dict) -> EnvConfig:
    e = dict(cfg["env"])
    e.pop("hand", None)
    dr = DRConfig(**e.pop("dr", {}))
    return EnvConfig(**e, dr=dr)


def build_vec(cfg: dict, seed: int) -> ThreadedVecEnv:
    hand = cfg["env"]["hand"]
    env_cfg = make_env_cfg(cfg)

    def make(i):
        return lambda: ReorientEnv(hand, env_cfg, seed=seed + i)

    return ThreadedVecEnv([make(i) for i in range(cfg["train"]["n_envs"])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg["device"])

    vec = build_vec(cfg, seed)
    t = cfg["train"]
    n_envs, horizon = int(t["n_envs"]), int(t["horizon"])
    obs = vec.reset(seeds=[seed + i for i in range(n_envs)])
    actor_dim = obs["actor"].shape[1]
    critic_dim = obs["critic"].shape[1]
    act_dim = vec.action_space.shape[0]

    policy = ActorCritic(actor_dim, critic_dim, act_dim, hidden=tuple(t["hidden"])).to(device)
    ppo = PPO(policy, lr=float(t["lr"]), clip=float(t["clip"]), epochs=int(t["epochs"]),
              n_minibatches=int(t["n_minibatches"]), ent_coef=float(t["ent_coef"]),
              vf_coef=float(t["vf_coef"]), max_grad_norm=float(t["max_grad_norm"]),
              target_kl=t.get("target_kl"))
    norm_a, norm_c = RunningNorm((actor_dim,)), RunningNorm((critic_dim,))

    run_name = cfg.get("run_name") or (
        f"{cfg['env']['hand']}-{cfg['env']['goal_mode']}-s{seed}-{config_hash(cfg)}"
    )
    out = Path(cfg["output_dir"]) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    log = open(out / "train_log.jsonl", "a", buffering=1)

    total_steps = int(t["total_steps"])
    n_iters = max(1, total_steps // (n_envs * horizon))
    print(f"[train] {run_name}: {n_iters} iters x {n_envs} envs x {horizon} steps "
          f"({n_iters * n_envs * horizon:,} env steps) on {device}")

    ep_returns = np.zeros(n_envs)
    finished_returns: list = []
    finished_successes: list = []
    curriculum_angle = float(cfg["env"]["max_goal_angle"])
    goal_mode = cfg["env"]["goal_mode"]
    start = time.time()

    for it in range(1, n_iters + 1):
        buf = RolloutBuffer(horizon, n_envs, actor_dim, critic_dim, act_dim)
        rot_dists = []
        for _ in range(horizon):
            norm_a.update(obs["actor"])
            norm_c.update(obs["critic"])
            a_t = torch.as_tensor(norm_a.normalize(obs["actor"]), dtype=torch.float32, device=device)
            c_t = torch.as_tensor(norm_c.normalize(obs["critic"]), dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logp, value = policy.act(a_t, c_t)
            action_np = action.cpu().numpy()
            next_obs, rew, done, infos = vec.step(action_np)
            rew = bootstrap_truncations(rew, done, infos, policy, norm_c, float(t["gamma"]), device)

            buf.add(norm_a.normalize(obs["actor"]), norm_c.normalize(obs["critic"]),
                    action_np, logp.cpu().numpy(), rew, done, value.cpu().numpy())
            ep_returns += rew
            for i, info in enumerate(infos):
                rot_dists.append(info["rot_dist"])
                if done[i]:
                    finished_returns.append(ep_returns[i])
                    finished_successes.append(infos[i]["successes"])
                    ep_returns[i] = 0.0
            obs = next_obs

        a_t = torch.as_tensor(norm_a.normalize(obs["actor"]), dtype=torch.float32, device=device)
        c_t = torch.as_tensor(norm_c.normalize(obs["critic"]), dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, last_value = policy.act(a_t, c_t)
        # `done` here is the final stored step's flag. Passing zeros instead
        # (the first version did) bootstraps an episode that ended on the
        # horizon edge with V(next episode's first obs): a drop on that step
        # gets a return of +100 instead of -20. Caught in review, verified by
        # hand-computing the target.
        buf.compute_gae(last_value.cpu().numpy(), done,
                        gamma=float(t["gamma"]), lam=float(t["gae_lambda"]))
        metrics = ppo.update(buf)

        # -- curriculum ----------------------------------------------------
        cur = cfg["curriculum"]
        recent = finished_successes[-20 * n_envs:]
        mean_succ = float(np.mean(recent)) if recent else 0.0
        if cur["enabled"] and goal_mode == "z" and recent and mean_succ >= float(cur["success_rate_to_widen"]):
            curriculum_angle = min(curriculum_angle * float(cur["widen_factor"]),
                                   float(cur["max_angle_cap"]))
            if curriculum_angle >= float(cur["switch_to_full_at"]):
                goal_mode = "full"
            for e in vec.envs:
                e.cfg.max_goal_angle = curriculum_angle
                e.cfg.goal_mode = goal_mode
            finished_successes.clear()
            print(f"[curriculum] widened to {curriculum_angle:.2f} rad, mode={goal_mode}")

        if it % int(t["log_every"]) == 0 or it == n_iters:
            row = {
                "iter": it,
                "env_steps": it * n_envs * horizon,
                "mean_return": float(np.mean(finished_returns[-50:])) if finished_returns else None,
                "mean_successes": mean_succ,
                "mean_rot_dist": float(np.mean(rot_dists)),
                "curriculum_angle": curriculum_angle,
                "goal_mode": goal_mode,
                "sps": int(it * n_envs * horizon / (time.time() - start)),
                **{k: float(v) for k, v in metrics.items()},
            }
            log.write(json.dumps(row) + "\n")
            print(f"  it {it:4d}/{n_iters}  return {row['mean_return'] if row['mean_return'] is None else round(row['mean_return'],1)}  "
                  f"succ/ep {mean_succ:.2f}  rot_dist {row['mean_rot_dist']:.2f}  sps {row['sps']}")

        if it % int(t["ckpt_every"]) == 0 or it == n_iters:
            torch.save({
                "policy": policy.state_dict(),
                "norm_actor": norm_a.state_dict(),
                "norm_critic": norm_c.state_dict(),
                "config": cfg,
                "actor_dim": actor_dim, "critic_dim": critic_dim, "act_dim": act_dim,
                "iter": it,
            }, out / "ckpt_last.pt")

    vec.close()
    log.close()
    print(f"[done] {(time.time()-start)/60:.1f} min -> {out}")
    print(f"[next] python -m dexhand.eval --ckpt {out}/ckpt_last.pt")


if __name__ == "__main__":
    main()
