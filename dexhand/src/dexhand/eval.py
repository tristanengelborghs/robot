"""Rollout evaluation with per-episode outcomes on disk.

    python -m dexhand.eval --ckpt runs/<run>/ckpt_last.pt --episodes 50

Same discipline as the sibling project: every episode gets a JSONL row (seed,
successes, time-to-first-success, dropped, mean rotation distance), because an
aggregate success rate that cannot be re-tested is a claim, not a result. The
observation normalizer is loaded from the checkpoint — evaluating with fresh
normalization statistics is the classic way to make a good policy look broken.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dexhand.env import ReorientEnv
from dexhand.networks import ActorCritic
from dexhand.ppo import RunningNorm
from dexhand.train import make_env_cfg


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = ActorCritic(ckpt["actor_dim"], ckpt["critic_dim"], ckpt["act_dim"],
                         hidden=tuple(ckpt["config"]["train"]["hidden"]))
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    norm_a = RunningNorm((ckpt["actor_dim"],))
    norm_a.load_state_dict(ckpt["norm_actor"])
    return policy, norm_a, ckpt["config"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)   # disjoint from training seeds
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the policy instead of taking its mean")
    ap.add_argument("--no-dr", action="store_true", help="evaluate on nominal physics")
    args = ap.parse_args()

    policy, norm_a, cfg = load_ckpt(args.ckpt)
    if args.no_dr:
        cfg["env"]["dr"] = {"enabled": False}
    env = ReorientEnv(cfg["env"]["hand"], make_env_cfg(cfg), seed=args.seed)

    out = Path(args.ckpt).parent / "eval_episodes.jsonl"
    rows = []
    with open(out, "a", buffering=1) as f:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            first_success_step = None
            dists = []
            step = 0
            while True:
                a_t = torch.as_tensor(norm_a.normalize(obs["actor"][None]), dtype=torch.float32)
                with torch.no_grad():
                    action, _, _ = policy.act(a_t, a_t, deterministic=not args.stochastic)
                obs, r, term, trunc, info = env.step(action.numpy()[0])
                dists.append(info["rot_dist"])
                step += 1
                if info["is_success"] and first_success_step is None:
                    first_success_step = step
                if term or trunc:
                    break
            row = {
                "episode": ep, "seed": args.seed + ep,
                "successes": info["successes"], "dropped": info["dropped"],
                "steps": step, "first_success_step": first_success_step,
                "mean_rot_dist": float(np.mean(dists)),
                "dr": not args.no_dr, "deterministic": not args.stochastic,
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    succ = [r["successes"] for r in rows]
    print(f"{len(rows)} episodes | successes/ep {np.mean(succ):.2f} "
          f"(median {np.median(succ):.0f}, max {max(succ)}) | "
          f"dropped {100*np.mean([r['dropped'] for r in rows]):.0f}% | rows -> {out}")


if __name__ == "__main__":
    main()
