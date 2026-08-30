"""Train PPO on the lift task. Runs on the GPU box.

The loop is deliberately thin: everything that is easy to get quietly wrong --
advantage estimation, the clipped objective, normalisation -- lives in
:mod:`harness.ppo` and is tested there against hand-computed values. What is left
here is collecting experience, calling those functions, and logging.

    ./isaaclab.sh -p scripts/train_ppo.py --headless --viz none \\
        --num_envs 1024 --iterations 500

``--pick-only`` is the interesting switch. The shipped reward is dominated by
goal tracking -- lift the block *and take it to a commanded pose* -- while the
task we asked for is just picking it up. Setting the goal weights to zero leaves
reaching and the step-function lift bonus, and asks the question
reference/LIFT.md raises: is a large sparse bonus enough for PPO to discover a
grasp on its own, or was goal tracking doing the work? Their reward cannot
answer that, because it never runs without it.

Success is measured the same way scripts/eval_diffusion.py measures it, so the
two halves of the project produce comparable numbers rather than two
incomparable ones.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO on the lift task.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=500)
parser.add_argument("--steps", type=int, default=32, help="Steps collected per environment per iteration.")
parser.add_argument("--out", type=str, default="/workspace/runs/ppo")
parser.add_argument("--pick-only", action="store_true", help="Zero the goal-tracking rewards; keep reach and lift.")
parser.add_argument("--success_height", type=float, default=0.10)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--checkpoint_every", type=int, default=50, help="Keep a numbered checkpoint this often.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs with a live simulator."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.ppo import ActorCritic, PPOConfig, RolloutBuffer, compute_gae, normalize, ppo_losses  # noqa: E402


def build_env_cfg():
    """Parse the task config, optionally stripping the goal-tracking rewards."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    if args_cli.pick_only:
        # Leave reaching and the lift bonus; drop both goal terms. The block
        # then has to be picked up for its own sake.
        env_cfg.rewards.object_goal_tracking.weight = 0.0
        env_cfg.rewards.object_goal_tracking_fine_grained.weight = 0.0
        print("[reward] pick-only: goal tracking disabled, reach + lift remain")

    return env_cfg


def lifted_fraction(env, height: float) -> float:
    """Fraction of environments with the block above ``height`` right now.

    A cheap running measure of whether anything is being picked up. It is not
    the same as the episode success rate -- a block held for one step counts --
    but it moves early, which is what a training log needs.
    """
    z = env.scene["object"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (z > height).float().mean().item()


def main() -> None:
    torch.manual_seed(args_cli.seed)

    env_cfg = build_env_cfg()
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = env.device

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    obs_dim = obs.shape[1]
    action_dim = env.action_space.shape[1]

    cfg = PPOConfig(obs_dim=obs_dim, action_dim=action_dim, learning_rate=args_cli.lr)
    policy = ActorCritic(cfg).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    buffer = RolloutBuffer(args_cli.steps, env.num_envs, obs_dim, action_dim, device)

    print(f"\nobservations {obs_dim}   actions {action_dim}   envs {env.num_envs}")
    print(f"policy       {sum(p.numel() for p in policy.parameters()):,} parameters")
    print(f"batch        {args_cli.steps * env.num_envs:,} transitions per iteration\n", flush=True)

    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    started = time.time()

    for iteration in range(1, args_cli.iterations + 1):
        buffer.reset()
        reward_total, lifted_total = 0.0, 0.0

        # --- collect -------------------------------------------------------
        for _ in range(args_cli.steps):
            action, log_prob, value = policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Termination ends an episode; truncation merely cuts it off, and
            # only the former should stop the value bootstrapping.
            buffer.add(obs, action, log_prob, reward, terminated, value)

            obs = next_obs["policy"]
            reward_total += reward.mean().item()
            lifted_total += lifted_fraction(env, args_cli.success_height)

        with torch.no_grad():
            last_values = policy.value(obs)

        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones, last_values, cfg.gamma, cfg.gae_lambda
        )

        # --- update --------------------------------------------------------
        flat_obs, flat_actions, flat_log_probs, flat_adv, flat_returns = buffer.flatten(advantages, returns)
        flat_adv = normalize(flat_adv)  # once, over the whole batch

        batch_size = flat_obs.shape[0]
        minibatch = max(batch_size // cfg.minibatches, 1)
        stats: dict[str, float] = {}

        for _ in range(cfg.epochs_per_batch):
            order = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch):
                index = order[start : start + minibatch]
                loss, stats = ppo_losses(
                    policy,
                    flat_obs[index],
                    flat_actions[index],
                    flat_log_probs[index],
                    flat_adv[index],
                    flat_returns[index],
                    cfg,
                )
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimiser.step()

        record = {
            "iteration": iteration,
            "reward": reward_total / args_cli.steps,
            "lifted": lifted_total / args_cli.steps,
            **stats,
        }
        history.append(record)

        if iteration % args_cli.log_every == 0 or iteration == 1:
            elapsed = time.time() - started
            print(
                f"iter {iteration:4d}/{args_cli.iterations}  reward {record['reward']:7.3f}  "
                f"lifted {record['lifted']:.1%}  entropy {stats['entropy']:6.2f}  "
                f"kl {stats['approx_kl']:+.4f}  clip {stats['clip_fraction']:.2f}  {elapsed:.0f}s",
                flush=True,
            )
            torch.save({"config": cfg.__dict__, "state_dict": policy.state_dict()}, out_dir / "policy.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # Numbered checkpoints are kept rather than overwritten, so the run can
        # be replayed afterwards: render each one in turn and you have a video
        # of the policy learning, without having paid to render during training.
        if iteration % args_cli.checkpoint_every == 0:
            torch.save(
                {"config": cfg.__dict__, "state_dict": policy.state_dict(), "iteration": iteration},
                out_dir / f"policy_{iteration:06d}.pt",
            )

    print(f"\nfinished {args_cli.iterations} iterations in {time.time() - started:.0f}s")
    print(f"checkpoint {out_dir / 'policy.pt'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
