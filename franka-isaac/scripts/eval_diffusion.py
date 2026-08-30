"""Roll a trained diffusion policy out in the simulator and count successes.

Runs on the GPU box. A denoising loss says how well the policy predicts noise on
held-out demonstrations; it says nothing about whether the block ends up in the
air. This is the number that means something, and it is the same number the RL
side will be judged by, so both are comparable.

    ./isaaclab.sh -p scripts/eval_diffusion.py --headless --viz none \\
        --checkpoint /workspace/runs/diffusion/policy.pt --episodes 50

**Receding horizon.** The policy predicts ``horizon`` actions from one
observation; this executes the first ``--execute`` of them and then re-plans.
Executing the whole chunk would be open-loop for its full length and drift;
re-planning every single step throws away most of what makes chunked prediction
smooth, and costs a full denoising pass per control step. Executing about half
is the usual compromise and is what the paper does.

Episodes run in parallel: rollouts are the expensive part of any evaluation, and
the simulator is a vectorised environment whether or not we use it as one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a diffusion policy in the simulator.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--checkpoint", type=str, default="/workspace/runs/diffusion/policy.pt")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--execute", type=int, default=4, help="Actions executed per plan, before re-planning.")
parser.add_argument("--max_steps", type=int, default=400, help="Give up on an episode after this many steps.")
parser.add_argument("--success_height", type=float, default=0.10)
parser.add_argument("--success_steps", type=int, default=10)
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

from harness.diffusion import NoiseSchedule, sample_actions  # noqa: E402
from harness.training import load_checkpoint  # noqa: E402


def build_observation(obs: dict, obs_keys: list[str]) -> torch.Tensor:
    """Assemble the policy's input in exactly the training order.

    Getting this order wrong is the classic silent evaluation bug: the policy
    runs, produces confident nonsense, and the failure looks like bad training.
    The key list travels in the checkpoint for that reason.
    """
    policy_obs = obs["policy"]
    missing = [key for key in obs_keys if key not in policy_obs]
    if missing:
        raise KeyError(f"the environment has no observation term(s) {missing}; it has {sorted(policy_obs.keys())}")

    return torch.cat([policy_obs[key] for key in obs_keys], dim=1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, obs_normalizer, action_normalizer, extra = load_checkpoint(args_cli.checkpoint, device=device)
    policy.eval()

    obs_keys = extra.get("obs_keys")
    if not obs_keys:
        raise ValueError("the checkpoint does not record which observation terms it was trained on")

    schedule = NoiseSchedule(policy.cfg.num_steps).to(torch.device(device))

    obs_mean = torch.as_tensor(obs_normalizer.mean, dtype=torch.float32, device=device)
    obs_scale = torch.as_tensor(obs_normalizer.scale, dtype=torch.float32, device=device)
    action_mean = torch.as_tensor(action_normalizer.mean, dtype=torch.float32, device=device)
    action_scale = torch.as_tensor(action_normalizer.scale, dtype=torch.float32, device=device)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None  # our step budget governs, not the task's
    env_cfg.observations.policy.concatenate_terms = False
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    print(f"\npolicy trained on {obs_keys}")
    print(f"evaluating {args_cli.episodes} episodes, {args_cli.num_envs} at a time\n", flush=True)

    successes, attempted = 0, 0

    with torch.inference_mode():
        while attempted < args_cli.episodes:
            obs, _ = env.reset()
            batch = min(args_cli.num_envs, args_cli.episodes - attempted)

            held = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

            for _ in range(0, args_cli.max_steps, args_cli.execute):
                inputs = (build_observation(obs, obs_keys) - obs_mean) / obs_scale
                chunk = sample_actions(policy, schedule, inputs) * action_scale + action_mean

                for step in range(min(args_cli.execute, policy.cfg.horizon)):
                    obs = env.step(chunk[:, step, :])[0]

                    height = env.scene["object"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                    held = torch.where(height > args_cli.success_height, held + 1, torch.zeros_like(held))
                    done |= held >= args_cli.success_steps

                if bool(done[:batch].all()):
                    break

            successes += int(done[:batch].sum().item())
            attempted += batch
            print(f"  {successes}/{attempted} so far", flush=True)

    rate = successes / max(attempted, 1)
    print(f"\nsuccess rate {successes}/{attempted} = {rate:.0%}")
    print(f"(lifted above {args_cli.success_height} m and held for {args_cli.success_steps} steps)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
