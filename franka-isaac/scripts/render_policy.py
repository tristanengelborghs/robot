"""Roll a saved policy out and record it as video. Runs on the GPU box.

Rendering during training would be the obvious way to watch a policy learn, and
it is the wrong one: Isaac Lab's ``--video`` needs ``--enable_cameras``, which
turns rendering on for every step of an hours-long run with a thousand
environments, to produce clips somebody watches for two minutes.

So training stays headless and fast, keeps a numbered checkpoint every so often,
and this script renders those checkpoints afterwards. Point it at a directory
and it renders each one in order, which is a film of the policy learning for the
cost of a few minutes of GPU rather than a tax on the whole run.

    ./isaaclab.sh -p scripts/render_policy.py --enable_cameras \\
        --checkpoint /workspace/runs/ppo/policy_000200.pt

    ./isaaclab.sh -p scripts/render_policy.py --enable_cameras \\
        --checkpoint-dir /workspace/runs/ppo        # every checkpoint, in order

Both policy kinds are handled. They are told apart by what is in the file rather
than by a flag, because a flag is one more thing to get wrong and the file
already knows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record video of a saved policy acting.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="A single checkpoint to render.")
parser.add_argument("--checkpoint-dir", type=str, default=None, help="Render every policy_*.pt here, in order.")
parser.add_argument("--out", type=str, default="/workspace/runs/videos")
parser.add_argument("--num_envs", type=int, default=4, help="Few: they are tiled into one frame.")
parser.add_argument("--steps", type=int, default=300, help="Frames to record per checkpoint.")
parser.add_argument("--execute", type=int, default=4, help="Diffusion only: actions run per plan.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Rendering is the entire point here, so cameras are forced on rather than left
# to be forgotten -- without them the recorder writes empty files.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs with a live simulator."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.diffusion import NoiseSchedule, sample_actions  # noqa: E402
from harness.ppo import ActorCritic, PPOConfig  # noqa: E402
from harness.training import load_checkpoint  # noqa: E402


def checkpoints_to_render() -> list[Path]:
    """The checkpoints asked for, in training order."""
    if args_cli.checkpoint:
        return [Path(args_cli.checkpoint)]
    if not args_cli.checkpoint_dir:
        raise SystemExit("give either --checkpoint or --checkpoint-dir")

    found = sorted(Path(args_cli.checkpoint_dir).glob("policy_*.pt"))
    if not found:
        raise SystemExit(f"no policy_*.pt files in {args_cli.checkpoint_dir}")
    return found


def load_any_policy(path: Path, device: str):
    """Load either kind of policy, deciding from the file's contents.

    A diffusion checkpoint carries its normalisers; a PPO one carries a
    PPOConfig. Sniffing that is more robust than a flag the caller has to
    remember to match to the file.
    """
    payload = torch.load(path, map_location=device, weights_only=False)

    if "obs_mean" in payload:
        policy, obs_norm, action_norm, extra = load_checkpoint(path, device=device)
        policy.eval()
        schedule = NoiseSchedule(policy.cfg.num_steps).to(torch.device(device))

        obs_mean = torch.as_tensor(obs_norm.mean, dtype=torch.float32, device=device)
        obs_scale = torch.as_tensor(obs_norm.scale, dtype=torch.float32, device=device)
        action_mean = torch.as_tensor(action_norm.mean, dtype=torch.float32, device=device)
        action_scale = torch.as_tensor(action_norm.scale, dtype=torch.float32, device=device)
        keys = extra["obs_keys"]

        def act(obs_dict, plan_state):
            """Receding horizon: re-plan when the current chunk runs out."""
            chunk, index = plan_state
            if chunk is None or index >= min(args_cli.execute, policy.cfg.horizon):
                inputs = torch.cat([obs_dict["policy"][k] for k in keys], dim=1)
                normalised = (inputs - obs_mean) / obs_scale
                chunk = sample_actions(policy, schedule, normalised) * action_scale + action_mean
                index = 0
            return chunk[:, index, :], (chunk, index + 1)

        return act, "diffusion", True

    cfg = PPOConfig(**payload["config"])
    policy = ActorCritic(cfg).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()

    def act(obs_dict, plan_state):
        # The mean action, not a sample: this is a recording of what the policy
        # has learned, not of its exploration noise.
        return policy.actor(obs_dict["policy"]), plan_state

    return act, "ppo", False


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None

    for path in checkpoints_to_render():
        act, kind, needs_named_terms = load_any_policy(path, device)

        # A diffusion policy picks observation terms by name; PPO takes the
        # concatenated vector it was trained on.
        env_cfg.observations.policy.concatenate_terms = not needs_named_terms

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(out_dir),
            name_prefix=path.stem,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.steps,
            disable_logger=True,
        )

        print(f"\nrendering {kind} policy {path.name} -> {out_dir}", flush=True)

        obs, _ = env.reset()
        plan_state = (None, 0)
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                action, plan_state = act(obs, plan_state)
                obs = env.step(action)[0]

        env.close()

    print(f"\nvideos in {out_dir}")
    print("pull them with `make pull-runs` on the laptop.")


if __name__ == "__main__":
    main()
    simulation_app.close()
