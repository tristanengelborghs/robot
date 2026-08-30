"""Measure the lift task's resting state, because its numbers do not add up on paper.

``object_is_lifted`` rewards the object being above ``z = 0.04``, and the cube is
spawned at ``z = 0.055`` on a table whose top is at ``z = 0``. Read literally
that means the largest shaped reward in the task is paid in full for a cube
nobody has touched. Either the cube settles lower than it spawns, or the
threshold is doing nothing.

Which of those is true decides what "lifted" has to mean in the success
termination we write, so it is measured rather than reasoned about. This also
prints the observation layout and the end-effector pose, which a scripted
controller needs and the observations do not carry.

Usage, inside the container::

    ./isaaclab.sh -p scripts/probe_lift.py --headless --viz none
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the lift task's resting state.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--settle-steps", type=int, default=60, help="Zero-action steps before measuring.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs with a live simulator."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=4)
    env_cfg.observations.policy.concatenate_terms = False  # so terms can be named
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    with torch.inference_mode():
        obs, _ = env.reset()

        # Let the cube settle under gravity with the arm holding still.
        zero = torch.zeros(env.action_space.shape, device=env.device)
        for _ in range(args_cli.settle_steps):
            obs = env.step(zero)[0]

        origins = env.scene.env_origins
        object_pos = env.scene["object"].data.root_pos_w.torch - origins
        ee_pos = env.scene["ee_frame"].data.target_pos_w.torch[:, 0, :] - origins

        print("\n" + "=" * 68)
        print("LIFT TASK, MEASURED AT REST")
        print("=" * 68)

        print(f"\ncontrol rate       {1.0 / (env_cfg.sim.dt * env_cfg.decimation):.0f} Hz")
        print(f"episode length     {env_cfg.episode_length_s}s")

        print(f"\nobject z at rest   {object_pos[:, 2].tolist()}")
        print(f"object xy at rest  {object_pos[0, :2].tolist()}")
        print(f"end-effector       {ee_pos[0].tolist()}")

        # The question this script exists for.
        threshold = 0.04
        already = (object_pos[:, 2] > threshold).float().mean().item()
        print(f"\nobject_is_lifted fires at rest for {already * 100:.0f}% of envs (threshold {threshold})")
        if already > 0:
            print("  -> the 15.0-weight lift bonus is paid for doing nothing.")
            print("     Our success termination must use a height above the resting one.")
        else:
            print("  -> the threshold is meaningful as shipped.")

        print("\nobservation terms:")
        for name, value in obs["policy"].items():
            print(f"  {name:<28} {tuple(value.shape)}")

        print(f"\naction space       {env.action_space.shape}")
        print("=" * 68 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
