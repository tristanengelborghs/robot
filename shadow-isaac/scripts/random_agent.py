"""Random agent against this project's task.

Isaac Lab's own `scripts/environments/random_agent.py` cannot run our task, and
the failure is confusing rather than obvious: `NameNotFound: Environment
Catch-Shadow-Direct doesn't exist`. Registration happens in
`catching/tasks/direct/catch/__init__.py`, and that module only executes if
something imports it. Isaac Lab's scripts import `isaaclab_tasks`, which
registers the stock tasks and knows nothing about an external extension --
installing the package is not enough, because installing does not import.

Hence a project-local script, which is what Isaac Lab's own template generator
emits for external projects too.

The import order below is not stylistic. Almost every `isaaclab` module expects
the simulation app to exist, so nothing from isaaclab (and therefore nothing
from `catching`, which imports it) may be imported before `AppLauncher` has
started. That is why the imports are split around the launcher and why ruff's
E402 is disabled for this file.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Catch-Shadow-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import catching  # noqa: E402, F401  -- registers Catch-Shadow-Direct-v0


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[smoke] {args_cli.task}: obs {env.observation_space}, act {env.action_space}")

    env.reset()
    reasons: dict[str, int] = {}
    for i in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        action = 2.0 * torch.rand(env.unwrapped.num_envs,
                                  env.unwrapped.cfg.action_space,
                                  device=env.unwrapped.device) - 1.0
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        if done.any():
            # The env exposes why it ended; a success rate with no failure
            # breakdown says nothing about what to fix.
            for code in env.unwrapped._reason[done].tolist():
                name = ["running", "dropped", "out_of_bounds", "excess_force",
                        "caught", "never_launched", "timeout"][code]
                reasons[name] = reasons.get(name, 0) + 1
        if i % 50 == 0:
            print(f"[smoke] step {i:4d}  reward {reward.mean().item():+.3f}")

    print(f"[smoke] episode endings over {args_cli.steps} steps: {reasons or 'none'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
