"""Dump everything static about a stock Isaac Lab task to files in the repo.

Component 1 of the plan is "inspect, change nothing": robot model, assets,
observation and action spaces, control frequency, reward, reset, success
condition, contact configuration. All of that is fixed by the task's config
classes, so it should be readable on a laptop with no GPU and no instance —
which means dumping it once, into the repo, rather than re-renting a GPU every
time someone asks what the action space is.

Runs INSIDE the Isaac Lab container:

    ./isaaclab.sh -p /workspace/franka-isaac/scripts/inspect_env.py

Every section is independently guarded: a config attribute that moved between
Isaac Lab versions costs you that one line, not the whole dump.
"""

from __future__ import annotations

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--out", default="/workspace/franka-isaac/reference")
# AppLauncher adds --headless, --viz and friends.
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402  (registers the tasks)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def safe(label, fn, into: dict):
    """Record fn()'s value under label, or the reason it could not be read."""
    try:
        into[label] = fn()
    except Exception as exc:  # noqa: BLE001 - a missing attr must not abort the dump
        into[label] = f"<unavailable: {type(exc).__name__}: {exc}>"


def as_plain(obj, depth=0):
    """Config objects -> JSON-able nesting, shallow enough to stay readable."""
    if depth > 4:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [as_plain(o, depth + 1) for o in obj]
    if isinstance(obj, dict):
        return {str(k): as_plain(v, depth + 1) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {
            k: as_plain(v, depth + 1)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return str(obj)


def main() -> None:
    os.makedirs(args.out, exist_ok=True)
    report: dict = {"task": args.task}

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    safe("observation_space", lambda: str(env.observation_space), report)
    safe("action_space", lambda: str(env.action_space), report)
    safe("num_observations", lambda: env_cfg.observation_space, report)
    safe("num_actions", lambda: env_cfg.action_space, report)

    # Control frequency: physics runs at 1/dt, the policy acts every
    # `decimation` physics steps.
    safe("sim_dt", lambda: env_cfg.sim.dt, report)
    safe("decimation", lambda: env_cfg.decimation, report)
    safe(
        "control_hz",
        lambda: round(1.0 / (env_cfg.sim.dt * env_cfg.decimation), 3),
        report,
    )
    safe("physics_hz", lambda: round(1.0 / env_cfg.sim.dt, 1), report)
    safe("episode_length_s", lambda: env_cfg.episode_length_s, report)

    safe("robot_cfg", lambda: as_plain(env_cfg.robot), report)
    safe("task_cfg", lambda: as_plain(env_cfg.task), report)
    safe("scene_cfg", lambda: as_plain(env_cfg.scene), report)
    safe("sim_cfg", lambda: as_plain(env_cfg.sim), report)

    safe(
        "robot_joint_names",
        lambda: list(unwrapped.scene["robot"].joint_names),
        report,
    )
    safe(
        "robot_body_names",
        lambda: list(unwrapped.scene["robot"].body_names),
        report,
    )
    safe("scene_keys", lambda: list(unwrapped.scene.keys()), report)

    # Whatever the cfg calls the assets varies by task; try the Factory names.
    for name in ("held_asset", "fixed_asset", "held_asset_cfg", "fixed_asset_cfg"):
        if hasattr(env_cfg, name):
            safe(f"asset::{name}", lambda n=name: as_plain(getattr(env_cfg, n)), report)

    with open(os.path.join(args.out, "peginsert_env.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    lines = [f"# {args.task}", "", "Dumped by `scripts/inspect_env.py`. Do not edit by hand.", ""]
    for key, value in report.items():
        lines.append(f"## {key}\n")
        if isinstance(value, (dict, list)):
            lines.append("```json")
            lines.append(json.dumps(value, indent=2, default=str))
            lines.append("```\n")
        else:
            lines.append(f"`{value}`\n")
    with open(os.path.join(args.out, "peginsert_env.md"), "w") as fh:
        fh.write("\n".join(lines))

    print(f"[inspect] wrote peginsert_env.json and .md to {args.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
