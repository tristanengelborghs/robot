"""Record scripted pick-and-lift demonstrations. Runs on the GPU box only.

Same recorder Isaac Lab uses, driven by :mod:`harness.lift_sm` instead of a
person, with two differences from scripts/record_scripted.py that come straight
out of reference/LIFT.md:

**We define success, because the task does not.** The lift task's terminations
are ``time_out`` and ``object_dropping``; there is no success term, so
``record_demos.py`` would mark nothing successful and export an empty file. What
counts as picking the block up is therefore our decision, and it is here: the
object above :data:`SUCCESS_HEIGHT`, held for several consecutive steps. The
task's own reward uses 0.04, which the cube clears by about 2 cm; we ask for
more, because a demonstration that clips the threshold on the way past is not a
demonstration of picking something up.

**The end-effector pose comes from the scene, not the observations.** The lift
task's policy group has no end-effector term. The ``ee_frame`` transformer knows
where the hand is -- the task's own rewards use it -- so the controller is fed
from ``env.scene`` directly.

The commanded goal pose is ignored throughout. The shipped task is "lift to a
commanded pose"; the task we want first is "pick it up".

Usage, inside the container::

    ./isaaclab.sh -p scripts/record_lift.py --headless --viz none --num_demos 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Our success height, in metres, in the environment frame. The cube rests at
# 0.021 (measured, see scripts/probe_lift.py), so this is ~8 cm of real lift.
SUCCESS_HEIGHT = 0.10

parser = argparse.ArgumentParser(description="Record scripted pick-and-lift demonstrations.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--dataset_file", type=str, default="/workspace/datasets/lift_scripted.hdf5")
parser.add_argument("--num_demos", type=int, default=100)
parser.add_argument("--max_attempts", type=int, default=0, help="Give up after this many episodes (0: num_demos * 3).")
parser.add_argument("--num_success_steps", type=int, default=10, help="Consecutive successful steps to conclude.")
parser.add_argument("--success_height", type=float, default=SUCCESS_HEIGHT)
parser.add_argument("--log_every", type=int, default=10, help="Print a progress line every N episodes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs with a live simulator."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.control import yaw_from_quat  # noqa: E402
from harness.lift_sm import LiftObservation, LiftStateMachine  # noqa: E402


def build_env_cfg(task: str, device: str, output_dir: str, dataset_name: str):
    """Configure the environment for recording.

    ``concatenate_terms`` is switched off so the dataset stores named
    observation terms. A concatenated vector would record fine and train fine,
    right up until someone needed to know which columns were the object.
    """
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    env_cfg.env_name = task.split(":")[-1]

    # No time limit: a scripted demonstration is finished when it succeeds or
    # when a state runs out of patience, not when 5 seconds are up.
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = dataset_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg


def observe(env) -> LiftObservation:
    """Read the controller's inputs out of the scene.

    Everything is shifted into the environment frame so a single environment at
    the origin and one offset in a grid behave identically.
    """
    origin = env.scene.env_origins[0]

    ee_frame = env.scene["ee_frame"]
    eef_pos = (ee_frame.data.target_pos_w.torch[0, 0, :] - origin).cpu().numpy()
    eef_quat = ee_frame.data.target_quat_w.torch[0, 0, :].cpu().numpy()

    obj = env.scene["object"]
    object_pos = (obj.data.root_pos_w.torch[0] - origin).cpu().numpy()
    object_quat = obj.data.root_quat_w.torch[0].cpu().numpy()

    return LiftObservation(
        eef_pos=eef_pos,
        eef_yaw=yaw_from_quat(eef_quat),
        object_pos=object_pos,
        object_yaw=yaw_from_quat(object_quat),
    )


def object_height(env) -> float:
    """Height of the object above its environment's origin."""
    return float(env.scene["object"].data.root_pos_w.torch[0, 2] - env.scene.env_origins[0, 2])


def export_successful_episode(env) -> None:
    """Close, label and write the episode currently being recorded."""
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes(
        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
    )
    env.recorder_manager.export_episodes([0])


def run_episode(env, success_height: float, num_success_steps: int):
    """Drive one episode, exporting it if the block ends up held in the air."""
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()

    controller = LiftStateMachine()
    consecutive = 0

    while not controller.finished:
        action = controller.step(observe(env))
        actions = torch.tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
        env.step(actions)

        if object_height(env) > success_height:
            consecutive += 1
            if consecutive >= num_success_steps:
                export_successful_episode(env)
                outcome = controller.outcome()
                outcome.succeeded = True
                return outcome
        else:
            consecutive = 0

        if env.sim.is_stopped():
            break

    outcome = controller.outcome()
    outcome.succeeded = False
    return outcome


def main() -> None:
    dataset_path = Path(args_cli.dataset_file)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = build_env_cfg(args_cli.task, args_cli.device, str(dataset_path.parent), dataset_path.stem)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    max_attempts = args_cli.max_attempts or args_cli.num_demos * 3
    recorded = 0
    attempts = 0
    failures: list[str] = []

    with torch.inference_mode():
        while recorded < args_cli.num_demos and attempts < max_attempts:
            attempts += 1
            outcome = run_episode(env, args_cli.success_height, args_cli.num_success_steps)
            recorded += int(outcome.succeeded)

            if not outcome.succeeded:
                # The state that ran out of patience is the whole diagnosis.
                last = outcome.transitions[-1] if outcome.transitions else None
                failures.append(last.reason if last else "no transitions")
                print(f"[episode {attempts}] FAILED after {outcome.steps} steps: {failures[-1]}", flush=True)
            elif recorded % args_cli.log_every == 0 or recorded == 1:
                print(
                    f"[episode {attempts}] {recorded}/{args_cli.num_demos} recorded, "
                    f"{outcome.steps} steps, peak {outcome.peak_object_height:.3f} m",
                    flush=True,
                )

    rate = recorded / attempts if attempts else 0.0
    print(f"\nrecorded {recorded}/{args_cli.num_demos} in {attempts} attempts ({rate:.0%}) -> {dataset_path}")
    if failures:
        print(f"{len(failures)} failure(s); most recent: {failures[-1]}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
