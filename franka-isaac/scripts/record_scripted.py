"""Record demonstrations from the scripted controller instead of a human.

Runs on the GPU box only. This is Isaac Lab's own ``scripts/tools/record_demos.py``
with the teleoperation device replaced by :mod:`harness.stack_sm` -- the same
environment configuration, the same ``RecorderManager``, the same HDF5 on the way
out -- so proving the recording pipeline does not need a person at a keyboard
driving a rented instance over a video stream.

Every choice in :func:`build_env_cfg` is copied deliberately from
record_demos.py, because a dataset recorded under a different configuration is
not the dataset a human teleoperator would have produced:

* one environment, because the recorder exports episode ``[0]``
* ``terminations.success`` pulled out of the config and evaluated by hand, so a
  success ends the *recording* rather than the episode
* ``terminations.time_out`` disabled, so a slow demonstration is not truncated
* ``observations.policy.concatenate_terms = False``, so the dataset stores named
  observation terms rather than one anonymous vector
* ``DatasetExportMode.EXPORT_SUCCEEDED_ONLY``, so a failed attempt costs GPU
  time and nothing else

Usage, inside the container::

    ./isaaclab.sh -p scripts/record_scripted.py --headless --viz none \\
        --num_demos 5 --dataset_file /workspace/datasets/stack_scripted.hdf5

The argument parsing and the ``AppLauncher`` call below have to happen at import
time, before anything that touches the simulator is imported. That is Isaac Lab's
structure, not a choice made here -- but it is the only part of this file that
depends on module-level state. Everything below takes what it needs as an
argument, so it can be read, and reasoned about, on its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record scripted demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-v0", help="Name of the task.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="/workspace/datasets/stack_scripted.hdf5",
    help="Where to export the recorded demonstrations.",
)
parser.add_argument("--num_demos", type=int, default=5, help="Number of successful demonstrations to record.")
parser.add_argument("--max_attempts", type=int, default=0, help="Give up after this many episodes (0: num_demos * 4).")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Consecutive steps meeting the success condition before a demo is concluded.",
)
parser.add_argument("--log_transitions", action="store_true", help="Print the state-transition log for every episode.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The simulator has to exist before anything that touches it can be imported.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs with a live simulator."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402  (importing registers the stock tasks)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.stack_sm import EpisodeLog, Observation, StackStateMachine, observation_from_arrays  # noqa: E402


def observe(obs: dict) -> Observation:
    """Move one policy observation group from the GPU to the controller.

    This function is the whole GPU-side half of reading an observation: take the
    single environment's row of each term off the device. What those numbers
    *mean* is worked out in :func:`harness.stack_sm.observation_from_arrays`,
    where it is tested without a simulator.
    """
    policy = obs["policy"]
    return observation_from_arrays(
        eef_pos=policy["eef_pos"][0].cpu().numpy(),
        eef_quat=policy["eef_quat"][0].cpu().numpy(),
        object_term=policy["object"][0].cpu().numpy(),
    )


def build_env_cfg(task: str, device: str, output_dir: str, dataset_name: str):
    """Configure the environment for recording. See the module docstring.

    Returns:
        The environment config, and the success termination term lifted out of
        it. The term is evaluated by hand in :func:`run_episode` so that meeting
        the success condition ends the recording rather than the episode.
    """
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    env_cfg.env_name = task.split(":")[-1]

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        print("[warn] no success termination in this task: demos cannot be marked successful")

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = dataset_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def is_successful(env, success_term) -> bool:
    """Has the task's own success condition been met on this step?"""
    if success_term is None:
        return False
    return bool(success_term.func(env, **success_term.params)[0])


def export_successful_episode(env) -> None:
    """Close, label and write out the episode currently being recorded.

    These are the same three calls record_demos.py makes when a human's
    demonstration succeeds.
    """
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
    env.recorder_manager.export_episodes([0])


def run_episode(env, success_term, num_success_steps: int) -> EpisodeLog:
    """Drive one episode to success or failure, exporting it if it succeeds."""
    env.sim.reset()
    env.recorder_manager.reset()
    obs, _ = env.reset()

    controller = StackStateMachine()
    consecutive_successes = 0

    while not controller.finished:
        action = controller.step(observe(obs))
        actions = torch.tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
        obs = env.step(actions)[0]

        if is_successful(env, success_term):
            consecutive_successes += 1
            if consecutive_successes >= num_success_steps:
                export_successful_episode(env)
                return EpisodeLog(controller.transitions, controller.step_count, succeeded=True)
        else:
            consecutive_successes = 0

        if env.sim.is_stopped():
            break

    return EpisodeLog(controller.transitions, controller.step_count, succeeded=False)


def main() -> None:
    dataset_path = Path(args_cli.dataset_file)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg, success_term = build_env_cfg(
        task=args_cli.task,
        device=args_cli.device,
        output_dir=str(dataset_path.parent),
        dataset_name=dataset_path.stem,
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    max_attempts = args_cli.max_attempts or args_cli.num_demos * 4
    recorded = 0
    attempts = 0

    with torch.inference_mode():
        while recorded < args_cli.num_demos and attempts < max_attempts:
            attempts += 1
            log = run_episode(env, success_term, args_cli.num_success_steps)
            recorded += int(log.succeeded)

            outcome = "success" if log.succeeded else "failed "
            print(f"[episode {attempts}] {outcome} -- {log.steps} steps, {recorded} recorded")
            if args_cli.log_transitions or not log.succeeded:
                # A failure's transition log is the whole diagnostic: it names
                # the state that ran out of patience.
                print(log.format())

    print(f"\nrecorded {recorded}/{args_cli.num_demos} demonstrations in {attempts} attempts -> {dataset_path}")
    if recorded < args_cli.num_demos:
        print("[warn] gave up before reaching the requested count")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
