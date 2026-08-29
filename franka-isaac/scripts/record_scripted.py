"""Record demonstrations from the scripted controller instead of a human.

Runs on the GPU box only. This is Isaac Lab's own ``scripts/tools/record_demos.py``
with the teleoperation device replaced by :mod:`harness.stack_sm` -- same
environment configuration, same ``RecorderManager``, same HDF5 on the way out --
so proving the recording pipeline does not require a person at a keyboard
driving a rented instance over a video stream.

Everything about the environment setup below is copied deliberately from
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
        --num_demos 5 --dataset_file ./datasets/stack_scripted.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record scripted demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--dataset_file", type=str, default="./datasets/stack_scripted.hdf5", help="Where to export.")
parser.add_argument("--num_demos", type=int, default=5, help="Number of successful demonstrations to record.")
parser.add_argument("--max_attempts", type=int, default=0, help="Give up after this many episodes (0: num_demos * 4).")
parser.add_argument("--num_success_steps", type=int, default=10, help="Consecutive successful steps to conclude a demo.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the reset randomisation.")
parser.add_argument("--log_transitions", action="store_true", help="Print the state-transition log for every episode.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The simulator has to exist before anything that touches it can be imported.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402  (registers the stock tasks)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.stack_sm import EpisodeLog, Observation, StackStateMachine, yaw_from_quat  # noqa: E402

# Layout of the `object` observation term, from the task's mdp.object_obs: three
# blocks of (position, quaternion), one per cube, positions already relative to
# the environment origin.
CUBE_BLOCK = 7


def observe(obs: dict) -> Observation:
    """Pull the controller's inputs out of one policy observation group."""
    policy = obs["policy"]
    # The first three (position, quaternion) blocks; the rest of the term is
    # relative vectors this controller derives for itself.
    poses = policy["object"][0].cpu().numpy().reshape(-1)[: 3 * CUBE_BLOCK].reshape(3, CUBE_BLOCK)

    return Observation(
        eef_pos=policy["eef_pos"][0].cpu().numpy(),
        eef_yaw=yaw_from_quat(policy["eef_quat"][0].cpu().numpy()),
        cube_pos=poses[:, :3].copy(),
        cube_yaw=np.array([yaw_from_quat(row[3:7]) for row in poses]),
    )


def build_env_cfg(output_dir: str, output_file_name: str):
    """Environment configuration for recording -- see the module docstring."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]

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
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def run_episode(env, success_term, log: EpisodeLog) -> bool:
    """Drive one episode to success or failure. Exports it if it succeeded."""
    env.sim.reset()
    env.recorder_manager.reset()
    obs, _ = env.reset()

    sm = StackStateMachine()
    success_steps = 0

    while not sm.finished:
        action = sm.step(observe(obs))
        actions = torch.tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
        obs = env.step(actions)[0]

        if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
            success_steps += 1
            if success_steps >= args_cli.num_success_steps:
                # The same three calls record_demos.py makes: close the episode,
                # label it, write it out.
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                log.transitions = sm.transitions
                log.steps = sm.step_count
                log.succeeded = True
                return True
        else:
            success_steps = 0

        if env.sim.is_stopped():
            break

    log.transitions = sm.transitions
    log.steps = sm.step_count
    log.succeeded = False
    return False


def main() -> None:
    dataset_path = Path(args_cli.dataset_file)
    output_dir = str(dataset_path.parent)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg, success_term = build_env_cfg(output_dir, dataset_path.stem)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.seed is not None:
        env.seed(args_cli.seed)

    max_attempts = args_cli.max_attempts or args_cli.num_demos * 4
    recorded = 0
    attempts = 0

    with torch.inference_mode():
        while recorded < args_cli.num_demos and attempts < max_attempts:
            attempts += 1
            log = EpisodeLog()
            ok = run_episode(env, success_term, log)
            recorded += int(ok)
            print(f"[episode {attempts}] {'success' if ok else 'failed '} -- {log.steps} steps, {recorded} recorded")
            if args_cli.log_transitions or not ok:
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
