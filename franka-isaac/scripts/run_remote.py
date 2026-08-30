"""Run one of the long GPU-host commands and stream its output.

The commands themselves are built in :mod:`harness.remote`, where they are unit
tested without a GPU, a network or a running instance. This module only runs
them: it prints what it ran, waits, and makes sure nothing is left simulating on
the box afterwards.

That last part is not politeness. A ``docker exec`` killed from this end kills
only the local client; the simulator inside the container keeps running, and the
instance keeps billing until someone notices.

    PYTHONPATH=. .venv/bin/python scripts/run_remote.py record --num-demos 5
    PYTHONPATH=. .venv/bin/python scripts/run_remote.py teleop --num-demos 5
    PYTHONPATH=. .venv/bin/python scripts/run_remote.py replay
    PYTHONPATH=. .venv/bin/python scripts/run_remote.py tunnel
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from harness import remote

# Every one of these includes Isaac Sim's start-up, which is minutes on its own.
# A teleoperation session is bounded by how long a person keeps driving, so it
# gets the longest leash.
RECORD_TIMEOUT_S = 3600
TELEOP_TIMEOUT_S = 7200
REPLAY_TIMEOUT_S = 1800

# The exit code shells conventionally use for "killed by a timeout".
TIMED_OUT = 124


def run_on_the_box(command: str, *, label: str, timeout_s: int, cleanup_script: str) -> int:
    """Run one command on the GPU host, then clean up if it went wrong.

    Args:
        command: The full ``ssh ... docker exec ... isaaclab.sh ...`` string.
        label: Name used in this module's own log lines.
        timeout_s: How long to wait before giving up.
        cleanup_script: The script name to ``pkill`` inside the container if the
            run did not end cleanly.

    Returns:
        The command's exit code, or 124 if it timed out.
    """
    print(f"$ {command}\n", flush=True)

    try:
        exit_code = subprocess.run(command, shell=True, timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        print(f"\n[{label}] no output for {timeout_s}s -- giving up", file=sys.stderr, flush=True)
        exit_code = TIMED_OUT

    if exit_code != 0:
        # Only on a bad exit. A clean run has already closed the simulator, and
        # a pkill afterwards would report a kill that did not happen.
        print(f"[{label}] exit {exit_code} -- killing anything left in the container", flush=True)
        subprocess.run(remote.kill_sim(cleanup_script), shell=True)

    return exit_code


def run_record(args: argparse.Namespace) -> int:
    """Record demonstrations from the scripted controller, unattended."""
    extra = ("--log_transitions",) if args.log_transitions else ()
    command = remote.record_scripted(
        num_demos=args.num_demos,
        dataset_file=args.dataset_file,
        extra=extra,
    )
    return run_on_the_box(
        command,
        label="record",
        timeout_s=RECORD_TIMEOUT_S,
        cleanup_script="record_scripted.py",
    )


def run_teleop(args: argparse.Namespace) -> int:
    """Record demonstrations by hand, driven through the browser viewer."""
    print_teleop_instructions(args)
    command = remote.record_teleop(
        num_demos=args.num_demos,
        dataset_file=args.dataset_file,
        device=args.device,
    )
    return run_on_the_box(
        command,
        label="teleop",
        timeout_s=TELEOP_TIMEOUT_S,
        cleanup_script="record_demos.py",
    )


def run_replay(args: argparse.Namespace) -> int:
    """Replay a recorded dataset and validate it against the recorded states."""
    command = remote.replay(dataset_file=args.dataset_file)
    return run_on_the_box(
        command,
        label="replay",
        timeout_s=REPLAY_TIMEOUT_S,
        cleanup_script="replay_demos.py",
    )


def run_tunnel(_args: argparse.Namespace) -> int:
    """Forward the browser viewer to this laptop until interrupted.

    This one deliberately does not go through :func:`run_on_the_box`: it has no
    simulator to clean up, no useful timeout, and ending it with ctrl-C is the
    normal way to finish rather than a failure.
    """
    command = remote.tunnel()
    print(f"$ {command}\n", flush=True)
    print(f"viewer:  {remote.viewer_url()}")
    print("ctrl-C closes the tunnel.\n", flush=True)

    try:
        return subprocess.run(command, shell=True).returncode
    except KeyboardInterrupt:
        print("\n[tunnel] closed", flush=True)
        return 0


KEYBOARD_BINDINGS = """  Keyboard (Isaac Lab's Se3Keyboard):

      W / S     end-effector +x / -x        Z / X   roll  + / -
      A / D     end-effector +y / -y        T / G   pitch + / -
      Q / E     end-effector +z / -z        C / V   yaw   + / -

      K         toggle the gripper open and closed
      L         recentre the teleoperation device
      R         abandon this episode and reset the scene"""

GAMEPAD_BINDINGS = """  Gamepad (Isaac Lab's Se3Gamepad; a DualSense works, the browser
  forwards it):

      Left stick            end-effector x / y
      Right stick up-down   end-effector z
      Right stick left-right    yaw
      X button              toggle the gripper open and closed

  The browser only sees a controller once you press a button on it with the
  page focused. Sticks are proportional, so sensitivity is a matter of feel:
  GAMEPAD_POS_SENSITIVITY and GAMEPAD_ROT_SENSITIVITY tune it between runs."""

BINDINGS = {"keyboard": KEYBOARD_BINDINGS, "gamepad": GAMEPAD_BINDINGS}


def print_teleop_instructions(args: argparse.Namespace) -> None:
    """Everything needed to actually drive the arm, printed before it starts."""
    bindings = BINDINGS.get(args.device, f"  Device: {args.device}")
    print(
        f"""
Teleoperated recording -- {args.num_demos} demonstrations to {args.dataset_file}

  1. Open the instance's Brev URL with /viewer on the end, from the console's
     Access tab. `make tunnel` serves the same page over ssh, but the video is
     WebRTC over UDP and will not survive the tunnel.
  2. Click the viewport so it has input focus.
  3. Drive the arm.

{bindings}

  An episode ends and is written out on its own once the cubes are stacked;
  only successful episodes are exported. Stacking order is cube_1 at the
  bottom, then cube_2, then cube_3 -- blue, red, green.

  Isaac Sim takes a few minutes to start before the viewer shows anything.
""",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    record = subcommands.add_parser("record", help="record scripted demonstrations into an HDF5 dataset")
    record.add_argument("--num-demos", type=int, default=5)
    record.add_argument("--dataset-file", default=remote.DATASET_FILE)
    record.add_argument("--log-transitions", action="store_true", help="print the state machine's transition log")
    record.set_defaults(handler=run_record)

    teleop = subcommands.add_parser("teleop", help="record demonstrations by hand over the browser viewer")
    teleop.add_argument("--num-demos", type=int, default=5)
    teleop.add_argument("--dataset-file", default=remote.TELEOP_DATASET_FILE)
    teleop.add_argument("--device", default="keyboard", choices=["keyboard", "gamepad", "spacemouse"])
    teleop.set_defaults(handler=run_teleop)

    replay = subcommands.add_parser("replay", help="replay a dataset and validate it against the recorded states")
    replay.add_argument("--dataset-file", default=remote.DATASET_FILE)
    replay.set_defaults(handler=run_replay)

    tunnel = subcommands.add_parser("tunnel", help="forward the browser viewer to this laptop")
    tunnel.set_defaults(handler=run_tunnel)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
