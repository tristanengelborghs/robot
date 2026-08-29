"""Run one of the long GPU-host commands and stream its output.

The commands themselves are built in :mod:`harness.remote`, where they are unit
tested; this only runs them, prints what it ran, and makes sure nothing is left
simulating on the instance afterwards. That last part is not politeness: a
``docker exec`` killed from this end leaves the simulator running inside the
container, and the container keeps billing.

    PYTHONPATH=. .venv/bin/python scripts/run_remote.py record --num-demos 5
    PYTHONPATH=. .venv/bin/python scripts/run_remote.py replay
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from harness import remote

# Recording five demonstrations includes Isaac Sim's start-up, which is minutes
# on its own; replay is shorter but loads the same simulator.
TIMEOUTS = {"record": 3600, "replay": 1800}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="record scripted demonstrations into an HDF5 dataset")
    record.add_argument("--num-demos", type=int, default=5)
    record.add_argument("--dataset-file", default=remote.DATASET_FILE)
    record.add_argument("--log-transitions", action="store_true")

    replay = sub.add_parser("replay", help="replay a recorded dataset and validate it against the original states")
    replay.add_argument("--dataset-file", default=remote.DATASET_FILE)

    args = parser.parse_args()

    if args.command == "record":
        extra = ("--log_transitions",) if args.log_transitions else ()
        cmd = remote.record_scripted(num_demos=args.num_demos, dataset_file=args.dataset_file, extra=extra)
        script_name = "record_scripted.py"
    else:
        cmd = remote.replay(dataset_file=args.dataset_file)
        script_name = "replay_demos.py"

    print(f"$ {cmd}\n", flush=True)
    try:
        code = subprocess.run(cmd, shell=True, timeout=TIMEOUTS[args.command]).returncode
    except subprocess.TimeoutExpired:
        print(f"\n[{args.command}] timed out", file=sys.stderr, flush=True)
        code = 124

    if code != 0:
        # Only on a bad exit: a clean run has already closed the simulator, and
        # a stray pkill would be a lie about what happened.
        print(f"[{args.command}] exit {code} -- killing anything left in the container", flush=True)
        subprocess.run(remote.kill_sim(script_name), shell=True)

    return code


if __name__ == "__main__":
    sys.exit(main())
