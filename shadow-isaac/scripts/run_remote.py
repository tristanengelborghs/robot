"""Run one remote command against the GPU box.

A thin CLI over `harness.remote`, which builds the strings and is tested
without a network. Everything here is argument parsing and one subprocess call,
deliberately: command construction that lives in a script cannot be tested.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from harness import remote


def main() -> int:
    # --dry-run is on a shared parent rather than the top-level parser, so it
    # is accepted AFTER the subcommand. Declared on the top-level parser it is
    # silently routed to the subparser, which has never heard of it, and every
    # invocation dies on a usage message instead.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true",
                        help="print the command, run nothing")

    p = argparse.ArgumentParser(description=__doc__, parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help):
        return sub.add_parser(name, help=help, parents=[common])

    add("install", "pip install -e the extension in the container")
    s = add("smoke", "random agent against the task, headless")
    s.add_argument("--num-envs", type=int, default=4)
    sc = add("scripted", "scripted catcher -- the gate before PPO")
    sc.add_argument("--num-envs", type=int, default=64)
    sc.add_argument("--steps", type=int, default=2000)
    sc.add_argument("--lead-time", type=float, default=0.08)
    sc.add_argument("--spawn-toward", type=float, default=None)
    sc.add_argument("--livestream", action="store_true",
                    help="stream to the browser viewer instead of running headless")
    t = add("train", "rl_games PPO")
    t.add_argument("--num-envs", type=int, default=4096)
    t.add_argument("--max-iterations", type=int, default=1000)
    pl = add("play", "watch a checkpoint in the browser viewer")
    pl.add_argument("checkpoint")
    pl.add_argument("--num-envs", type=int, default=1)
    add("orient", "measure the palm's facing, print the rotation to make it up")
    add("pose", "print the running hand's raw world geometry")
    add("palmup", "search all 24 orientations for palm-up")
    add("envs", "list registered task ids")
    add("bodies", "print the hand's real body and joint names")
    add("tunnel", "forward the viewer (blocks)")
    add("kill", "stop a simulator left running in the container")

    a = p.parse_args()
    cmd = {
        "install": lambda: remote.install_extension(),
        "smoke": lambda: remote.smoke(a.num_envs),
        "scripted": lambda: remote.scripted(a.num_envs, a.steps, a.lead_time,
                                            getattr(a, "livestream", False),
                                            getattr(a, "spawn_toward", None)),
        "train": lambda: remote.train(a.num_envs, a.max_iterations),
        "play": lambda: remote.play(a.checkpoint, a.num_envs),
        "orient": lambda: remote.orient_hand(),
        "pose": lambda: remote.inspect_pose(),
        "palmup": lambda: remote.find_palm_up(),
        "envs": lambda: remote.list_envs(),
        "bodies": lambda: remote.body_names(),
        "tunnel": lambda: remote.tunnel(),
        "kill": lambda: remote.kill(),
    }[a.cmd]()

    print(cmd, file=sys.stderr)
    if a.dry_run:
        return 0
    return subprocess.run(cmd, shell=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
