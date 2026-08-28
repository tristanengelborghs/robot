#!/usr/bin/env python3
"""Fetch the MuJoCo Menagerie hand models, pinned to a recorded commit.

Pinned for the same reason the sibling project pins LIBERO: hand models change
(collision meshes, actuator gains, joint limits), and an unpinned asset makes
your trained policies and success rates uncomparable to anyone else's — or to
your own from last month. The commit lives in .menagerie_commit; put it in any
results table.

    python scripts/fetch_assets.py            # clone at the pinned commit
    python scripts/fetch_assets.py --update   # move the pin to upstream HEAD
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "third_party" / "mujoco_menagerie"
PIN = ROOT / ".menagerie_commit"
URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
DIRS = ["shadow_hand", "wonik_allegro"]


def run(*cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()

    if not DEST.exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--sparse", URL, str(DEST))
        run("git", "sparse-checkout", "set", *DIRS, cwd=DEST)

    if args.update or not PIN.exists():
        run("git", "checkout", "main", cwd=DEST)
        run("git", "pull", cwd=DEST)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=DEST).decode().strip()
        PIN.write_text(sha + "\n")
        print(f"pinned menagerie at {sha}")
    else:
        sha = PIN.read_text().strip()
        run("git", "fetch", "origin", sha, cwd=DEST)
        run("git", "checkout", sha, cwd=DEST)
        print(f"menagerie at pinned {sha}")

    for d in DIRS:
        xml = DEST / d / "right_hand.xml"
        if not xml.is_file():
            sys.exit(f"expected {xml} after fetch — menagerie layout changed?")
    print("assets ok:", ", ".join(DIRS))


if __name__ == "__main__":
    main()
