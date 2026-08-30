#!/usr/bin/env bash
#
# Start a teleoperated recording session from *inside* the container.
#
# `make teleop` on the laptop is the outside half of a pair: it syncs this
# project to the box, patches Isaac Lab in the container, and then reaches in
# over ssh -> docker exec. Run it from the browser VS Code terminal and it tries
# to ssh to the machine it is already on, so it fails on its first line.
#
# This is the inside half. Run it from the VS Code terminal on the box:
#
#     /workspace/robot/franka-isaac/scripts/teleop_in_container.sh
#     /workspace/robot/franka-isaac/scripts/teleop_in_container.sh 3
#
# It applies the same container patches `make sync` would have applied, then
# launches Isaac Lab's own record_demos.py with the same flags harness.remote
# builds -- tests/test_remote.py checks the two stay in step.
#
# It cannot be headless: a keyboard device attaches to an application window, so
# the run streams (--livestream 2) and that window is served to a browser.

set -euo pipefail

ISAACLAB=/workspace/isaaclab
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK=Isaac-Stack-Cube-Franka-IK-Rel-v0
NUM_DEMOS="${1:-5}"
DATASET_FILE="${2:-/workspace/datasets/stack_teleop.hdf5}"

# The container ships no system Python; /isaac-sim/python.sh is the only
# interpreter on it. Prefer a real python3 if one ever appears.
PYTHON="$(command -v python3 || echo /isaac-sim/python.sh)"

echo "[teleop] applying container patches"
"$PYTHON" "$PROJECT/scripts/patch_container.py"

mkdir -p "$(dirname "$DATASET_FILE")"

cat <<INSTRUCTIONS

Teleoperated recording -- $NUM_DEMOS demonstrations to $DATASET_FILE

  Watch it at the instance's Brev URL with /viewer on the end, for example
  https://isaac-fzq49bb7n.brevlab.com/viewer -- the console's Access tab has
  the current one. Isaac Sim takes a few minutes to start before the viewer
  shows anything.

  Keyboard, with the viewer focused:

      W / S     end-effector +x / -x        Z / X   roll  + / -
      A / D     end-effector +y / -y        T / G   pitch + / -
      Q / E     end-effector +z / -z        C / V   yaw   + / -

      K         toggle the gripper open and closed
      L         recentre the teleoperation device
      R         abandon this episode and reset the scene

  Stack blue, then red, then green. An episode is written out on its own once
  the cubes are stacked and the gripper is open; only successes are exported,
  so a botched attempt costs nothing but time.

INSTRUCTIONS

cd "$ISAACLAB"
exec ./isaaclab.sh -p "$ISAACLAB/scripts/tools/record_demos.py" \
    --task "$TASK" \
    --livestream 2 \
    --teleop_device keyboard \
    --num_demos "$NUM_DEMOS" \
    --dataset_file "$DATASET_FILE"
