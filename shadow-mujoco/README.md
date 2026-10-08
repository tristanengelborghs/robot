# Dexterous manipulation with reinforcement learning

A simulated Shadow Hand learns to rotate a cube to a target orientation using
reinforcement learning. Built with Python, PyTorch, and MuJoCo, with support for
both Shadow and Allegro hands.

The project includes a custom environment, PPO training, domain randomization,
and a teacher–student pipeline. The teacher uses the cube's position, orientation,
and velocity from the simulator. The student learns from joint state, fingertip
contact forces, the target orientation, and recent observations.

## Getting started

Run these commands from `shadow-mujoco/`:

```bash
make install     # install dependencies in a local virtual environment
make test        # run tests using the included toy hand; no GPU or downloads
make assets      # download the Shadow and Allegro models
make smoke       # train on simplified rotation goals on CPU
```

The project runs on a laptop. Longer training runs can use CUDA. The hand models
come from MuJoCo Menagerie, pinned to the commit recorded in `.menagerie_commit`.

## Current results

The CPU smoke run uses the Shadow Hand with all 20 actuators, domain
randomization, and rotation goals around the z-axis of up to 1 radian.
One run on an M1 Mac with eight simulation threads produced:

| Training steps | Goals reached per episode | Mean rotation error (rad) |
|---|---|---|
| 1,024 | 0.00 | 1.08 |
| 29,696 | 0.64 | 0.91 |
| 58,368 | 1.44 | 0.76 |
| 87,040 | 1.97 | 0.64 |
| 119,808 | 1.66 | 0.72 |

These results cover simplified rotation goals. Full 3D reorientation is a longer
training task and has not been demonstrated by this smoke run. The student
training pipeline is implemented; the table above reports teacher performance.

## Training and evaluation

```bash
make train                      # train the PPO teacher
make eval RUN=your-run             # evaluate a saved teacher checkpoint
make distill RUN=your-run          # train a student from that teacher
make view RUN=your-run             # watch the policy in the MuJoCo viewer
```

`your-run` is the folder name under `runs/`. The viewer uses `mjpython` on macOS.
Training settings are in `configs/default.yaml`; `configs/smoke.yaml` contains
the shorter CPU run.

The teacher uses an asymmetric actor–critic and a curriculum that increases goal
difficulty as performance improves, eventually moving to full 3D rotations.
Student training uses DAgger: the student controls the hand, and the teacher
provides action labels for the states it visits. Evaluation saves per-episode
results as JSONL.
