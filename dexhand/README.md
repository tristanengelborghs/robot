# dexhand

In-hand cube reorientation on dexterous hands: a MuJoCo environment, a PPO
teacher trained on privileged state, and a DAgger-distilled student that sees
only what plausible hardware provides. The community-standard dexterity task
(OpenAI 2018, DeXtreme 2022), built from scratch.

The sibling project ([../1dof](../1dof)) collapses the hand to one scalar —
a parallel-jaw gripper is open or closed, and everything interesting happens
in the arm. This project is the opposite bet: the arm barely moves and
everything interesting happens in **contact**. A Shadow Hand has 24 joints, 20
actuators, and tendon-coupled distal pairs (one actuator drives two joints
through a shared tendon — real underactuation, in the vendored model, not a
simplification of it). The cube is controllable only through intermittent
frictional contact with five fingertips.

## Why MuJoCo and not Isaac Lab

One honest reason: Isaac requires an NVIDIA RTX GPU and this repo was built on
an M1 Mac. MuJoCo is the stronger contact and tendon simulator anyway — what
Isaac buys is scale, thousands of parallel environments. The seams are cut for
that port: everything above `vec.py` consumes "step a batch of envs, get
stacked dicts back", so a GPU backend replaces one file, not the task.
Meanwhile `vec.py` exploits a MuJoCo-specific fact: `mj_step` releases the
GIL, so a plain thread pool gets real parallelism with zero IPC —
~1,700 control steps/s (42k physics steps/s) across 8 envs on the M1.

## Quickstart

```bash
make install     # venv: mujoco, torch, gymnasium
make test        # toy 3-finger hand — no downloads, no GPU, <1 min
make assets      # Shadow + Allegro models, pinned Menagerie commit
make smoke       # ~5 min on a laptop CPU; the curve below must reproduce
```

The smoke run trains PPO on easy goals (rotations about z, up to 1 rad) with
the full 20-actuator Shadow Hand and domain randomization on. Successes per
episode must rise; this run's curve, 8 CPU threads:

```
steps       succ/ep   mean rot_dist
  1,024      0.00       1.08
 29,696      0.64       0.91
 58,368      1.44       0.76
 87,040      1.97       0.64
119,808      1.66       0.72     (~2 min wall on an idle M1, 8 threads)
```

A flat curve means the learning signal is broken — reward, observation
normalization, or GAE — and no amount of further compute will fix it. That is
what a smoke test is for. It has already earned its keep once: the first
version of the training loop cut the GAE bootstrap at time-limit truncations
and bootstrapped horizon-edge episodes from the *next* episode's first
observation; its curve reached 0.50 successes/episode at 58k steps. The
corrected loop reaches 1.44 at the same point.

## Compute honesty

Full SO(3) reorientation at the 0.4 rad threshold needs on the order of 1e8+
environment steps (OpenAI used years of simulated experience; DeXtreme used
~10^10 steps on GPU farms). On this laptop that is weeks, not minutes. The
`make train` config is written for a CUDA box or a long patient run; what this
repo demonstrates end-to-end on a laptop is the machinery — environment,
randomization, curriculum, teacher, distillation — with a verifiable learning
signal, not a solved Rubik's cube.

## The training recipe

```
teacher   PPO on privileged state (exact cube pose, velocities, tactile),
          asymmetric actor-critic, adaptive curriculum
student   DAgger distillation onto proprio + tactile + goal + history —
          no cube pose; the seam where vision would plug in
```

1. **Teacher** — `python -m dexhand.train`. The curriculum widens goal
   difficulty when measured success clears a threshold (not on a step
   schedule, so smoke runs and GPU runs traverse it at their own pace), then
   switches from z-rotations to full SO(3).
2. **Student** — `python -m dexhand.distill --ckpt runs/<run>/ckpt_last.pt`.
   Rolls out the *student*, labels with the teacher: BC-on-teacher-rollouts
   fails here specifically because in-hand errors compound within a few
   control steps, and a student that has never seen its own slippage states
   cannot recover from them.
3. **Evaluate** — `python -m dexhand.eval --ckpt ...`. Per-episode JSONL, same
   discipline as the sibling project: aggregates you cannot re-test are
   claims, not results.

## Design decisions that were bugs first

Every one of these was caught by running, not by reading:

- **Truncation is not failure.** A time-limit cut has full continuation
  value; a drop has none. The first loop stored one merged `done` flag and
  treated both alike — and since no observation carries time-in-episode, the
  critic was pulled down everywhere. `bootstrap_truncations` folds
  γ·V(final_obs) into the last reward; `test_train.py` pins both cases.
- **Goals are born outside the success threshold.** The first smoke run
  reported 50 successes/episode at iteration 1: goals sampled inside the
  threshold pay the bonus for doing nothing. `test_env.py` pins the fix.
- **Solver options do not survive scene composition.** MjSpec attachment
  keeps the parent's `<option>` and only warns; the Shadow model's elliptic
  cones and `impratio 10` — the settings that make grasps stick — silently
  vanish unless re-applied. Composed scenes are tested for them.
- **Domain randomization restores a pristine snapshot before every draw,
  then calls `mj_setConst`.** Perturbing the live model compounds
  multiplicatively; after a thousand resets friction has random-walked out of
  its band and training is quietly non-stationary. And a bare write to
  `body_mass` leaves `body_subtreemass` stale, so the "heavier" cube
  accelerated like the nominal one. A test does 200 resets and asserts the
  band for every randomized field.
- **Every hand preset is dropped-tested.** The Allegro preset shipped with
  the Shadow's cube spawn — 30 cm down a forearm the Allegro does not have —
  and every episode was a one-step drop no metric flagged. `settle_check`
  drops the cube on a relaxed hand at construction and refuses a hand that
  does not hold it.
- **Action smoothing lives inside the env.** Raw PPO exploration at 20 Hz
  commands finger thrash that breaks grasps in sim and hardware alike; the
  EMA is part of the dynamics the policy learns through, not a post-filter
  that invalidates what it learned.
- **Tactile counts only cube contacts, in log scale.** Finger-on-finger force
  is real force and zero task information; raw Newtons waste the input range
  on rare hard presses.

## Layout

```
configs/            default.yaml is the full surface; smoke.yaml merges on top
scripts/            fetch_assets.py (pinned Menagerie commit -> .menagerie_commit)
src/dexhand/
  hands.py          HandSpec: which MJCF, which actuators, which fingertips
  scene.py          MjSpec composition: hand + cube + goal ghost -> one model
  env.py            the task; obs views {actor, critic, student}
  tactile.py        per-fingertip net contact force from the solver
  randomize.py      DR: snapshot -> log-uniform perturbation per reset
  vec.py            threaded vec env (GIL-released mj_step); the Isaac seam
  rotations.py      quat math, wxyz, tested to the double-cover edge cases
  networks.py       asymmetric actor-critic + student MLP
  ppo.py            PPO/GAE with the bootstrap cut at dones tested explicitly
  train.py          teacher loop, adaptive curriculum, JSONL logging
  distill.py        DAgger teacher -> student
  eval.py           per-episode outcomes on disk
  view.py           interactive viewer (macOS: mjpython)
tests/              toy 3-finger hand ships in-repo: no downloads, no GPU
```

## Hands

| name    | source                    | actuators | notes                                  |
|---------|---------------------------|-----------|----------------------------------------|
| shadow  | Menagerie `shadow_hand`   | 20        | tendon-coupled distal joints; default  |
| allegro | Menagerie `wonik_allegro` | 16        | fully actuated; `env.hand=allegro`     |
| toy     | ships in `tests/assets`   | 6         | test rig, not a good hand              |

Menagerie is pinned (`.menagerie_commit`) because hand models change and
unpinned assets make results uncomparable — put the commit in your table.
