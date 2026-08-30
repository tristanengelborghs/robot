# shadow-isaac

Throw-and-catch on a Shadow Hand in Isaac Lab. The hand is horizontal, palm up;
a ball is dropped into it, then tossed into it, and finally popped up by the
fingers themselves and caught again.

Named for the robot and the simulator, like its siblings.
[../shadow-mujoco](../shadow-mujoco) is the same hand in MuJoCo doing in-hand
reorientation; this project is the same hand in Isaac Lab doing a task that
actually needs the parallelism, because a random policy essentially never
catches anything and rare success is what thousands of environments are for.

## The split

Isaac Lab environments are Python, not scene files, so there is very little to
author in the Isaac Sim GUI -- the ball is a primitive sphere and the hand is a
stock asset. The division of labour that matters is different:

* **Laptop side** (`harness/`): phases, reward, termination, curriculum, and
  the ballistics they are built on. Pure NumPy, no `isaaclab` import, no GPU.
  All of it is tested here, because the hard logic is where the bugs are and
  the box bills by the hour. This is the split `franka-isaac` uses for
  `stack_sm.py`, for the same reason.
* **Box side** (`source/`, to come): a thin `DirectRLEnv` that wires the above
  to the simulator. Small by construction.
* **Human side**: whether the catch *looks* right -- ball mass, radius,
  restitution, friction, and the hand pose. Nobody can judge that from a test.
  A ball with lively restitution bounces straight back out of a good catch.

## Status

Laptop side is written and green: **42 tests**, no downloads, no GPU, under two
seconds.

```bash
make install    # venv: numpy, torch, pytest
make test       # 42 passed
```

The Isaac Lab extension is written but **has never been executed**. It imports
`isaaclab`, which does not exist on this laptop, so everything in
`source/catching/` is unvalidated until it runs on the box. That is the whole
reason the logic is not in there -- see the next section.

## How the untestable part is kept small

`source/catching/.../catch_env.py` cannot be imported here, let alone run. So
it holds no task logic: it is buffer bookkeeping and asset plumbing, which a
simulator has to check anyway. Everything else lives in two places that need no
simulator, and they are cross-checked against each other:

```
harness/catch.py                        NumPy, scalar, readable. The reference.
source/catching/.../rewards.py          torch, batched. What runs on the GPU.
tests/test_parity.py                    drives both with 512 random states
                                        across all three stages and asserts
                                        phase, reward and termination agree.
```

Two implementations of the same thing is a liability unless something forces
them to agree. That test is the something. It loads `rewards.py` straight from
its file, bypassing the package `__init__` that would drag in isaaclab, which
is possible only because `rewards.py` deliberately imports nothing from Isaac.

## First run on the box

```bash
make start                 # boot, poll to RUNNING, refresh ssh config
make bodies                # <- do this FIRST. See below.
make install-ext           # pip install -e source/catching in the container
make envs                  # confirm Catch-Shadow-Direct-v0 registered
make smoke                 # random agent, headless, 4 envs (OUR script -- see below)
make fetch-scripts         # then patch train/play the same way
make train NUM_ENVS=4096
make tunnel                # separate shell
make play CKPT=...         # browser viewer
make kill && make stop
```

**Installing an extension does not import it.** Registration lives in
`catching/tasks/direct/catch/__init__.py` and only runs when something imports
the package; Isaac Lab's own scripts import `isaaclab_tasks` and nothing else,
so the task comes back `NameNotFound: Environment Catch-Shadow-Direct doesn't
exist` however correctly it is installed. `scripts/random_agent.py` is
project-local for exactly this reason, as Isaac Lab's template generator does
for external projects. `train.py` and `play.py` need the same one-line
treatment -- `make fetch-scripts` copies Isaac Lab's out of the container so
the import can be added and version-controlled rather than reconstructed.

`make bodies` is first for a reason. The palm lookup and the contact sensor in
`catch_env_cfg.py` are **regexes over body names that have not been checked
against the asset**, and a contact-sensor regex matching nothing does not
raise -- it reports no contacts, and that surfaces much later as a policy that
can never catch. `scripts/inspect_hand.py` prints the real names. The env also
refuses to start if the palm pattern does not match exactly one body or the
fingertip pattern does not match five, and asserts the assembled observation
width against `observation_space` on the first step.

Expect asset-path friction generally: the Franka USD moved upstream and 404'd
under the sibling project, which is what `franka-isaac/scripts/patch_container.py`
exists for.

## What only you can judge

Three things no test can settle, all of them in the viewer:

* **The hand must rest palm up.** Gravity is what holds a caught ball. The
  stock `SHADOW_HAND_CFG` resting pose is used as-is and may need a rotation.
* **Ball restitution.** Defaulted to 0.02 -- a lively ball bounces back out of
  a good catch and the policy gets punished for succeeding.
* **Ball mass, radius and friction.** 50 g and 3 cm are a guess at something a
  20-actuator hand can cradle.

## The three things this design is defending against

**A reward desert during the flight.** Between release and catch the ball is
ballistic and the hand cannot touch it, so a catch-only payout leaves that
whole stretch without a gradient. The flight is instead scored on *predicted*
intercept error -- the distance from the palm to where the ball will cross palm
height, closed form in `ballistics.predict_intercept`. The policy cannot move
the ball, but it can be in the right place when it arrives.

**A reward whose optimum is to never throw.** Pay per step for holding and
penalise dropping, and sitting still is optimal on a launch stage: no risk,
endless reward, a healthy-looking return curve, and the task never attempted.
Hold reward is gated on a completed flight, and a launch stage terminates as
`never_launched` if nothing gets airborne. `test_catch.py` asserts the idle
episode scores exactly zero and strictly below a catch.

**A control rate that cannot aim.** This one is arithmetic, and it is checked
rather than assumed:

```
  20 Hz  REFUSED: 4.0 control steps of flight for a 5 cm toss; need >= 20
  50 Hz  REFUSED: 10.1 control steps of flight for a 5 cm toss; need >= 20
 100 Hz  OK
 200 Hz  OK
```

A 5 cm toss lasts 0.2 s. At 20 Hz that is four decisions, and the ball crosses
its own diameter between two of them. `CatchConfig.validate` refuses the
configuration up front, instead of letting it surface later as a policy that
mysteriously will not learn.

## Stages

Catching is learned before throwing, because stage 1 has no ballistic phase at
all and therefore no credit-assignment gap. Each stage adds exactly one
difficulty.

```
DROP   ball released above the palm at rest -> catch and hold
TOSS   ball spawned with upward velocity    -> track apex, catch on descent
POP    the fingers launch the ball          -> then catch it
```

Within a stage the toss height widens on *measured* success (`Curriculum`),
never on a step schedule -- a schedule tuned on one machine traverses the
curriculum differently on another, and a smoke run and a real run then see
different tasks under the same config.

## Success and failure

A catch is `min_contacts` fingertips, inside `catch_radius` of the palm centre,
above the palm, below `secure_speed` relative speed, held for `hold_steps`.
Contact alone is not a catch: a ball grazing a fingertip on its way past
reports contact too.

Failures are named -- `dropped`, `out_of_bounds`, `excess_force`,
`never_launched`, `timeout` -- because a success rate without a failure
breakdown says nothing about what to fix.

## Layout

```
harness/
  ballistics.py   closed-form free flight; the fake plant, and the prediction
                  the policy is given during the flight
  catch.py        Stage/Phase, CatchConfig (+validate), is_secured, phase_of,
                  reward_terms, terminate, Curriculum -- the reference
  remote.py       ssh -> docker exec -> isaaclab.sh command construction
scripts/
  run_remote.py   thin CLI over harness.remote; --dry-run prints, runs nothing
  inspect_hand.py prints the hand's real body and joint names (runs on the box)
source/catching/  the Isaac Lab extension (installable; unvalidated)
  .../catch/rewards.py         batched torch task logic, imports no isaaclab
  .../catch/catch_env_cfg.py   scene, spaces, rates, task scalars
  .../catch/catch_env.py       DirectRLEnv wiring and nothing else
  .../catch/agents/            rl_games PPO config
tests/            42 tests, laptop-only
```

The extension layout mirrors what Isaac Lab's own template generator emits, and
`pyproject.toml` is lint config only -- deliberately not the generator's root
file, which declares `name = "isaaclab-dev"` and points `[tool.uv.sources]` at
paths that exist only inside the Isaac Lab repo.
