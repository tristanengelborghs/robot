# The stock lift task, read before writing our own

*`Isaac-Lift-Cube-Franka-IK-Rel-v0`, Isaac Lab 3.0.0. Source vendored under
`reference/isaaclab_source/.../manipulation/lift/`.*

This is the task the pick-and-lift work starts from, and this document exists so
that the reward function we eventually write is informed by theirs rather than
copied from it. Everything below is a fact about their code, with the parts that
change what we build called out.

## What the task actually is

**Not "pick up the block".** It is *lift the block to a commanded goal pose*. A
`UniformPoseCommand` generates a target every 5 seconds — x in [0.4, 0.6], y in
[-0.25, 0.25], **z in [0.25, 0.5]** — and most of the reward is for closing the
distance between the block and that goal. Picking the block up is the first third
of it.

That matters for us in two ways. The goal appears in the policy's observations,
so a policy trained on demonstrations that ignore the goal learns to ignore an
input that is meant to matter. And the reward is dominated by goal tracking, so
copying it wholesale would train something other than the task we asked for.

| | |
|---|---|
| Robot | Franka, `panda_hand`, IK-rel with `scale=0.5`, offset `[0, 0, 0.107]` |
| Object | `dex_cube_instanceable.usd`, spawns at `z=0.055`, **settles to `z=0.021`** |
| Control | `decimation=2`, `sim.dt=0.01` → **50 Hz**, episodes 5 s (250 steps) |
| Action | 7: end-effector delta pose (6) + gripper (1), same as the stacking task |

Note the control rate: 50 Hz here against 20 Hz for stacking, and episodes of 5 s
against 30. A controller tuned for one will be visibly wrong on the other.

## Observations

One concatenated vector, `enable_corruption = True` (so there is observation
noise, unlike the stacking task):

| term | width | note |
|---|---|---|
| `joint_pos` | 9 | relative to the default pose |
| `joint_vel` | 9 | relative |
| `object_position_in_robot_root_frame` | 3 | **root frame**, not world |
| `target_object_position` | 7 | the commanded goal, pos + quat |
| `actions` | 7 | last action |

**There is no end-effector pose in the observations.** The stacking task gave us
`eef_pos` and `eef_quat` directly; this one does not. The scene still has an
`ee_frame` transformer — the rewards use it — so a scripted controller reads it
off `env.scene` rather than out of the observation dict.

## Rewards, and what each one is buying

```python
reaching_object                    weight   1.0    1 - tanh(d_ee_obj / 0.1)
lifting_object                     weight  15.0    1 if object.z > 0.04 else 0
object_goal_tracking               weight  16.0    is_lifted * (1 - tanh(d_goal / 0.3))
object_goal_tracking_fine_grained  weight   5.0    is_lifted * (1 - tanh(d_goal / 0.05))
action_rate                        weight  -1e-4   penalise jerky actions
joint_vel                          weight  -1e-4   penalise thrashing
```

The shape is worth taking seriously, because it is the answer to "how do you get
PPO to discover a grasp":

* **Reaching is dense and cheap** (weight 1). It exists only to get the gripper
  near the block, and its `tanh` kernel with `std=0.1` means it saturates once
  the hand is within about 10 cm — it stops paying before it can compete with
  the real objective.
* **Lifting is a step function with a large weight** (15). No shaping at all: the
  block is either above 4 cm or it is not. The size of the jump is what makes the
  grasp worth discovering, since nothing rewards *approaching* a grasp.

  On paper this threshold looked broken — the cube is *spawned* at `z = 0.055`,
  which is already above it — so it was measured instead (`scripts/probe_lift.py`):
  the cube falls and **settles at `z = 0.021`**, and the reward fires for 0% of
  environments at rest. The threshold asks for about 2 cm of genuine lift. Worth
  recording because the spawn height in the config invites exactly the wrong
  conclusion.
* **Goal tracking is gated on `is_lifted`** — it pays nothing until the block is
  off the table. Without that gate an agent could farm goal-distance reward by
  hovering the empty gripper near the target.
* **Two goal terms, coarse and fine** (`std` 0.3 then 0.05). The coarse one
  gives gradient from far away; the fine one only pays near the goal and stops
  the policy settling for "close enough".
* **The penalties are tiny** (1e-4). They smooth the motion without ever
  outweighing the task.

## Terminations, and the gap we have to fill

```python
time_out          episode length reached
object_dropping   object falls below -0.05
```

**There is no success term.** `record_demos.py` looks for
`env_cfg.terminations.success` to decide when a demonstration is finished and
whether to export it; without one it warns and marks nothing successful, and
`EXPORT_SUCCEEDED_ONLY` then writes an empty file. So recording demonstrations on
this task requires us to define what success means — which is the first piece of
this task that is genuinely ours rather than theirs.

For "just pick up the block" that is: the object above a height, held for a few
consecutive steps, with the gripper closed on it. Not the commanded goal, which
we are deliberately not chasing yet.

## Measured at rest

From `scripts/probe_lift.py`, four environments, 60 settling steps:

```
control rate       50 Hz          episode length  5.0s
object z at rest   0.021          (spawns at 0.055 and falls)
object xy at rest  (0.597, 0.058) randomised per reset
end-effector       (0.463, 0.000, 0.385)
```

Observation term widths, unconcatenated: `joint_pos` 9, `joint_vel` 9,
`object_position` 3, `target_object_position` 7, `actions` 7 — 35 in total.

## What this tells us to build

1. **Our own success termination**, because the task ships without one.
2. **A scripted controller that reads `ee_frame` from the scene**, not from
   observations, and is tuned for 50 Hz rather than the stacking task's 20 Hz.
3. **Our own reward**, later, for pick-and-lift: keep `reaching_object` and
   `lifting_object`, drop both goal terms, keep the small penalties. The
   interesting question is whether a step-function lift bonus alone is enough for
   PPO to find a grasp, or whether it needs a grasp-shaped term as well — which
   is a question their reward does not answer, because goal tracking carries the
   policy once the block is up.
4. **An observation subset for imitation learning** that excludes
   `target_object_position`, since our demonstrations will not be pursuing a
   goal.
