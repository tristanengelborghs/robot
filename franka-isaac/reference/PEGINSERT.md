# Isaac-Factory-PegInsert-Direct-v0 — Component 1 inspection

Everything below is static: it comes from the task's config classes and env
implementation, not from a run. Captured 2026-08-28 from Isaac Lab 3.0.0 on the
L4 box, so it can be read on a laptop with the instance stopped.

Sources in this directory:
* `factory_source/` — the task's own code, copied verbatim from the container
* `peginsert_env.json` / `.md` — machine dump from `scripts/inspect_env.py`

## Robot model

Franka Emika Panda, 9 actuated joints: `panda_joint1..7` plus
`panda_finger_joint1`, `panda_finger_joint2`. The same arm `../panda-libero`
drives through LIBERO — robosuite calls it Panda, Isaac calls it Franka.

Scene contents: `terrain`, `robot`, `fixed_asset`, `held_asset`.

## Plug and socket assets

| | peg (held) | hole (fixed) |
|---|---|---|
| USD | `factory_peg_8mm.usd` | `factory_hole_8mm.usd` |
| diameter | 0.007986 m | 0.0081 m |
| height | 0.050 m | 0.025 m |
| mass | 0.019 kg | — |

**Clearance is 114 µm** on diameter. That single number is the task: it is far
below what the policy can perceive from pose observations alone, so the last
millimetre has to be solved by contact, not by aiming.

Both assets spawn with `activate_contact_sensors=True`.

## Observation space

`Box(-inf, inf, (19,), float32)` — asymmetric actor-critic, so the policy and
the critic see different things.

* **policy (19)** — `fingertip_pos_rel_fixed`, `fingertip_quat`, `ee_linvel`,
  `ee_angvel`, then `prev_actions`.
* **critic (privileged)** — the above in world frame plus `joint_pos`,
  `held_pos`, `held_pos_rel_fixed`, `held_quat`, `fixed_pos`, `fixed_quat`.

Note the policy never observes the socket pose directly, only its fingertip
position *relative* to it.

⚠️ `factory_env_cfg.py` literally reads `observation_space = 21`, but the
comment above it says the value "will be overwritten to correspond to
obs_order". The runtime value is **19**. Trust the dump, not the literal.

## Action space

`Box(-inf, inf, (6,), float32)` — relative Cartesian pose targets (3 position,
3 rotation). The gripper is *not* in the action space: it holds the peg
throughout.

Actions are EMA-smoothed before use, in `_pre_physics_step`:
`actions = ema_factor * new + (1 - ema_factor) * old`.

## Control frequency

| | |
|---|---|
| physics | **120 Hz** (`sim.dt = 1/120`) |
| policy | **15 Hz** (`decimation = 8`) |
| episode | 10.0 s → **150 policy steps** |

## Reward function

Keypoint-distance shaping, not a sparse success bonus. Offsets are generated
along the peg and the hole, transformed into world frame, and the mean L2
distance between the two keypoint sets drives the reward
(`_get_rewards`, `factory_env.py:408`).

Three coefficient pairs sharpen the basin as the peg closes in:

```
keypoint_coef_baseline = [5, 4]
keypoint_coef_coarse   = [50, 2]
keypoint_coef_fine     = [100, 0]
```

## Success condition

`_get_curr_successes` (`factory_env.py:344`) — both must hold:

1. **centred**: `xy_dist < 0.0025` (2.5 mm lateral)
2. **inserted**: `z_disp < height * success_threshold`, with
   `success_threshold = 0.04` (4% of socket height)

`check_rot` adds a rotation test, used by nut-threading, not peg insertion.
`engage_threshold = 0.9` marks partial engagement separately from success.

## Reset procedure

Randomised on every reset:

* hand starts at `[0, 0, 0.047]` relative to the fixed asset tip, noise
  `[0.02, 0.02, 0.01]`, orientation `[π, 0, 0]` with yaw noise `±0.785 rad` (45°)
* socket position noise `[0.05, 0.05, 0.05]`, yaw anywhere in `360°`
* peg sits in the gripper with noise `[0.003, 0, 0.003]` — the grasp itself is
  uncertain, which is realistic and makes pose-only policies brittle

## Contact configuration

PhysX, from `factory_env_cfg.py:99`:

```
bounce_threshold_velocity  = 0.2
friction_offset_threshold  = 0.01
gpu_max_rigid_contact_count = 2**23
gpu_max_rigid_patch_count   = 2**23
gpu_max_num_partitions      = 1     # "Important for stable simulation"
```

Assets use `max_depenetration_velocity = 5.0` and zero linear/angular damping.

## PPO configuration

`factory_source/agents/rl_games_ppo_cfg.yaml` — rl_games is the only agent
config the task ships.

| | |
|---|---|
| network | MLP `[512, 128, 64]`, critic 1024 |
| gamma / tau | 0.995 / 0.95 |
| learning rate | 1.0e-4 |
| horizon | 128 |
| minibatch | 512 |
| mini-epochs | 4 |
| clip (e_clip) | 0.2 |
| entropy coef | 0.0 |
| max epochs | 200 |

Entropy coefficient of zero is worth noting: exploration comes from the reset
randomisation and the initial policy sigma, not from an entropy bonus.

## The other two Factory tasks

Same env class, different assets and episode budget: `gear_mesh` (20 s) and
`nut_thread` (30 s), selected by `task_name`.
