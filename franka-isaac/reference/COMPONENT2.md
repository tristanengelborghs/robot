# Component 2 — validating the recording pipeline

*Recorded and replayed on the GPU box, 2026-08-29. Isaac Lab 3.0.0, Isaac Sim
6.0.1, one NVIDIA L4.*

[../note.md](../note.md) asks for five recorded episodes, a successful replay, a
script that prints each dataset's observation and action shapes, and a plot of
end-effector position over time — on
`Isaac-Stack-Cube-Franka-IK-Rel-v0`, before any of it is pointed at insertion.
All four exist. The demonstrations come from a scripted controller rather than a
human at a keyboard, for the reason in the README: keyboard teleoperation on a
headless box means a browser WebRTC viewer and a person driving a Franka while
the instance bills by the hour, and the pipeline being validated here is the same
either way.

## What was recorded

| | |
|---|---|
| Task | `Isaac-Stack-Cube-Franka-IK-Rel-v0` |
| Episodes | 5 successes in 5 attempts |
| Length | 231, 243, 238, 235, 235 steps (≈12 s each at 20 Hz) |
| Dataset | `datasets/stack_scripted.hdf5`, format version 1 (quaternions `xyzw`) |
| Controller | `harness/stack_sm.py`, driven by `scripts/record_scripted.py` |

Per episode the recorder stores `actions (T, 7)`, `processed_actions`, nine named
observation terms — `eef_pos (T, 3)`, `eef_quat (T, 4)`, `gripper_pos (T, 2)`,
`joint_pos (T, 9)`, `joint_vel (T, 9)`, `object (T, 39)`, `cube_positions
(T, 9)`, `cube_orientations (T, 12)`, `actions (T, 7)` — an `initial_state`, and
a full per-step `states` group for every articulation and rigid object.

`reference/stack_scripted.ee.png` is the end-effector trace. Its z axis is the
clearest picture of what the controller does: down to 0.03 m to grasp, up to
0.165 m to carry, down to 0.078 m to place the first cube, then the same cycle
again ending 0.047 m higher, because the second cube goes on top of the first.

## Does a replay reproduce the original trajectory?

`replay_demos.py --validate_states --validate_success_rate`, one environment:

* **5 of 5 episodes replayed to success.** Every episode, replayed from its
  recorded initial state through its recorded actions, ends in a stack that
  satisfies the task's own `cubes_stacked` termination.
* **Per-step state comparison: 444 of 1182 steps matched exactly**, where a match
  means every component of every asset's pose and velocity is within 0.01 of the
  recording.

The 738 steps that did not match are worth reading rather than counting, because
of *what* diverges:

| asset | quantity | components off | median | max |
|---|---|---|---|---|
| cube_2 | root_velocity | 1653 | 0.036 | 19.1 |
| cube_3 | root_velocity | 994 | 0.043 | 19.9 |
| cube_1 | root_velocity | 456 | 0.025 | 0.57 |
| cube_2 | root_pose | 280 | 0.013 | 0.028 |
| cube_3 | root_pose | 13 | 0.015 | 0.016 |
| robot | joint_velocity | 3 | 0.014 | 0.027 |

Nearly all of it is the *velocity of a free cube*, with occasional spikes of
20 rad/s — the instant of a contact impulse, where GPU PhysX is not
bit-reproducible and a fixed 0.01 tolerance is far tighter than the quantity is
stable. Cube poses drift by at most 2.8 cm, well inside the task's own 4 cm
`xy_threshold` for a completed stack. The robot itself is essentially identical:
three joint-velocity components across 1182 steps × 9 joints.

So: the arm reproduces, the outcome reproduces, and the cubes' contact dynamics
do not reproduce to the last digit. That is the expected behaviour of contact-rich
simulation on a GPU, and it is the same reason Component 4's insertion task will
need a success condition defined on poses rather than on trajectories.

## Three things in Isaac Lab 3.0.0 had to be patched first

None of them are ours, all of them stop a headless Franka run before it produces
anything, and none can be fixed from an environment config, because Isaac Lab's
own scripts hit them too. They are applied to the container by
`scripts/patch_container.py` on every `make sync` — the container is disposable,
so they are reapplied rather than persisted — and each is pinned by a test in
`tests/test_patches.py` that checks the pattern still matches the vendored
upstream source.

1. **The Franka asset moved.** `isaaclab_assets/robots/franka.py` asks for
   `.../Robots/FrankaEmika/panda_instanceable.usd`. NVIDIA reorganised the 6.0
   asset bucket and it now lives under `FrankaEmika/Legacy/`. The shipped path
   returns 404 and every Franka task dies in `spawn_from_usd`. The table asset in
   the same scene resolves fine, which is what makes it look like a network fault
   rather than a moved file.
2. **`replay_demos.py` cannot start headless.** It builds an `Se3Keyboard` for
   its pause/resume keys unconditionally, and that reaches for
   `omni.appwindow.get_default_app_window()`, which does not exist without a
   window.
3. **`--validate_states` raises on every dataset.** `EpisodeData.get_state`
   returns `states[index, None]`, keeping a leading singleton dimension, while
   `compare_states` has already indexed the runtime state down to one
   environment. It compares a length of 1 against a length of 7 and raises
   *"State shape of root_pose for asset robot don't match"* at the first
   comparison — for Isaac Lab's own recorded datasets as much as for ours.

## Recording by hand

note.md's Component 2 asks for keyboard teleoperation, and the plumbing for it
is built and verified as far as the network allows: `make teleop` runs Isaac
Lab's own `record_demos.py` with `--teleop_device keyboard`, streaming rather
than headless because a keyboard device attaches to an application window.
Launched, it starts the app, brings up Kit's WebRTC extensions, and serves the
viewer page — which loads correctly over an ssh tunnel.

It stops at one hard boundary. The viewer signals over TCP 443 and takes its
video over **WebRTC on UDP**; the running app opens ephemeral UDP ports for it.
The instance's security group allows port 22 and nothing else, and `ssh -L`
forwards TCP only, so a tunnel can serve the page and can never serve the
stream. The viewer sits on "WAITING FOR STREAM...", which is the correct
behaviour for a media path that has nowhere to go.

Crossing it means opening TCP 443 and the WebRTC media UDP range on the
instance, then browsing directly to `https://<public-ip>/viewer/` — what the
launchable's viewer is hard-wired for: its `main.tsx` sets
`signalingPort: 443` and a `mediaServer` fixed to the box's public IP. That is a
security-group change and a decision about exposing a GPU box to the internet,
so it is not automated here.

This is why the scripted controller was worth building first rather than second.
It records unattended, over ssh, with no ports open at all.

## What this leaves for Component 5

Component 5 asks for a scripted controller with state-transition logs and a video
of nominal completion. `harness/stack_sm.py` is that controller for stacking, and
`--log_transitions` prints the logs. What it is not is a controller for
*insertion*: stacking tolerates centimetres, and the last five millimetres of a
plug going into a socket are settled by contact, not by a pose error driven to
zero. The state machine's shape should survive; its tolerances and its notion of
"seated" will not.
