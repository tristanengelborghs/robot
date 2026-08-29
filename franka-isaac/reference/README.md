# reference

Captured facts about the stock Isaac Lab task this project starts from, so that
Component 1 ("inspect, change nothing") stays answerable on a laptop with the
GPU box stopped, and so that the code Component 2 is written against can be read
on the laptop instead of remembered. Regenerate any of it with `make source`,
`make teleop-source`, `make inspect`, `make video`.

* `PEGINSERT.md` — written summary: robot, assets, spaces, control rate, reward,
  reset, success, contact config, PPO config.
* `peginsert_env.json` / `.md` — machine dump from `scripts/inspect_env.py`.
  Authoritative where it disagrees with the config source: the config declares
  `observation_space = 21`, but the runtime value is 19.
* `videos/` — mp4 recorded headless on the GPU box.
* `factory_source/` — **third-party code, not ours.** A verbatim read-only copy
  of Isaac Lab's Factory task from
  `source/isaaclab_tasks/isaaclab_tasks/direct/factory/`, taken from Isaac Lab
  3.0.0 in the container. Copyright (c) 2022-2026 The Isaac Lab Project
  Developers, BSD-3-Clause; the original copyright headers are intact in every
  file. It is vendored rather than gitignored like `third_party/` because the
  whole point is to read it *without* renting a GPU. Never edit it — anything we
  change belongs in `source/insertion/`.
* `isaaclab_source/` — **third-party code, not ours**, same terms as
  `factory_source/` above: a path-preserving copy of the Isaac Lab files
  Component 2 is built against, so a file's location here is its location in the
  Isaac Lab tree. `scripts/tools/record_demos.py` and `replay_demos.py` (the
  recording pipeline `scripts/record_scripted.py` mirrors),
  `scripts/environments/state_machine/lift_cube_sm.py` (where the `+1` open /
  `-1` close gripper convention is pinned down), the HDF5 dataset handler that
  defines the file layout `harness/dataset.py` reads, the recorder manager, and
  the whole `manipulation/stack` task — its observation terms, its
  `cubes_stacked` success test and the reset randomisation the scripted
  controller has to cope with.
* `COMPONENT2.md` — written summary of the recording pipeline validation: what
  was recorded, what a replay does and does not reproduce, and the three Isaac
  Lab 3.0.0 bugs `scripts/patch_container.py` works around.
* `stack_scripted.ee.png` — end-effector position over time for the recorded
  demonstrations, from `make dataset`. Component 2's visualization deliverable.
