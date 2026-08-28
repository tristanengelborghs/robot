# reference

Captured facts about the stock Isaac Lab task this project starts from, so that
Component 1 ("inspect, change nothing") stays answerable on a laptop with the
GPU box stopped. Regenerate any of it with `make source`, `make inspect`,
`make video`.

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
