# robobench

A harness for testing new visuomotor architectures on LIBERO, built around one claim:

> **A small architecture, trained from scratch on one GPU, competitive with what
> large pretrained models achieve.**

That is a compute-efficiency claim, and it needs a different setup than a
beat-SOTA claim. The relevant comparison is not the ~98% that 3–7B models
pretrained on Open-X report — it is the **from-scratch column**, where the
published reference (Diffusion Policy, ~150M params, no pretraining) sits around
**78% on LIBERO-Spatial** and **93% on LIBERO-Object**. There is real room under
those numbers, and beating them on a single GPU is a result people will believe.

## Why this repo is shaped the way it is

A 2026 audit of manipulation benchmarks ([What Are We Actually Benchmarking in
Robot Manipulation?](https://arxiv.org/html/2606.04233v1)) found that only about a
fifth of published LIBERO improvements are provably significant, and that a 90M
model *with no language encoder at all* matches state of the art on the easier
suites. Three things follow, and all three are built in rather than left to you:

1. **Per-episode outcomes are always written to disk**, never just an aggregate.
   Without them nobody can test your claim, and they cannot be reconstructed later.
2. **Input ablations are first-class** (`ablation=no_lang`, `no_wrist`, …), applied
   identically in training and rollout. If your language-conditioned policy scores
   the same with language destroyed, you want to know before a reviewer does.
3. **Significance testing ships with the harness.** `robobench.stats` pairs
   episodes on identical initial states and runs a permutation test.

## Install

**Python 3.10+ is required** — mujoco publishes no wheels for 3.9, and pip
silently falls back to a source build that dies on a missing `MUJOCO_PATH`.
The error reads like a MuJoCo problem but is an interpreter-version problem.

Training and development need no simulator at all:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Rollout evaluation additionally needs LIBERO, pinned to a commit:

```bash
make install-sim     # records the hash in .libero_commit — put it in your results table
```

Tracking human video needs a third, also optional, stack:

```bash
make install-hand    # mediapipe + opencv; see "Demos from human video" below
make hand-model      # the landmark weights, which MediaPipe 1.0 no longer bundles
```

`requirements-sim.txt` pins `mujoco` and `numpy` exactly. Both of robosuite's
declarations are unbounded (`mujoco>=2.3.0`, `numpy>=1.13.3`), so an unpinned
install grabs the newest mujoco and upgrades numpy to 2.x on top of a training
stack built against 1.x. The setup script verifies neither happened before
declaring success.

## Ten-minute path, no downloads

```bash
make smoke
```

Generates LIBERO-shaped HDF5 files with a learnable signal drawn into the images,
then trains for 600 steps. The loss must fall (≈0.53 → ≈0.16). If it stays flat,
gradient is not reaching the vision tower — which is exactly what you want a smoke
test to tell you before you spend a day of GPU time.

`make vla-smoke` does the same for the VLA, on a random unit-test-sized stand-in
for its backbone (no download): loss ≈0.55 → ≈0.39, and `gripper_l1` from ≈0.95
to under 0.1, which is the channel that proves the action queries are reading the
sequence rather than the proprio side channel.

## Real path

```bash
make data SUITES=libero_goal                       # demos + frozen CLIP instruction cache
python -m robobench.train --config configs/resnet_film_bc_goal.yaml
python -m robobench.eval  --ckpt runs/<run>/ckpt_last.pt --episodes 50
```

The same three steps run on the GPU box as `make box-data`, `make box-train`,
`make box-eval` — see [The GPU box](#the-gpu-box).

## Demos from human video

Behaviour cloning is bounded by how many demos you have, and a teleoperated LIBERO
demo costs minutes of a robot and an operator. `robobench.hand` records a person
doing the task on camera instead and writes the result in the same HDF5 layout, so
training, ablations, evaluation and statistics work on it unchanged.

```bash
make install-hand                                   # mediapipe + opencv, optional
make hand-model                                     # 7MB landmark model, once
python scripts/watch_hands.py                       # live preview: is it seeing your hand?
python scripts/track_hands.py --video demo.mp4 --instruction "pick up the mug"
python scripts/cache_language.py --data-root data/hand --suites hand_demos --out cache/lang_hand.npz
python -m robobench.train --config configs/hand_bc.yaml
```

No camera and no downloads needed to see the whole path work:

```bash
make hand-smoke
```

Generates four scripted pick-and-places, tracks them, retargets them and trains for
600 steps. The loss must fall (≈0.43 → ≈0.17), and `gripper_l1` must fall with it
(≈0.99 → ≈0.05) — that second number is the one worth watching, because the gripper
channel can only be predicted if grasp timing survived the trip from thumb-to-index
distance through the aperture mapping into the action. A falling loss with a flat
`gripper_l1` means the arm motion transferred and the grasp did not.

Start with the live preview — it answers "is the tracker seeing my hand at all"
before any question about demos arises:

```bash
python scripts/watch_hands.py            # webcam; q to quit
python scripts/watch_hands.py --synthetic  # no camera needed
```

Green skeleton means the frame is usable, amber means the palm is too foreshortened
to trust its orientation, grey means the position is extrapolated through an
occlusion rather than observed. The grip bar is the one to watch: it is what becomes
the gripper command, so if it does not swing decisively as you open and close your
hand, nothing downstream can recover the grasp.

The pipeline is five stages, each swappable:

```
sources    file/camera -> (frame, t)     frame rate read, never assumed
detect     one frame -> hands            MediaPipe by default; a Protocol, not an import
tracker    hands -> stable tracks        association, 1-Euro filtering, occlusion coasting
pose       landmarks -> 6-DoF + grip     Kabsch on the palm; scale-normalized aperture
retarget   poses -> 7-D OSC actions      resampled onto the 20 Hz control clock
record     tracks + video -> HDF5        LIBERO layout, including the vertical flip
```

Each stage exists because of a specific way the naive version fails silently:

| Stage | The version that looks fine and is wrong |
|---|---|
| pose | Building the hand frame from three landmarks (`x = wrist->index`, `z = x × wrist->pinky`). Exact, and so it hands any error on the pinky knuckle straight to the palm normal. Kabsch over all five palm points spreads it. |
| pose | Using the detector's per-frame handedness label. It flips under self-occlusion, and handedness picks the palm template — so one bad frame yields **full confidence and a 180° error**. The tracker votes over the track instead. |
| pose | Gating on the fit's singular-value ratio. It is not monotone in foreshortening: a palm tilted 60° projects *more* squarely than a head-on one, so the score improves as the pose gets worse. The fit residual is monotone. |
| retarget | A deadband that discards sub-threshold motion. A careful 3 cm/s approach is 1.5 mm per control step — under any threshold that suppresses jitter — so the approach phase of every careful demo becomes zeros. Carrying the residual keeps 96% of it. |
| retarget | Clipping that discards the excess. Every step where the hand outran the controller loses the difference permanently, so actions fall further behind the images with each one. |
| retarget | Composing rotations frame-to-frame. Video is 30 fps, LIBERO's controller is 20 Hz; without resampling every demo runs 1.5× fast. Rotations resample by Slerp — interpolating rotation-vector components is not a rotation. |
| record | Writing images right-way-up. LIBERO's are bottom-up and the loader flips them, so a demo stored upright trains upside down relative to every other demo. |
| sources | Defaulting the frame rate when it cannot be read. Timestamps drive every velocity in the demo, so a 60 fps clip read as 30 halves the whole recording's speed — smooth trajectory, in-range actions, falling loss, policy trained to move at half speed. `probe_fps` raises and asks for `--fps` instead. |

**Proprioception is the one thing that does not transfer.** A human hand has no robot
arm behind it, so the nine proprio dimensions carry wrist pose and hand scale rather
than joint angles. They train without complaint, which is exactly the danger, so the
files are stamped and `LiberoChunkDataset` refuses to build a dataset that mixes the
two meanings. `configs/hand_bc.yaml` sets `ablation=no_proprio`, which is logged.

**What transfers and what does not.** Trajectory shape and grasp timing are properties
of the task and do transfer. Contact forces, the robot's kinematics, and anything
needing more than one gripper degree of freedom do not — a parallel-jaw gripper has
nothing in common with a hand there. Hand demos are worth the most as a pretraining
or co-training corpus for the vision-to-action mapping, and whether they helped is a
rollout number against the same architecture trained without them:

```bash
python -m robobench.eval  --ckpt runs/sim_only/ckpt_last.pt --episodes 50 --seed 0
python -m robobench.eval  --ckpt runs/hand_pre/ckpt_last.pt --episodes 50 --seed 0
python -m robobench.stats results/sim_only/libero_object_episodes.jsonl \
                          results/hand_pre/libero_object_episodes.jsonl
```

Check the per-episode summary the recorder prints before training on anything. A high
`clipped` fraction means the demonstration outran the controller and the actions no
longer describe the motion in the images — re-record it slower.

## The VLA, on LIBERO-Goal

`configs/vla_goal.yaml` is the pretrained column of the table: SmolVLM2-500M
(SigLIP → pixel-shuffle connector → SmolLM2-360M) reads both camera frames and the
instruction *text*, LoRA adapters train on the language model's attention, and a
small head regresses the action chunk from the hidden states at eight learned
action-query positions appended to the sequence. That is OpenVLA-OFT's recipe —
parallel decoding, continuous actions, L1 — and it lands on the harness's masked-L1
objective unchanged. `src/robobench/models/vla.py` is the whole model; the module
docstring is the design note.

Why Goal, and not the easier suites: Goal is ten instructions over *one* scene.
Object and Spatial can be solved by looking, which is how a model with no language
encoder matches state of the art on them. On Goal the instruction is the only thing
that separates the tasks, so `ablation=no_lang` is a real test there and the
comparison below means something:

| run | what it shows |
|---|---|
| `resnet_film_bc_goal.yaml` | the from-scratch column: 27M params, no pretraining |
| `resnet_film_bc_goal.yaml ablation=no_lang` | how much of that is language at all — expect a collapse |
| `vla_goal.yaml` | the pretrained column |
| `vla_goal.yaml ablation=no_lang` | whether the VLA's margin is language, or just a better vision tower |
| `vla_goal_frozen.yaml` | how much is the representation vs the fine-tune |

Then `make stats A=... B=...` on each pair, paired on identical initial states.
Report params, *trainable* params (`run.json` carries both) and GPU-hours next to
the success rate; the compute ratio is half the point.

### Results, seed 0

One L4 (24 GB), LIBERO at commit `8f1084e3`, 50 episodes per task on identical
initial states, 500 per row. The `no_lang` rows destroy the instruction (zeroed
CLIP vector *and* blanked string) either only at rollout or in training too.

| model | language | params (trainable) | train | success | 95% CI |
|---|---|---|---|---|---|
| ResNet-FiLM-BC | train + eval | 27M (27M) | 2.5 h | **85.0%** | 81.8–88.0 |
| ResNet-FiLM-BC | zeroed at eval | 27M (27M) | — | 5.2% | 3.4–7.2 |
| ResNet-FiLM-BC | never | 27M (27M) | 2.7 h | 7.4% | 5.2–9.8 |
| VLA (SmolVLM2-500M + LoRA) | train + eval | 465M (4.8M) | 11.4 h | **91.0%** | 88.4–93.4 |
| VLA (SmolVLM2-500M + LoRA) | blanked at eval | 465M (4.8M) | — | 0.4% | 0.0–1.0 |

Paired tests (sign-flip permutation on the 500 shared initial states):

| comparison | delta | 95% CI | p |
|---|---|---|---|
| VLA vs baseline | **+6.0 pp** | +2.6 to +9.4 | 0.0014 |
| baseline vs baseline, language zeroed at eval | −79.8 pp | −83.4 to −76.0 | < 1e-4 |
| baseline vs baseline trained without language | −77.6 pp | −81.4 to −73.6 | < 1e-4 |
| VLA vs VLA, language blanked at eval | −90.6 pp | −93.0 to −88.0 | < 1e-4 |

Reading it: on Goal the instruction carries almost everything for both models —
language-blind, the baseline sits at the 1-in-10 guess rate and the VLA below it.
The VLA's six points come from the tasks the baseline finds hardest (push the
plate 72% → 100%, cream cheese 68% → 78%); its one loss is the two-stage
drawer-then-bowl task (72% → 62%). The from-scratch 27M model at 85% already
clears the published from-scratch reference on Goal (Diffusion Policy, ~68%), so
the pretrained column is buying six points for 4.6× the GPU-hours and 17× the
inference-time parameters. Single seed; run ≥3 before quoting any of it.
Not run yet: `vla_goal_frozen.yaml` (another ~11 h) and the VLA trained
without language.

Three things the VLA forced on the harness, all of which the baseline also gets:

- **The instruction string travels with the batch** (`text`), alongside the cached
  CLIP vector. Every model's `forward` takes `text=None`; the VLA reads it, the
  others ignore it. `no_lang` blanks the string as well as zeroing the vector.
- **LIBERO spells each instruction two ways**, and they differ on four Goal tasks:
  the demo file says "Open the middle layer of the drawer", the evaluator says
  "open the middle drawer of the cabinet". Training now reads the instruction the
  way the evaluator does — from the file name — by default
  (`data.instruction_source: filename`). The CLIP baseline was silently training
  on one embedding and rolling out on another; a model that tokenises text would
  not have been silent about it.
- **Checkpoints and the EMA hold only what trained.** `trainable_state_dict()` is
  everything for a from-scratch model and a few million parameters for the VLA;
  the evaluator rebuilds the backbone from its source and loads the delta on top,
  and refuses if any trainable key is missing.

Before the first real run, `make box-check-vla` loads the actual backbone once
and checks what the offline tests cannot: that the hand-built prompt is
token-for-token what the HF processor produces, one timed forward/backward with
finite gradients, peak memory, and the checkpoint round trip through the
evaluator's loader. `image_res: 256` (16 tokens per camera instead of 64) is the
first knob if a step is too slow.

## Adding your own architecture

One file. Copy the template, implement `forward`, register it:

```bash
cp src/robobench/models/template.py src/robobench/models/my_arch.py
```

```python
@register_model("my_arch")
class MyArch(BasePolicy):
    def forward(self, images, proprio, lang, text=None):
        # images  dict[camera -> uint8 (B, 3, H, W)]
        # proprio float32 (B, 9)     joint angles + gripper
        # lang    float32 (B, 512)   frozen CLIP instruction embedding
        # text    list of B instruction strings, for models with their own text encoder
        return actions              # (B, chunk_size, 7) in normalized [-1, 1]
```

Add the import to `models/__init__.py`, then:

```bash
python -m robobench.train --config configs/resnet_film_bc.yaml model.name=my_arch
```

Everything under `model:` in the YAML besides `name` is passed to your constructor,
so new hyperparameters need no plumbing; switching `model.name` starts that block
fresh, so the baseline's hyperparameters never reach a constructor that has not
heard of them. The masked-L1 objective, action chunking,
normalization, EMA, evaluation and statistics are shared, so a difference in the
results table is a difference in your architecture. If your method needs another
objective — diffusion, flow matching, a VAE term — override `loss()` and `predict()`.

`pytest tests/ -q` runs every registered model through a shape and gradient check,
so a new architecture is verified as wired-in without touching a simulator.

## The comparison you actually want to publish

```bash
# same seed, same episodes, same initial states — only the architecture differs
python -m robobench.eval --ckpt runs/baseline/ckpt_last.pt  --episodes 50 --seed 0
python -m robobench.eval --ckpt runs/my_arch/ckpt_last.pt   --episodes 50 --seed 0

python -m robobench.stats results/baseline/libero_object_episodes.jsonl \
                          results/my_arch/libero_object_episodes.jsonl
```

```
  paired on 500 episodes across 10 tasks
  discordant pairs: 154  (the only ones that carry signal)

  baseline    74.60%   95% CI [70.8, 78.4]
  candidate   88.60%   95% CI [85.8, 91.4]

  delta      +14.00 pp   95% CI [+9.2, +18.8]
  p          0.0000  (paired sign-flip permutation, two-sided)
```

Run it across ≥3 training seeds before believing the number.

## The GPU box

Training the VLA and running 500 rollouts want a Linux GPU. The Makefile drives
the same Brev instance `../franka-isaac` rents (`isaac-launchable-e4cbd5`, one L4,
24 GB), but on the host rather than inside its Isaac container — this project needs
a plain Python with CUDA, which the host has.

```bash
make start                       # boot it; polls until RUNNING
make box-install                 # .venv + LIBERO on the box; prints the toolchain it found
make box-check-vla               # the real backbone, once
make box-data SUITES=libero_goal # demos + language cache, on the box
make box-train CONFIG=configs/resnet_film_bc_goal.yaml RUN=bc-goal
make box-train CONFIG=configs/vla_goal.yaml RUN=vla-goal
make box-logs RUN=vla-goal
make box-eval CKPT=runs/<run>/ckpt_last.pt RUN=vla-goal
make box-eval CKPT=runs/<run>/ckpt_last.pt RUN=vla-goal ABLATION=no_lang
make pull-results                # run.json, train logs, results/ — not checkpoints
make stop                        # ALWAYS — it bills by the hour
```

Runs are detached with `setsid`, so they outlive the ssh session and a sleeping
laptop (`nohup` does not: under Brev's ssh config it keeps the session, and
`make`, open for the life of the run); `make box-ps` and `make box-kill` are the
other half of that. `make sync`
copies code only — `data/`, `runs/`, `results/`, `cache/` and `third_party/` are
excluded on both sides, so nothing downloaded or trained on the box is ever
deleted by a sync. `brev login` first if `make status` says you are logged out.

## Protocol notes

- **Pin the LIBERO commit.** Task definitions and init states have changed over
  time; an unpinned install makes your numbers uncomparable to anyone else's.
- **50 episodes per task** is the informal floor. Fix it before you look at results.
- **Report params and GPU-hours** next to success rate — on a saturated benchmark
  that is often the more defensible contribution.
- **Run the ablation.** `ablation=no_lang` costs one extra training run and is the
  cheapest credibility available.
- **Headroom check.** LIBERO-Object and Spatial are close to ceiling for pretrained
  models. For a genuinely hard scene-diversity claim on the same checkpoints, see
  [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) (7 perturbation dimensions;
  frontier models fall from 95% to under 30%).

## Layout

```
configs/          default.yaml is the full surface; others merge on top
scripts/          data download, language cache, synthetic generator, box install, backbone check
src/robobench/
  models/         base.py (the contract) · resnet_film_bc.py (baseline) · vla.py (pretrained) · template.py (start here)
  data/           LIBERO HDF5 -> action chunks, action normalization
  hand/           human video -> demos: sources · detect · tracker · pose · retarget · record
  train.py        shared training loop
  eval.py         rollouts + per-episode logging
  stats.py        paired permutation test + bootstrap CIs
  ablate.py       input ablations, applied in train AND eval
tests/            no simulator, no GPU, no downloads, no camera
```

## Simulator install: what actually breaks

`make install-sim` handles all of these; they are documented because the error
messages point somewhere other than the cause.

| Symptom | Cause |
|---|---|
| `RuntimeError: MUJOCO_PATH environment variable is not set` | Python 3.9. mujoco ships no 3.9 wheel, so pip falls back to a source build. Use 3.10+. |
| numpy silently upgraded to 2.x | robosuite declares `numpy>=1.13.3` unbounded. Pinned to `<2.0`; the setup script asserts it afterwards. |
| `import libero` works inside the LIBERO checkout, fails elsewhere | LIBERO's `libero/` has no `__init__.py`, so upstream's `find_packages()` registers nothing and `pip install -e .` is a no-op. Setup writes a `.pth` pointing at the checkout. |
| First import hangs forever | LIBERO calls `input()` to ask where datasets live. Setup pre-writes `~/.libero/config.yaml` (never overwrites an existing one). |
| `UnpicklingError` loading init states | LIBERO calls `torch.load` without `weights_only`; torch 2.6 flipped that default. `eval.py` allowlists the numpy reconstructors. |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | mediapipe removed the legacy `mp.solutions` API (gone by 0.10.30) in favour of `mediapipe.tasks`. The detector targets the Tasks API; the weights that used to ship inside the wheel now come from `make hand-model`. |
| `Check failed: service_ Service is unavailable` + abort, on macOS arm64 | mediapipe **1.0.1** only. A Metal helper is constructed for a graph that never registers the service, and the process dies in a native CHECK before Python sees anything. Not recoverable from Python — IMAGE, VIDEO and explicit-CPU-delegate all fail identically. `requirements-hand.txt` excludes exactly that release (`!=1.0.1`); 1.0.0 and 0.10.3x are fine. |
| `ModuleNotFoundError: future / easydict / gym` | Undeclared transitive deps. LIBERO's own `requirements.txt` is unusable — it pins `numpy==1.22.4` and `robosuite==1.4.0` and drags in wandb/robomimic. Installed `--no-deps` with the four real runtime deps added back. |

**Rendering backend.** MuJoCo offscreen rendering needs `MUJOCO_GL=egl` on
headless Linux and `MUJOCO_GL=glfw` on macOS — EGL is Linux-only. `make
install-sim` prints the right one for your OS. macOS rollouts run on CPU physics
and are slow; use them to verify the eval path, not to produce numbers.

### Device note

`non_blocking=True` host→device copies are enabled only for CUDA. On MPS an async
copy from pageable memory can return before it completes and deliver silently
corrupted tensors — a loss that looks plausible and means nothing. There is a
regression test for this in `tests/test_smoke.py`.
