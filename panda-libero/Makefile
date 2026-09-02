.PHONY: help install install-sim install-hand hand-model data smoke vla-smoke hand-smoke test train eval stats clean \
        start stop status sync box-install box-check-vla box-data box-train box-eval box-logs box-runs box-ps box-kill pull-results pull-ckpt

# ---- the GPU box (shared with ../franka-isaac) ----------------------------------
INSTANCE := isaac-launchable-e4cbd5
REMOTE := ~/robot/panda-libero
BOX_ENV := PATH=$$PWD/.venv/bin:$$PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_DISABLE_PROGRESS_BARS=1

# ---- knobs ----------------------------------------------------------------------
SUITES ?= libero_goal
SUITE ?= libero_goal
CONFIG ?= configs/vla_goal.yaml
RUN ?= vla-goal
OVERRIDES ?=
CKPT ?=
ABLATION ?= none
EPISODES ?= 50
SEED ?= 0
A ?=
B ?=

help:
	@echo "install      core deps (train + develop, no simulator)"
	@echo "install-sim  LIBERO + robosuite, pinned (needed only for eval)"
	@echo "install-hand mediapipe + opencv (needed only to track real video)"
	@echo "hand-model   download the 7MB hand landmark model (once, for real video)"
	@echo "data         download LIBERO demos + build the language cache (SUITES=libero_goal)"
	@echo "smoke        synthetic data -> 600 training steps of the baseline; loss must fall"
	@echo "vla-smoke    same, for the VLA on a random offline backbone; no download"
	@echo "hand-smoke   synthetic hand demos -> 600 training steps; no camera needed"
	@echo "test         unit tests (fast, no simulator, no GPU, no downloads)"
	@echo "train        train CONFIG here (CONFIG=configs/vla_goal.yaml OVERRIDES='train.steps=20')"
	@echo "eval         roll CKPT out on SUITE (EPISODES=50 SEED=0 ABLATION=none)"
	@echo "stats        paired permutation test: A=<episodes.jsonl> B=<episodes.jsonl>"
	@echo ""
	@echo "The GPU box (bills by the hour — 'make stop' when you walk away):"
	@echo "start / stop / status   the Brev instance"
	@echo "sync         rsync this project to the box (code only; data/runs/results stay put)"
	@echo "box-install  build .venv + LIBERO on the box; prints the toolchain it found"
	@echo "box-check-vla  load the real backbone there once: prompt, one step, checkpoint, loader"
	@echo "box-data     download SUITES on the box and cache the instruction embeddings"
	@echo "box-train    detached training run: CONFIG= RUN= OVERRIDES="
	@echo "box-eval     detached rollouts: CKPT=runs/<run>/ckpt_last.pt RUN= ABLATION= EPISODES= SEED="
	@echo "box-logs     tail runs/RUN.log on the box"
	@echo "box-runs     list runs and results on the box"
	@echo "box-ps       what robobench processes are running there"
	@echo "box-kill     stop them"
	@echo "pull-results copy run.json, train logs and results/ back here (no checkpoints)"
	@echo "pull-ckpt    copy runs/RUN/ckpt_last.pt back here"

install:
	pip install -r requirements.txt && pip install -e .

install-sim:
	bash scripts/setup_libero.sh

install-hand:
	pip install -r requirements-hand.txt

# MediaPipe's Tasks API takes an explicit model file; the wheel no longer bundles
# weights. A separate target rather than an automatic fetch, so nothing downloads
# behind your back the first time you point the tool at a video.
hand-model:
	mkdir -p cache
	curl -fL -o cache/hand_landmarker.task \
	    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

data:
	bash scripts/download_data.sh $(SUITES)
	python scripts/cache_language.py --data-root data/libero --suites $(SUITES)

smoke:
	python scripts/make_synthetic_data.py --length 60
	python -m robobench.train --config configs/smoke.yaml train.steps=600 train.log_every=100

# The VLA on a random, unit-test-sized stand-in for its backbone: exercises the
# prompt, the LoRA wrapping, the trainable-only checkpoint and the EMA without
# fetching anything. `gripper_l1` falling alongside the loss is the number to watch.
vla-smoke:
	python scripts/make_synthetic_data.py --length 60
	python -m robobench.train --config configs/smoke.yaml model.name=vla model.backbone=tiny \
	    model.hidden_dim=32 train.steps=600 train.log_every=100

hand-smoke:
	python scripts/track_hands.py --synthetic --instruction "pick up the block" \
	    --episodes 4 --seconds 24 --out data/hand/hand_demos --image-size 64
	python -m robobench.train --config configs/hand_bc.yaml data.root=data/hand \
	    data.image_size=64 data.lang_cache=cache/lang_hand_synthetic.npz \
	    model.hidden_dim=64 model.vision_width=16 model.num_layers=2 \
	    model.num_heads=2 train.steps=600 train.batch_size=16 train.log_every=100

test:
	pytest tests/ -q

train:
	python -m robobench.train --config $(CONFIG) $(OVERRIDES)

eval:
	@test -n "$(CKPT)" || (echo "usage: make eval CKPT=runs/<run>/ckpt_last.pt [SUITE= EPISODES= SEED= ABLATION=]"; exit 1)
	python -m robobench.eval --ckpt $(CKPT) --suite $(SUITE) --episodes $(EPISODES) --seed $(SEED) --ablation $(ABLATION)

stats:
	@test -n "$(A)" -a -n "$(B)" || (echo "usage: make stats A=results/<run>/<suite>_episodes.jsonl B=..."; exit 1)
	python -m robobench.stats $(A) $(B)

clean:
	rm -rf runs results cache data/synthetic data/hand .pytest_cache

# ---- the GPU box ------------------------------------------------------------------
# Same Brev instance franka-isaac uses, but nothing here runs inside its Isaac
# container: this project needs a plain Python on the host with a GPU, which the
# box has. Long jobs are detached with nohup so they outlive the ssh session and
# a sleeping laptop; the instance bills until `make stop`.

start:
	brev start $(INSTANCE) -d
	@echo "waiting for RUNNING (brev start returns while the box is still booting)"
	@for i in $$(seq 1 60); do \
		s=$$(brev ls 2>/dev/null | awk '/$(INSTANCE)/{print $$2}'); \
		echo "  $$s"; \
		[ "$$s" = "RUNNING" ] && break; \
		sleep 15; \
	done
	brev refresh

status:
	brev ls

stop:
	brev stop $(INSTANCE)

# Code only. data/, runs/, results/, cache/ and third_party/ are excluded on both
# sides, so --delete never touches what was downloaded or trained on the box.
# The excludes are anchored with a leading slash: an unanchored 'data' matches
# every path component of that name, including src/robobench/data/.
sync:
	ssh $(INSTANCE) 'mkdir -p $(REMOTE)'
	rsync -az --delete --exclude '/.venv' --exclude '/data' --exclude '/runs' --exclude '/results' \
		--exclude '/cache' --exclude '/third_party' --exclude '__pycache__' --exclude '.pytest_cache' \
		./ $(INSTANCE):$(REMOTE)/

box-install: sync
	ssh $(INSTANCE) 'cd $(REMOTE) && bash scripts/box_install.sh'

box-check-vla: sync
	ssh $(INSTANCE) 'cd $(REMOTE) && $(BOX_ENV) python scripts/check_vla_backbone.py'

box-data: sync
	ssh $(INSTANCE) 'cd $(REMOTE) && $(BOX_ENV) make data SUITES="$(SUITES)"'

# `setsid -f`, not `nohup ... &`: with Brev's ssh config (RequestTTY, connection
# multiplexing) a nohup'd python keeps the ssh session open for the life of the
# run and `make` sits there for nine hours. A new session with no controlling
# terminal returns at once and survives the laptop sleeping.
box-train: sync
	ssh $(INSTANCE) 'cd $(REMOTE) && mkdir -p runs && \
		$(BOX_ENV) setsid -f python -m robobench.train --config $(CONFIG) run_name=$(RUN) $(OVERRIDES) \
		> runs/$(RUN).log 2>&1 < /dev/null; \
		sleep 8; echo "started $(RUN)"; tail -c 400 runs/$(RUN).log | tr "\r" "\n" | tail -n 3'

box-eval: sync
	@test -n "$(CKPT)" || (echo "usage: make box-eval CKPT=runs/<run>/ckpt_last.pt RUN=<name> [ABLATION= EPISODES= SEED=]"; exit 1)
	ssh $(INSTANCE) 'cd $(REMOTE) && mkdir -p runs && \
		$(BOX_ENV) setsid -f python -m robobench.eval --ckpt $(CKPT) --suite $(SUITE) \
		  --episodes $(EPISODES) --seed $(SEED) --ablation $(ABLATION) \
		> runs/eval-$(RUN)-$(SUITE)-$(ABLATION)-s$(SEED).log 2>&1 < /dev/null; \
		sleep 8; echo "started eval-$(RUN)-$(SUITE)-$(ABLATION)-s$(SEED)"'

box-logs:
	ssh $(INSTANCE) 'tail -n 40 $(REMOTE)/runs/$(RUN).log'

box-runs:
	ssh $(INSTANCE) 'cd $(REMOTE) && ls -lht runs/ 2>/dev/null; echo; find results -name "*_summary.json" 2>/dev/null | sort'

box-ps:
	ssh $(INSTANCE) 'pgrep -af "robobench\.(train|eval)" || echo "nothing running"'

box-kill:
	ssh $(INSTANCE) 'pkill -f "robobench\.(train|eval)" && echo killed || echo "nothing running"'

pull-results:
	mkdir -p runs results
	rsync -az --include '*/' --include 'run.json' --include 'train_log.jsonl' --include '*.log' --exclude '*' \
		$(INSTANCE):$(REMOTE)/runs/ runs/
	rsync -az $(INSTANCE):$(REMOTE)/results/ results/ 2>/dev/null || true
	@find results -name '*_summary.json' | sort

pull-ckpt:
	mkdir -p runs/$(RUN)
	rsync -az $(INSTANCE):$(REMOTE)/runs/$(RUN)/ckpt_last.pt runs/$(RUN)/ckpt_last.pt
