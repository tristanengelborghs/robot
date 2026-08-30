INSTANCE  := isaac-launchable-e4cbd5
CONTAINER := vscode
REMOTE    := ~/robot/shadow-isaac
PY        := .venv/bin/python
NUM_ENVS  := 4096
ITERS     := 1000
LEAD      := 0.08
VIEWER    := https://isaac-fzq49bb7n.brevlab.com/viewer

.PHONY: help install test start status stop sync install-ext smoke scripted watch orient palmup pose fetch-scripts train play envs bodies tunnel kill clean

help:
	@echo "Laptop (no GPU needed):"
	@echo "  make install      venv with numpy, torch, pytest"
	@echo "  make test         the whole laptop-side suite"
	@echo ""
	@echo "Box (bills by the hour):"
	@echo "  make start        boot the instance and refresh ssh config"
	@echo "  make sync         copy this project into the container"
	@echo "  make install-ext  pip install -e the extension in the container"
	@echo "  make bodies       print the hand's real body/joint names"
	@echo "  make envs         list registered task ids"
	@echo "  make smoke        random agent against our task, headless"
	@echo "  make fetch-scripts  copy Isaac Lab's rl_games train/play here to patch"
	@echo "  make scripted     scripted catcher -- run this BEFORE train"
	@echo "  make watch        scripted catcher, streamed to the browser viewer"
	@echo "  make train        rl_games PPO (NUM_ENVS=$(NUM_ENVS) ITERS=$(ITERS))"
	@echo "  make play CKPT=…  watch a checkpoint in the browser viewer"
	@echo "  make tunnel       forward the viewer (own shell; run before play)"
	@echo "  make kill         stop a simulator left running in the container"
	@echo "  make stop         stop the instance"

install:
	python3.11 -m venv .venv && $(PY) -m pip install -U pip && \
	$(PY) -m pip install pytest numpy torch

test:
	$(PY) -m pytest -q

# `brev start` exits non-zero on an instance that is already RUNNING, which
# aborts the target before `brev refresh` — leaving a live box unreachable
# behind a stale ssh config. Starting an already-started instance is the
# desired end state, so it is not an error here.
start:
	-brev start $(INSTANCE) -d
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

# /workspace is NOT bind-mounted from the host, so the project is copied into
# the container itself. The rm -rf is why logs and datasets live outside it.
sync:
	rsync -az --delete --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
		./ $(INSTANCE):$(REMOTE)/
	ssh $(INSTANCE) 'docker exec $(CONTAINER) rm -rf /workspace/robot/shadow-isaac'
	ssh $(INSTANCE) 'docker exec $(CONTAINER) mkdir -p /workspace/robot /workspace/logs/shadow-isaac'
	ssh $(INSTANCE) 'docker cp $(REMOTE) $(CONTAINER):/workspace/robot/'

install-ext: sync
	PYTHONPATH=. $(PY) scripts/run_remote.py install

pose: sync
	PYTHONPATH=. $(PY) scripts/run_remote.py pose

palmup: sync
	PYTHONPATH=. $(PY) scripts/run_remote.py palmup

orient: sync
	PYTHONPATH=. $(PY) scripts/run_remote.py orient

bodies: sync
	PYTHONPATH=. $(PY) scripts/run_remote.py bodies

envs:
	PYTHONPATH=. $(PY) scripts/run_remote.py envs

smoke: sync
	PYTHONPATH=. $(PY) scripts/run_remote.py smoke --num-envs 4

# Isaac Lab's train.py and play.py have the same blind spot as its random_agent:
# they import isaaclab_tasks and never our extension, so the task is
# NameNotFound. Copy them here, add `import catching` after the AppLauncher
# block, and point remote.train/play at the local copies. Fetching rather than
# reconstructing, because these scripts carry real machinery.
fetch-scripts:
	ssh $(INSTANCE) 'docker cp $(CONTAINER):/workspace/isaaclab/scripts/reinforcement_learning/rl_games /tmp/rl_games'
	mkdir -p scripts/rl_games
	rsync -az $(INSTANCE):/tmp/rl_games/ scripts/rl_games/
	@echo "fetched. add 'import catching  # noqa: F401' after the AppLauncher block."

# The gate: a controller with ground-truth state. If this cannot catch, the
# fault is the environment and no amount of PPO will fix it.
scripted: sync
	PYTHONPATH=. $(PY) scripts/run_remote.py scripted --num-envs 64 --lead-time $(LEAD)

# Watch it. The viewer is the Brev console URL + /viewer, NOT an ssh tunnel:
# Kit streams WebRTC over UDP and ssh forwards TCP only, so a tunnelled page
# loads and then never shows a picture.
watch: sync
	@echo "open $(VIEWER)"
	PYTHONPATH=. $(PY) scripts/run_remote.py scripted --num-envs 2 --steps 100000 \
		--lead-time $(LEAD) --livestream

train: sync
	PYTHONPATH=. $(PY) scripts/run_remote.py train --num-envs $(NUM_ENVS) --max-iterations $(ITERS)

play:
	PYTHONPATH=. $(PY) scripts/run_remote.py play $(CKPT)

tunnel:
	PYTHONPATH=. $(PY) scripts/run_remote.py tunnel

kill:
	PYTHONPATH=. $(PY) scripts/run_remote.py kill

clean:
	rm -rf .pytest_cache **/__pycache__
