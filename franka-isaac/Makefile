.PHONY: help install test start stop status shell code sync pull smoke video inspect source teleop-source \
        record teleop tunnel bridge bridge-axes bridge-demo replay dataset ext envs kill clean

INSTANCE := isaac-launchable-e4cbd5
CONTAINER := vscode
TASK := Isaac-Factory-PegInsert-Direct-v0
VIDEO_LEN := 200
NUM_DEMOS := 5
TELEOP_DEVICE := keyboard
DATASET := datasets/stack_scripted.hdf5
DEMOS := 100
RUN := record-lift
BOX_DATASETS := /workspace/datasets
ISAACLAB := /workspace/isaaclab
WS := /workspace/robot/franka-isaac
REMOTE := ~/robot/franka-isaac

help:
	@echo "install   deps into .venv (pytest, numpy, h5py, matplotlib — no simulator)"
	@echo "test      unit tests: commands, scripted controller, dataset reader — no GPU, <1 min"
	@echo "start     boot the Brev GPU box and refresh its (always-changing) IP"
	@echo "status    is the box running? are we being billed?"
	@echo "sync      rsync this project to the GPU box, and patch its moved Franka asset"
	@echo "pull      copy edits made in the browser VS Code back to this laptop"
	@echo "shell     interactive shell inside the Isaac Lab container"
	@echo "code      tunnel browser VS Code to http://localhost:8080"
	@echo "smoke     headless peg-insert on the GPU box; verifies Isaac Lab still runs"
	@echo "record    scripted cube-stacking demos on the GPU box -> HDF5 (Component 2)"
	@echo "teleop    record demos by hand over the browser viewer (TELEOP_DEVICE=keyboard|gamepad)"
	@echo "tunnel    forward that viewer to http://localhost:8090/viewer/ (run in its own shell)"
	@echo "bridge    laptop half of network teleop: read the controller, stream it to the box"
	@echo "bridge-axes  print what the controller reports, to check the axis mapping"
	@echo "bridge-demo  send scripted motion instead of a controller, to test the link"
	@echo "replay    replay those demos and validate them against the recorded states"
	@echo "record-lift  detached: record DEMOS scripted pick-and-lift demonstrations"
	@echo "train-diffusion  detached: train the diffusion policy on the box"
	@echo "eval-diffusion   detached: roll the trained policy out and count successes"
	@echo "train-ppo        detached: train PPO from scratch"
	@echo "logs      tail a detached run's log (RUN=record-lift)"
	@echo "runs      list the detached runs on the box"
	@echo "pull-runs copy logs and checkpoints back to the laptop"
	@echo "dataset   pull the dataset to the laptop, print its shapes, plot the end-effector"
	@echo "source    copy Isaac Lab's Factory task source into reference/ (read it offline)"
	@echo "teleop-source  same, for the teleop/recording/stack-task sources"
	@echo "inspect   dump obs/action spaces, control Hz, joints into reference/"
	@echo "video     record an mp4 of the task and pull it into reference/videos/"
	@echo "ext       pip install source/insertion into Isaac Lab's python on the box"
	@echo "envs      list the gym tasks our extension registers"
	@echo "kill      stop any simulator running in the container (it bills until you do)"
	@echo "stop      stop the box — do this every time you walk away"

install:
	python3.11 -m venv .venv && .venv/bin/python -m pip install -U pip && \
	.venv/bin/python -m pip install pytest numpy h5py matplotlib pygame

test:
	.venv/bin/python -m pytest tests/ -q

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

sync:
	rsync -az --delete --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
		./ $(INSTANCE):$(REMOTE)/
	ssh $(INSTANCE) 'docker exec $(CONTAINER) rm -rf /workspace/robot/franka-isaac'
	ssh $(INSTANCE) 'docker cp $(REMOTE) $(CONTAINER):/workspace/robot/'
	PYTHONPATH=. .venv/bin/python -c "import subprocess; from harness import remote; subprocess.run(remote.patch_container(), shell=True, check=True)"

shell:
	ssh -t $(INSTANCE) 'docker exec -it $(CONTAINER) bash'

smoke:
	PYTHONPATH=. .venv/bin/python scripts/smoke.py

record: sync
	PYTHONPATH=. .venv/bin/python scripts/run_remote.py record --num-demos $(NUM_DEMOS)

# Human teleoperation needs three things at once: the tunnel (its own shell),
# a browser on the viewer, and this run. Start `make tunnel` first.
teleop: sync
	PYTHONPATH=. .venv/bin/python scripts/run_remote.py teleop \
		--num-demos $(NUM_DEMOS) --device $(TELEOP_DEVICE)

tunnel:
	PYTHONPATH=. .venv/bin/python scripts/run_remote.py tunnel

# The laptop half of network teleoperation: reads the controller here and
# streams commands to the device waiting in the container.
bridge:
	PYTHONPATH=. .venv/bin/python scripts/teleop_bridge.py

bridge-axes:
	PYTHONPATH=. .venv/bin/python scripts/teleop_bridge.py --show-axes

bridge-demo:
	PYTHONPATH=. .venv/bin/python scripts/teleop_bridge.py --demo

replay:
	PYTHONPATH=. .venv/bin/python scripts/run_remote.py replay

# Long runs are detached on the box: they outlive the ssh connection, so a
# sleeping laptop cannot kill a job the instance is still being billed for.
record-lift: sync
	PYTHONPATH=. .venv/bin/python -c "import subprocess; from harness import remote; \
		subprocess.run(remote.detached(remote.record_lift($(DEMOS)), 'record-lift'), shell=True)"

train-diffusion: sync
	PYTHONPATH=. .venv/bin/python -c "import subprocess; from harness import remote; \
		subprocess.run(remote.detached(remote.train_diffusion(), 'train-diffusion'), shell=True)"

train-ppo: sync
	PYTHONPATH=. .venv/bin/python -c "import subprocess; from harness import remote; \
		subprocess.run(remote.detached(remote.train_ppo(), 'train-ppo'), shell=True)"

eval-diffusion: sync
	PYTHONPATH=. .venv/bin/python -c "import subprocess; from harness import remote; \
		subprocess.run(remote.detached(remote.eval_diffusion(), 'eval-diffusion'), shell=True)"

logs:
	PYTHONPATH=. .venv/bin/python -c "import subprocess; from harness import remote; \
		subprocess.run(remote.tail_log('$(RUN)'), shell=True)"

runs:
	PYTHONPATH=. .venv/bin/python -c "import subprocess; from harness import remote; \
		subprocess.run(remote.list_runs(), shell=True)"

pull-runs:
	PYTHONPATH=. .venv/bin/python -c "import subprocess; from harness import remote; \
		subprocess.run(remote.pull_runs(), shell=True)"

dataset:
	mkdir -p datasets
	ssh $(INSTANCE) 'rm -rf ~/datasets && docker cp $(CONTAINER):$(BOX_DATASETS) ~/datasets'
	rsync -az $(INSTANCE):~/datasets/ datasets/
	PYTHONPATH=. .venv/bin/python scripts/inspect_dataset.py $(DATASET) \
		--plot --plot-file reference/stack_scripted.ee.png

kill:
	PYTHONPATH=. .venv/bin/python -c "import subprocess; from harness import remote; subprocess.run(remote.kill_sim(), shell=True)"

clean:
	rm -rf runs .pytest_cache

inspect: sync
	ssh $(INSTANCE) 'mkdir -p $(REMOTE)/reference'
	ssh $(INSTANCE) 'docker exec $(CONTAINER) bash -lc "cd $(ISAACLAB) && ./isaaclab.sh -p $(WS)/scripts/inspect_env.py --headless --viz none"'
	ssh $(INSTANCE) 'docker cp $(CONTAINER):$(WS)/reference/. $(REMOTE)/reference/'
	rsync -az $(INSTANCE):$(REMOTE)/reference/ reference/

source:
	ssh $(INSTANCE) 'mkdir -p $(REMOTE)/reference'
	ssh $(INSTANCE) 'rm -rf $(REMOTE)/reference/factory_source && docker cp $(CONTAINER):$(ISAACLAB)/source/isaaclab_tasks/isaaclab_tasks/direct/factory $(REMOTE)/reference/factory_source'
	rsync -az --exclude '__pycache__' $(INSTANCE):$(REMOTE)/reference/factory_source/ reference/factory_source/

# The teleoperation, recording and stack-task sources Component 2 is built
# against. Path-preserving, so a file's location in reference/ is its location
# in the Isaac Lab tree.
teleop-source:
	mkdir -p reference/isaaclab_source
	ssh $(INSTANCE) 'docker exec $(CONTAINER) bash -lc "cd $(ISAACLAB) && tar cf - --exclude=__pycache__ --exclude=\*.pyc \
		scripts/tools/record_demos.py scripts/tools/replay_demos.py \
		scripts/environments/state_machine/lift_cube_sm.py \
		source/isaaclab/isaaclab/utils/datasets \
		source/isaaclab/isaaclab/managers/recorder_manager.py \
		source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack"' \
		| tar xf - -C reference/isaaclab_source

video: sync
	ssh $(INSTANCE) 'docker exec $(CONTAINER) bash -lc "cd $(ISAACLAB) && ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task $(TASK) --headless --enable_cameras --video --video_length $(VIDEO_LEN) --num_envs 64 --max_iterations 3"'
	ssh $(INSTANCE) 'rm -rf ~/vid && mkdir -p ~/vid && docker exec $(CONTAINER) bash -lc "find $(ISAACLAB)/logs -name \"*.mp4\" -newermt \"-1 hour\"" | while read f; do docker cp $(CONTAINER):"$$f" ~/vid/; done'
	rsync -az $(INSTANCE):~/vid/ reference/videos/
	@echo "video(s) in reference/videos/"

ext: sync
	ssh $(INSTANCE) 'docker exec $(CONTAINER) bash -lc "cd $(ISAACLAB) && ./isaaclab.sh -p -m pip install -e $(WS)/source/insertion"'

envs: sync
	ssh $(INSTANCE) 'docker exec $(CONTAINER) bash -lc "cd $(WS) && $(ISAACLAB)/isaaclab.sh -p scripts/list_envs.py"'

pull:
	ssh $(INSTANCE) 'rm -rf $(REMOTE).back && docker cp $(CONTAINER):$(WS) $(REMOTE).back'
	rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
		$(INSTANCE):$(REMOTE).back/ ./
	@echo "pulled container -> laptop; run 'git diff' to see what changed there"

code:
	@echo "tunnelling code-server to http://localhost:8080/?folder=/workspace/robot"
	@echo "(ctrl-C to close; the browser keeps it alive while you use it)"
	ssh -N -L 8080:localhost:8080 $(INSTANCE)
