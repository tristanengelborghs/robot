.PHONY: help install test start stop status shell sync smoke video inspect source kill clean

INSTANCE := isaac-launchable-e4cbd5
CONTAINER := vscode
TASK := Isaac-Factory-PegInsert-Direct-v0
VIDEO_LEN := 200
ISAACLAB := /workspace/isaaclab
REMOTE := ~/robot/franka-isaac

help:
	@echo "install   deps into .venv (pytest only — the simulator lives on the GPU host)"
	@echo "test      unit tests: command construction only — no GPU, no network, <1 min"
	@echo "start     boot the Brev GPU box and refresh its (always-changing) IP"
	@echo "status    is the box running? are we being billed?"
	@echo "sync      rsync this project to the GPU box and into the container"
	@echo "shell     interactive shell inside the Isaac Lab container"
	@echo "smoke     headless peg-insert on the GPU box; verifies Isaac Lab still runs"
	@echo "source    copy Isaac Lab's Factory task source into reference/ (read it offline)"
	@echo "inspect   dump obs/action spaces, control Hz, joints into reference/"
	@echo "video     record an mp4 of the task and pull it into reference/videos/"
	@echo "kill      kill a simulator left running in the container (it bills until you do)"
	@echo "stop      stop the box — do this every time you walk away"

install:
	python3.11 -m venv .venv && .venv/bin/pip install -U pip && \
	.venv/bin/pip install pytest && \
	.venv/bin/pip install -e .

test:
	.venv/bin/python -m pytest tests/ -q

start:
	brev start $(INSTANCE) -d && brev refresh

status:
	brev ls

stop:
	brev stop $(INSTANCE)

sync:
	rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
		./ $(INSTANCE):$(REMOTE)/
	ssh $(INSTANCE) 'docker cp $(REMOTE) $(CONTAINER):/workspace/'

shell:
	ssh -t $(INSTANCE) 'docker exec -it $(CONTAINER) bash'

smoke:
	.venv/bin/python scripts/smoke.py

kill:
	ssh $(INSTANCE) 'docker exec $(CONTAINER) pkill -f random_agent.py' || true

clean:
	rm -rf runs .pytest_cache

inspect: sync
	ssh $(INSTANCE) 'mkdir -p $(REMOTE)/reference'
	ssh $(INSTANCE) 'docker exec $(CONTAINER) bash -lc "cd $(ISAACLAB) && ./isaaclab.sh -p /workspace/franka-isaac/scripts/inspect_env.py --headless --viz none"'
	ssh $(INSTANCE) 'docker cp $(CONTAINER):/workspace/franka-isaac/reference/. $(REMOTE)/reference/'
	rsync -az $(INSTANCE):$(REMOTE)/reference/ reference/

source:
	ssh $(INSTANCE) 'mkdir -p $(REMOTE)/reference'
	ssh $(INSTANCE) 'rm -rf $(REMOTE)/reference/factory_source && docker cp $(CONTAINER):$(ISAACLAB)/source/isaaclab_tasks/isaaclab_tasks/direct/factory $(REMOTE)/reference/factory_source'
	rsync -az --exclude '__pycache__' $(INSTANCE):$(REMOTE)/reference/factory_source/ reference/factory_source/

video: sync
	ssh $(INSTANCE) 'docker exec $(CONTAINER) bash -lc "cd $(ISAACLAB) && ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task $(TASK) --headless --enable_cameras --video --video_length $(VIDEO_LEN) --num_envs 64 --max_iterations 3"'
	ssh $(INSTANCE) 'rm -rf ~/vid && mkdir -p ~/vid && docker exec $(CONTAINER) bash -lc "find $(ISAACLAB)/logs -name \"*.mp4\" -newermt \"-1 hour\"" | while read f; do docker cp $(CONTAINER):"$$f" ~/vid/; done'
	rsync -az $(INSTANCE):~/vid/ reference/videos/
	@echo "video(s) in reference/videos/"
