.PHONY: help install assets test smoke train eval distill view clean

help:
	@echo "install   deps into .venv (torch, mujoco, gymnasium)"
	@echo "assets    fetch Shadow/Allegro hand models (pinned Menagerie commit)"
	@echo "test      unit tests: toy hand only — no downloads, no GPU, <1 min"
	@echo "smoke     ~5 min CPU: PPO on easy z-goals; successes/ep must rise (see README)"
	@echo "train     full teacher run (long; see README for compute honesty)"
	@echo "distill   teacher -> proprio+tactile student via DAgger"
	@echo "view      watch a policy (macOS: runs under mjpython)"

install:
	python3.11 -m venv .venv && .venv/bin/pip install -U pip && \
	.venv/bin/pip install mujoco torch gymnasium numpy pyyaml pytest tqdm imageio && \
	.venv/bin/pip install -e .

assets:
	.venv/bin/python scripts/fetch_assets.py

test:
	.venv/bin/python -m pytest tests/ -q

smoke:
	.venv/bin/python -m dexhand.train --config configs/smoke.yaml

train:
	.venv/bin/python -m dexhand.train

eval:
	.venv/bin/python -m dexhand.eval --ckpt runs/$(RUN)/ckpt_last.pt --episodes 50

distill:
	.venv/bin/python -m dexhand.distill --ckpt runs/$(RUN)/ckpt_last.pt

view:
	.venv/bin/mjpython -m dexhand.view --ckpt runs/$(RUN)/ckpt_last.pt

clean:
	rm -rf runs .pytest_cache
