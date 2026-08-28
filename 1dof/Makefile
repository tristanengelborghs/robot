.PHONY: help install install-sim install-hand hand-model data smoke hand-smoke test train eval stats clean

help:
	@echo "install      core deps (train + develop, no simulator)"
	@echo "install-sim  LIBERO + robosuite, pinned (needed only for eval)"
	@echo "install-hand mediapipe + opencv (needed only to track real video)"
	@echo "hand-model   download the 7MB hand landmark model (once, for real video)"
	@echo "data         download LIBERO demos + build the language cache"
	@echo "smoke        synthetic data -> 600 training steps; loss must fall"
	@echo "hand-smoke   synthetic hand demos -> 600 training steps; no camera needed"
	@echo "test         unit tests (fast, no simulator, no GPU)"

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
	bash scripts/download_data.sh libero_object libero_spatial
	python scripts/cache_language.py --data-root data/libero --suites libero_object libero_spatial

smoke:
	python scripts/make_synthetic_data.py --length 60
	python -m robobench.train --config configs/smoke.yaml train.steps=600 train.log_every=100

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
	python -m robobench.train --config configs/resnet_film_bc.yaml

clean:
	rm -rf runs results cache data/synthetic data/hand .pytest_cache
