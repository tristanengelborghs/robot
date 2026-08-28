"""Runs without a simulator, a dataset or a GPU. `pytest tests/ -q`.

Covers the parts that break silently: chunk padding at episode boundaries, action
normalization round-tripping, the ablations actually destroying their input, and
every registered model producing the right shape and a finite gradient.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robobench import ablate  # noqa: E402
from robobench.data.libero_dataset import (  # noqa: E402
    LiberoChunkDataset,
    collate,
    read_instruction,
)
from robobench.data.normalizer import ActionNormalizer  # noqa: E402
from robobench.models import *  # noqa: E402,F401,F403
from robobench.models.base import ObsSpec  # noqa: E402
from robobench.registry import build_model, list_models  # noqa: E402
from robobench.stats import paired_permutation_test, pair  # noqa: E402

CHUNK, IMG = 4, 64


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    subprocess.run(
        [sys.executable, "scripts/make_synthetic_data.py",
         "--out", str(root / "libero_object"), "--tasks", "3", "--demos", "2",
         "--length", "12", "--image-size", str(IMG)],
        check=True, capture_output=True,
    )
    return root


@pytest.fixture(scope="module")
def lang_cache(synthetic):
    """Real (non-zero) embeddings, keyed by the instruction strings in the files."""
    rng = np.random.default_rng(0)
    texts = [
        read_instruction_from(p) for p in sorted((synthetic / "libero_object").glob("*.hdf5"))
    ]
    return {t: rng.normal(0, 1, 512).astype(np.float32) for t in texts}


def read_instruction_from(path):
    import h5py
    with h5py.File(path, "r") as f:
        return read_instruction(f, path.stem)


@pytest.fixture(scope="module")
def dataset(synthetic, lang_cache):
    return LiberoChunkDataset(
        root=synthetic, suites=["libero_object"], chunk_size=CHUNK,
        image_size=IMG, augment=False, lang_cache=lang_cache,
    )


def test_missing_language_cache_is_refused(synthetic):
    """A zero language vector would silently produce a language-blind policy."""
    with pytest.raises(ValueError, match="no language cache"):
        LiberoChunkDataset(root=synthetic, suites=["libero_object"], chunk_size=CHUNK,
                           image_size=IMG, augment=False)


def test_dataset_shapes(dataset):
    meta = dataset.describe()
    assert meta["num_tasks"] == 3
    assert meta["num_samples"] == 3 * 2 * 12
    assert meta["action_dim"] == 7

    item = dataset[0]
    assert item["images"]["agentview"].shape == (3, IMG, IMG)
    assert item["images"]["agentview"].dtype == torch.uint8
    assert item["proprio"].shape == (9,)
    assert item["actions"].shape == (CHUNK, 7)
    assert item["mask"].shape == (CHUNK,)


def test_chunk_padding_at_episode_end(dataset):
    """The final timestep of a demo must be fully masked past the first step."""
    ends = [i for i, (_, _, t, ep_len) in enumerate(dataset.index) if t == ep_len - 1]
    item = dataset[ends[0]]
    assert item["mask"][0] == 1.0
    assert item["mask"][1:].sum() == 0.0, "padded chunk steps must not contribute to the loss"


def test_normalized_actions_in_range(dataset):
    for i in np.random.default_rng(0).integers(0, len(dataset), 40):
        a = dataset[int(i)]["actions"]
        assert torch.isfinite(a).all()
        assert a.min() >= -1.001 and a.max() <= 1.001


def test_normalizer_roundtrip():
    rng = np.random.default_rng(0)
    acts = rng.normal(0, 3, (500, 7)).astype(np.float32)
    n = ActionNormalizer.from_actions(acts)
    assert np.allclose(n.denormalize(n.normalize(acts)), acts, atol=1e-4)
    restored = ActionNormalizer.from_state_dict(n.state_dict())
    assert np.allclose(restored.normalize(acts), n.normalize(acts))


def test_normalizer_handles_constant_dim():
    acts = np.concatenate([np.random.randn(100, 6), np.ones((100, 1))], axis=-1).astype(np.float32)
    n = ActionNormalizer.from_actions(acts)
    assert np.isfinite(n.normalize(acts)).all()


@pytest.mark.parametrize("name", list_models())
def test_every_registered_model_trains(name, dataset):
    """Shape contract plus one real backward pass, for every architecture in the registry.
    A new model that passes this is wired correctly."""
    batch = collate([dataset[i] for i in range(4)])
    spec = ObsSpec(cameras=["agentview", "wrist"], image_size=IMG, proprio_dim=9, lang_dim=512)
    kwargs = {"hidden_dim": 32} if name != "resnet_film_bc" else {
        "hidden_dim": 32, "num_layers": 1, "num_heads": 2, "vision_width": 8, "num_keypoints": 8
    }
    model = build_model(name, obs_spec=spec, action_dim=7, chunk_size=CHUNK, **kwargs)

    pred = model(batch["images"], batch["proprio"], batch["lang"])
    assert pred.shape == (4, CHUNK, 7), f"{name} returned {tuple(pred.shape)}"

    out = model.loss(batch)
    assert "loss" in out and torch.isfinite(out["loss"])
    out["loss"].backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{name} produced no gradients"
    assert all(torch.isfinite(g).all() for g in grads)


def test_eval_mode_is_deterministic(dataset):
    batch = collate([dataset[i] for i in range(2)])
    spec = ObsSpec(cameras=["agentview", "wrist"], image_size=IMG, proprio_dim=9, lang_dim=512)
    model = build_model("resnet_film_bc", obs_spec=spec, action_dim=7, chunk_size=CHUNK,
                        hidden_dim=32, num_layers=1, num_heads=2, vision_width=8, num_keypoints=8)
    model.eval()
    a = model.predict(batch["images"], batch["proprio"], batch["lang"])
    b = model.predict(batch["images"], batch["proprio"], batch["lang"])
    assert torch.equal(a, b), "dropout is still active at rollout time"


@pytest.mark.parametrize("mode", [m for m in ablate.MODES if m != "none"])
def test_ablation_destroys_its_input(mode, dataset):
    batch = collate([dataset[i] for i in range(2)])
    out = ablate.apply(batch, mode)
    changed = (
        not torch.equal(out["lang"], batch["lang"])
        or not torch.equal(out["proprio"], batch["proprio"])
        or any(not torch.equal(out["images"][c], batch["images"][c]) for c in batch["images"])
    )
    assert changed, f"ablation '{mode}' was a no-op"
    assert torch.equal(batch["lang"], collate([dataset[i] for i in range(2)])["lang"]), \
        "ablation mutated the original batch in place"


def test_paired_stats_detect_a_real_difference():
    rows_a, rows_b = [], []
    rng = np.random.default_rng(0)
    for task in range(10):
        for ep in range(50):
            base = {"suite": "libero_object", "task_id": task, "init_state_idx": ep, "episode": ep}
            rows_a.append({**base, "success": int(rng.random() < 0.60)})
            rows_b.append({**base, "success": int(rng.random() < 0.85)})
    a, b, shared = pair(rows_a, rows_b)
    assert len(shared) == 500
    assert paired_permutation_test(a, b, n_perm=2000) < 0.05


def test_paired_stats_reject_noise():
    rows_a, rows_b = [], []
    rng = np.random.default_rng(1)
    for task in range(10):
        for ep in range(20):
            base = {"suite": "libero_object", "task_id": task, "init_state_idx": ep, "episode": ep}
            rows_a.append({**base, "success": int(rng.random() < 0.7)})
            rows_b.append({**base, "success": int(rng.random() < 0.71)})
    a, b, _ = pair(rows_a, rows_b)
    assert paired_permutation_test(a, b, n_perm=2000) > 0.05


def test_device_transfer_preserves_values(dataset):
    """Regression: `non_blocking=True` from pageable memory corrupts tensors on MPS.

    The copy returns before it completes, the source buffer is recycled, and the
    batch arrives as garbage — masks of ~1e-23 and an infinite language norm — which
    trains without erroring and produces a meaningless loss. Values must survive the
    move on whatever device this machine has.
    """
    from robobench.train import move_to
    from robobench.utils import pick_device

    device = pick_device("auto")
    batch = collate([dataset[i] for i in range(4)])
    moved = move_to(batch, device)

    for key in ("proprio", "lang", "actions", "mask"):
        assert torch.equal(moved[key].cpu(), batch[key]), f"{key} changed crossing to {device}"
    for cam in batch["images"]:
        assert torch.equal(moved["images"][cam].cpu(), batch["images"][cam])

    assert moved["mask"].min() >= 0.0 and moved["mask"].max() <= 1.0
    assert torch.isfinite(moved["lang"]).all()
