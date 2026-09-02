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
    instruction_from_stem,
    read_instruction,
)
from robobench.data.normalizer import ActionNormalizer  # noqa: E402
from robobench.models import *  # noqa: E402,F401,F403
from robobench.models.base import ObsSpec  # noqa: E402
from robobench.registry import build_model, list_models  # noqa: E402
from robobench.stats import paired_permutation_test, pair  # noqa: E402

CHUNK, IMG = 4, 64

# Unit-test-sized constructor kwargs per model. `vla` gets the random offline
# stand-in for its backbone; everything else just shrinks.
SMALL_KWARGS = {
    "resnet_film_bc": {"hidden_dim": 32, "num_layers": 1, "num_heads": 2, "vision_width": 8, "num_keypoints": 8},
    "vla": {"hidden_dim": 32, "backbone": "tiny"},
}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    subprocess.run(
        [sys.executable, "scripts/make_synthetic_data.py",
         "--out", str(root / "libero_object"), "--tasks", "3", "--demos", "2",
         "--length", "12", "--image-size", str(IMG), "--cache", str(root / "lang.npz")],
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


def test_batch_carries_the_instruction_string(dataset):
    """Models with their own text encoder read `text`; it must line up with task_id."""
    batch = collate([dataset[i] for i in range(3)])
    assert isinstance(batch["text"], list) and len(batch["text"]) == 3
    for text, task_id in zip(batch["text"], batch["task_id"].tolist()):
        assert text and text == dataset.instructions[task_id]


def test_instruction_from_stem_mirrors_libero():
    """LIBERO derives `task.language` from the BDDL file name; the demo files are
    named the same way, so the stem is the spelling the evaluator will use."""
    assert instruction_from_stem("open_the_middle_drawer_of_the_cabinet_demo") == \
        "open the middle drawer of the cabinet"
    assert instruction_from_stem("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo") == \
        "turn on the stove and put the moka pot on it"
    assert instruction_from_stem("KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo") == \
        "close the top drawer of the cabinet"


def test_instruction_from_stem_agrees_with_installed_libero():
    benchmark = pytest.importorskip("libero.libero.benchmark")
    # get_benchmark_dict() also lists 'libero_100', which cannot be instantiated.
    for suite_name in benchmark.libero_suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            assert instruction_from_stem(task.name + "_demo") == task.language, (suite_name, task.name)


def test_instruction_source_selects_the_spelling(synthetic):
    import h5py
    p = sorted((synthetic / "libero_object").glob("*.hdf5"))[0]
    with h5py.File(p, "r") as f:
        by_name = read_instruction(f, p.stem, "filename")
        by_attr = read_instruction(f, p.stem, "hdf5")
        with pytest.raises(ValueError, match="instruction source"):
            read_instruction(f, p.stem, "bddl")
    assert by_name == instruction_from_stem(p.stem)
    assert by_attr  # the synthetic files carry problem_info as LIBERO's do


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
    kwargs = SMALL_KWARGS.get(name, {"hidden_dim": 32})
    model = build_model(name, obs_spec=spec, action_dim=7, chunk_size=CHUNK, **kwargs)

    pred = model(batch["images"], batch["proprio"], batch["lang"])
    assert pred.shape == (4, CHUNK, 7), f"{name} returned {tuple(pred.shape)}"

    out = model.loss(batch)
    assert "loss" in out and torch.isfinite(out["loss"])
    out["loss"].backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{name} produced no gradients"
    assert all(torch.isfinite(g).all() for g in grads)


# ---- the VLA: everything that can be checked without the real backbone ----------


def _tiny_vla(**kw):
    spec = ObsSpec(cameras=["agentview", "wrist"], image_size=IMG, proprio_dim=9, lang_dim=512)
    return build_model("vla", obs_spec=spec, action_dim=7, chunk_size=CHUNK, hidden_dim=32, backbone="tiny", **kw)


def test_vla_prompt_matches_smolvlm_layout():
    """One <image> expands to fake/global/<image>*n/fake, and the cache is keyed by string."""
    model = _tiny_vla()
    ids = model._prompt_ids("pick up the block")
    assert int((ids == model.image_token_id).sum()) == model.n_img_tok * 2
    fake = model.tok.convert_tokens_to_ids("<fake_token_around_image>")
    assert int((ids == fake).sum()) == 4
    assert model._prompt_ids("pick up the block") is ids
    assert not torch.equal(model._prompt_ids("open the drawer"), ids)


def test_vla_left_pads_and_reserves_the_tail(dataset):
    """Different instruction lengths in one batch: queries must still be the last
    n_tail positions of every row, attended, and the output finite (no NaN from
    fully-masked padding rows)."""
    model = _tiny_vla().eval()
    batch = collate([dataset[i] for i in range(2)])
    text = ["a", "a much longer instruction than the other one"]
    ids, attn = model._batch_prompts(text, torch.device("cpu"))
    assert ids.shape == attn.shape
    assert attn[:, -model.n_tail:].all()
    assert attn[0, 0] == 0 and attn[1, 0] == 1
    assert (ids[:, -model.n_tail:] == model.pad_id).all()
    pred = model.predict(batch["images"], batch["proprio"], batch["lang"], text=text)
    assert pred.shape == (2, CHUNK, 7) and torch.isfinite(pred).all()


def test_vla_reads_the_text_not_the_embedding(dataset):
    """Same images, different instruction -> different actions. Same instruction,
    different cached embedding -> identical actions. That is the whole point."""
    model = _tiny_vla().eval()
    batch = collate([dataset[i] for i in range(2)])
    a = model.predict(batch["images"], batch["proprio"], batch["lang"], text=["put the bowl on the stove"] * 2)
    b = model.predict(batch["images"], batch["proprio"], batch["lang"], text=["turn on the stove"] * 2)
    c = model.predict(batch["images"], batch["proprio"], torch.zeros_like(batch["lang"]),
                      text=["put the bowl on the stove"] * 2)
    assert not torch.allclose(a, b)
    assert torch.equal(a, c)
    blank = ablate.apply({**batch, "text": ["put the bowl on the stove"] * 2}, "no_lang")
    d = model.predict(blank["images"], blank["proprio"], blank["lang"], text=blank["text"])
    assert torch.isfinite(d).all() and not torch.allclose(a, d)


def test_vla_black_frames_are_frames_not_padding(dataset):
    """The backbone's own image path drops all-zero images; ours must not, or
    `no_vision` would silently change the token count."""
    model = _tiny_vla().eval()
    batch = ablate.apply(collate([dataset[i] for i in range(2)]), "no_vision")
    pred = model.predict(batch["images"], batch["proprio"], batch["lang"], text=batch["text"])
    assert pred.shape == (2, CHUNK, 7) and torch.isfinite(pred).all()


def test_vla_checkpoint_holds_only_what_trained():
    model = _tiny_vla()
    sd = model.trainable_state_dict()
    assert sd and all(not k.endswith(".base.weight") for k in sd)
    assert any(".A" in k for k in sd) and any(k.startswith("action_queries") for k in sd)
    assert not any("vision_model" in k for k in sd)
    n_backbone = sum(p.numel() for n, p in model.named_parameters() if n.startswith("backbone."))
    n_train = sum(v.numel() for v in sd.values())
    assert n_train < n_backbone

    fresh = _tiny_vla()
    result = fresh.load_state_dict(sd, strict=False)
    assert not result.unexpected_keys
    frozen = {n for n, p in fresh.named_parameters() if not p.requires_grad} | {n for n, _ in fresh.named_buffers()}
    assert all(k in frozen for k in result.missing_keys)
    for k, v in sd.items():
        assert torch.equal(fresh.state_dict()[k], v)


def test_vla_frozen_mode_trains_only_the_head():
    model = _tiny_vla(finetune="frozen")
    assert all(not n.startswith("backbone.") for n, p in model.named_parameters() if p.requires_grad)
    with pytest.raises(ValueError):
        _tiny_vla(finetune="full")


def test_vla_lora_starts_as_identity():
    """B is zero-initialised, so the wrapped model equals the pretrained one at step 0."""
    from robobench.models.vla import LoRALinear
    base = torch.nn.Linear(8, 6)
    lora = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert torch.equal(lora(x), base(x))
    lora.B.data.normal_()
    assert not torch.equal(lora(x), base(x))


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


def test_no_lang_blanks_the_instruction_string(dataset):
    """A model that tokenises its instruction must be ablated as thoroughly as one
    reading the cached embedding."""
    batch = collate([dataset[i] for i in range(2)])
    out = ablate.apply(batch, "no_lang")
    assert out["text"] == ["", ""]
    assert all(batch["text"]), "ablation mutated the original batch in place"
    out = ablate.apply(batch, "no_proprio")
    assert out["text"] == batch["text"]


def test_switching_model_name_starts_the_model_block_fresh(tmp_path):
    """Every key under model: is a constructor kwarg of one architecture. The
    default ResNet's must not leak into a run that picks the VLA."""
    from robobench.config import load_config

    cfg = load_config(None, ["model.hidden_dim=32", "model.name=vla", "model.backbone=tiny"])
    assert cfg["model"] == {"name": "vla", "hidden_dim": 32, "backbone": "tiny"}

    f = tmp_path / "vla.yaml"
    f.write_text("model:\n  name: vla\n  lora_rank: 4\n")
    cfg = load_config(str(f), ["train.steps=3"])
    assert cfg["model"] == {"name": "vla", "lora_rank": 4}
    assert cfg["train"]["steps"] == 3

    same = load_config(None, ["model.num_layers=2"])
    assert same["model"]["name"] == "resnet_film_bc" and same["model"]["num_layers"] == 2
    assert "num_keypoints" in same["model"], "same architecture keeps its defaults"


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
