"""Batching and checkpoints.

The checkpoint tests matter more than they look. A policy saved without the
normalisation it was trained under is not a policy: the weights expect inputs
scaled a particular way, and nothing about loading them would complain if the
scaling were lost. That failure produces a policy that runs, acts confidently,
and is wrong -- so the round trip is checked here rather than discovered during
an evaluation on a rented GPU.
"""

import numpy as np
import pytest
import torch

from harness.demo_data import Normalizer
from harness.diffusion import DiffusionConfig, DiffusionPolicy
from harness.training import (
    CHECKPOINT_VERSION,
    iterate_batches,
    load_checkpoint,
    pick_device,
    save_checkpoint,
)

CONFIG = DiffusionConfig(obs_dim=5, action_dim=3, horizon=4, hidden_dim=32, hidden_layers=1, num_steps=10)


def test_batches_cover_every_item_exactly_once():
    rng = np.random.default_rng(0)
    seen = np.concatenate(list(iterate_batches(10, 3, rng)))
    assert sorted(seen.tolist()) == list(range(10))


def test_the_last_short_batch_is_kept():
    # Dropping it is the usual convention and quietly discards up to a batch of
    # data every epoch -- on a hundred demonstrations that is not nothing.
    rng = np.random.default_rng(0)
    sizes = [len(batch) for batch in iterate_batches(10, 4, rng)]
    assert sizes == [4, 4, 2]


def test_shuffling_can_be_turned_off():
    rng = np.random.default_rng(0)
    ordered = np.concatenate(list(iterate_batches(6, 2, rng, shuffle=False)))
    assert ordered.tolist() == list(range(6))


def test_a_batch_size_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        list(iterate_batches(4, 0, np.random.default_rng(0)))


def test_a_checkpoint_round_trips_weights_and_normalisers(tmp_path):
    torch.manual_seed(0)
    policy = DiffusionPolicy(CONFIG)
    obs_norm = Normalizer(mean=np.arange(5, dtype=float), scale=np.full(5, 2.0))
    action_norm = Normalizer(mean=np.zeros(3), scale=np.full(3, 0.5))

    save_checkpoint(tmp_path / "policy.pt", policy, obs_norm, action_norm, extra={"epoch": 7})
    loaded, loaded_obs, loaded_action, extra = load_checkpoint(tmp_path / "policy.pt")

    assert loaded.cfg == CONFIG
    assert extra["epoch"] == 7
    assert loaded_obs.mean == pytest.approx(obs_norm.mean)
    assert loaded_obs.scale == pytest.approx(obs_norm.scale)
    assert loaded_action.scale == pytest.approx(action_norm.scale)

    for before, after in zip(policy.state_dict().values(), loaded.state_dict().values()):
        assert torch.allclose(before, after)


def test_a_loaded_policy_predicts_what_the_saved_one_did(tmp_path):
    # The point of the round trip: identical outputs, not merely identical
    # tensors on disk.
    torch.manual_seed(0)
    policy = DiffusionPolicy(CONFIG)
    policy.eval()

    save_checkpoint(
        tmp_path / "policy.pt",
        policy,
        Normalizer(np.zeros(5), np.ones(5)),
        Normalizer(np.zeros(3), np.ones(3)),
    )
    loaded, _, _, _ = load_checkpoint(tmp_path / "policy.pt")
    loaded.eval()

    noisy = torch.randn(2, CONFIG.horizon, CONFIG.action_dim)
    obs = torch.randn(2, CONFIG.obs_dim)
    t = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        assert torch.allclose(policy(noisy, obs, t), loaded(noisy, obs, t))


def test_a_checkpoint_from_another_layout_is_refused(tmp_path):
    # Loading weights into a mismatched architecture usually succeeds and then
    # behaves strangely, which is worse than refusing.
    path = tmp_path / "old.pt"
    torch.save({"version": CHECKPOINT_VERSION + 1, "config": {}, "state_dict": {}}, path)

    with pytest.raises(ValueError, match="checkpoint version"):
        load_checkpoint(path)


def test_normalisation_survives_the_round_trip_numerically(tmp_path):
    values = np.random.default_rng(0).normal(size=(20, 5))
    obs_norm = Normalizer.fit(values)

    policy = DiffusionPolicy(CONFIG)
    save_checkpoint(tmp_path / "p.pt", policy, obs_norm, Normalizer(np.zeros(3), np.ones(3)))
    _, loaded_obs, _, _ = load_checkpoint(tmp_path / "p.pt")

    assert loaded_obs.normalize(values) == pytest.approx(obs_norm.normalize(values))


def test_device_selection_returns_something_usable():
    device = pick_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}
    assert pick_device("cpu").type == "cpu"
