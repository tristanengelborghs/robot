"""Demonstrations into training tensors, checked on synthetic data.

The failures this guards against are all silent ones. Chunked actions that are
off by a step, normalisation fitted on the validation set, a split that puts
near-duplicate frames on both sides -- none of them raise, and all of them
produce a training curve that looks fine and a policy that is not.
"""

import numpy as np
import pytest

from harness.demo_data import (
    DemoDataset,
    Episode,
    Normalizer,
    build_dataset,
    chunk_episode,
    load_episodes,
    split_episodes,
)
from tests.test_dataset import write_dataset


def make_episode(name: str = "demo_0", steps: int = 10, obs_dim: int = 3, action_dim: int = 7) -> Episode:
    """An episode whose values encode their own index, so slips are visible."""
    obs = np.arange(steps * obs_dim, dtype=np.float32).reshape(steps, obs_dim)
    actions = np.arange(steps * action_dim, dtype=np.float32).reshape(steps, action_dim)
    return Episode(name=name, obs=obs, actions=actions)


def test_an_episode_must_have_matching_observations_and_actions():
    with pytest.raises(ValueError, match="observations against"):
        Episode(name="demo_0", obs=np.zeros((10, 3)), actions=np.zeros((9, 7)))


def test_a_chunk_holds_the_next_actions_in_order():
    episode = make_episode(steps=10, action_dim=2)
    _, chunks = chunk_episode(episode, horizon=4)

    assert chunks.shape == (10, 4, 2)
    # The chunk at step 0 is actions 0, 1, 2, 3 -- not 1, 2, 3, 4.
    assert chunks[0, 0] == pytest.approx(episode.actions[0])
    assert chunks[0, 3] == pytest.approx(episode.actions[3])
    assert chunks[5, 0] == pytest.approx(episode.actions[5])


def test_the_end_of_an_episode_is_padded_by_holding_the_last_action():
    # The final steps are the end of the task, the most valuable part of a
    # demonstration; dropping them to make the arithmetic tidy would be worse.
    episode = make_episode(steps=5, action_dim=2)
    _, chunks = chunk_episode(episode, horizon=4)

    last = episode.actions[-1]
    assert chunks[4, 0] == pytest.approx(last)
    assert chunks[4, 3] == pytest.approx(last), "padding repeats the last action"
    assert chunks[3, 3] == pytest.approx(last)


def test_a_horizon_of_one_is_plain_behaviour_cloning():
    episode = make_episode(steps=6, action_dim=2)
    _, chunks = chunk_episode(episode, horizon=1)
    assert chunks.shape == (6, 1, 2)
    assert chunks[:, 0] == pytest.approx(episode.actions)


def test_a_horizon_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        chunk_episode(make_episode(), horizon=0)


def test_building_a_dataset_stacks_every_episode():
    episodes = [make_episode("demo_0", steps=10), make_episode("demo_1", steps=7)]
    dataset = build_dataset(episodes, horizon=3)

    assert len(dataset) == 17
    assert dataset.obs_dim == 3
    assert dataset.action_dim == 7
    assert dataset.horizon == 3
    assert dataset.episode_names == ["demo_0", "demo_1"]


def test_the_split_holds_out_whole_episodes():
    # Consecutive frames are nearly identical, so a per-sample split would put
    # near-copies of the validation set into training.
    episodes = [make_episode(f"demo_{i}") for i in range(5)]
    train, validation = split_episodes(episodes, validation_episodes=2)

    assert len(train) == 3
    assert len(validation) == 2
    train_names = {e.name for e in train}
    assert train_names.isdisjoint({e.name for e in validation})


def test_the_split_is_reproducible_for_a_seed():
    episodes = [make_episode(f"demo_{i}") for i in range(5)]
    first = [e.name for e in split_episodes(episodes, 2, seed=7)[1]]
    second = [e.name for e in split_episodes(episodes, 2, seed=7)[1]]
    assert first == second


def test_holding_out_everything_is_rejected():
    episodes = [make_episode(f"demo_{i}") for i in range(3)]
    with pytest.raises(ValueError, match="still have any to train on"):
        split_episodes(episodes, validation_episodes=3)


def test_normaliser_centres_and_scales():
    values = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    normalizer = Normalizer.fit(values)

    normalized = normalizer.normalize(values)
    assert normalized.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-6)
    assert normalized.std(axis=0) == pytest.approx([1.0, 1.0], abs=1e-6)


def test_a_constant_dimension_passes_through_instead_of_exploding():
    # A gripper open for a whole dataset has zero variance. Dividing by it would
    # turn one column into infinities and poison every gradient.
    values = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    normalizer = Normalizer.fit(values)

    assert normalizer.scale[1] == 1.0
    assert np.isfinite(normalizer.normalize(values)).all()


def test_normalisation_round_trips():
    values = np.random.default_rng(0).normal(size=(50, 4))
    normalizer = Normalizer.fit(values)
    assert normalizer.denormalize(normalizer.normalize(values)) == pytest.approx(values, abs=1e-6)


def test_loading_a_recorded_dataset(tmp_path):
    path = write_dataset(tmp_path / "demos.hdf5", episodes=3, steps=20)
    episodes = load_episodes(path, obs_keys=("eef_pos", "gripper_pos"))

    assert len(episodes) == 3
    assert episodes[0].name == "demo_0"
    assert episodes[0].obs.shape == (20, 5), "3 for the position, 2 for the gripper"
    assert episodes[0].actions.shape == (20, 7)


def test_loading_names_the_terms_an_unfamiliar_dataset_actually_has(tmp_path):
    path = write_dataset(tmp_path / "demos.hdf5")
    with pytest.raises(KeyError) as excinfo:
        load_episodes(path, obs_keys=("eef_pos", "tactile_sensor"))

    message = str(excinfo.value)
    assert "tactile_sensor" in message
    assert "eef_pos" in message, "the error should say what is available"


def test_dataset_summary_is_readable():
    dataset = DemoDataset(
        obs=np.zeros((12, 3)),
        action_chunks=np.zeros((12, 8, 7)),
        episode_names=["demo_0"],
    )
    assert "12 samples" in dataset.format()
    assert "7 x 8 steps" in dataset.format()
