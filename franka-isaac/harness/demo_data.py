"""Turning recorded demonstrations into something a policy can train on.

This is the step between an HDF5 file and a training loop, and it is deliberately
NumPy only: no torch, no simulator. Everything here is reshaping and arithmetic,
so it runs and is tested on the laptop, and the parts that are easy to get
quietly wrong -- which observations go in, how actions are chunked, what the
normalisation is fitted on, where the validation split is drawn -- are checked
without a GPU.

Three decisions are worth stating rather than discovering later.

**Actions come in chunks.** A diffusion policy predicts a short sequence of
future actions from one observation, not a single step. Chunking is most of why
these policies are smooth: predicting eight steps at once and executing a few of
them keeps the arm from dithering between conflicting single-step guesses.

**The validation split is by episode, not by sample.** Consecutive steps in one
demonstration are almost identical, so splitting at random puts near-copies of
validation samples in the training set and the validation loss becomes a
flattering lie. Whole episodes are held out instead.

**Normalisation statistics come from the training split alone.** Fitting them on
everything leaks the validation set's distribution into training, which is the
same mistake in a quieter form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The observation terms a state-based policy sees, in this order. Deliberately
# not everything the recorder stored: `object` repeats cube poses the other
# terms already carry, and feeding a policy the same number three times only
# teaches it that the number matters three times as much.
DEFAULT_OBS_KEYS = ("eef_pos", "eef_quat", "gripper_pos", "cube_positions")


@dataclass(frozen=True)
class Episode:
    """One demonstration: aligned observations and actions."""

    name: str
    obs: np.ndarray  # (T, obs_dim)
    actions: np.ndarray  # (T, action_dim)

    def __post_init__(self):
        if len(self.obs) != len(self.actions):
            raise ValueError(f"{self.name}: {len(self.obs)} observations against {len(self.actions)} actions")

    def __len__(self) -> int:
        return len(self.obs)


@dataclass(frozen=True)
class Normalizer:
    """Per-dimension mean and standard deviation.

    Fitted on the training split only. Dimensions that never vary -- a gripper
    that was open for an entire dataset, say -- would divide by zero, so their
    scale is forced to one and they pass through unchanged rather than becoming
    infinities that quietly poison every gradient.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, floor: float = 1e-4) -> Normalizer:
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < floor, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


def load_episodes(
    path: str | Path,
    obs_keys: tuple[str, ...] = DEFAULT_OBS_KEYS,
) -> list[Episode]:
    """Read a recorded dataset into episodes of flat observation vectors.

    Args:
        path: An HDF5 file written by Isaac Lab's recorder.
        obs_keys: Which observation terms to concatenate, in order.

    Raises:
        KeyError: if an episode lacks a requested term, naming what it does
            have. A dataset from a different task is the likely cause, and
            silently training on the wrong columns is the worst outcome.
    """
    import h5py

    episodes: list[Episode] = []
    with h5py.File(Path(path), "r") as handle:
        data = handle["data"]
        for name in sorted(data.keys(), key=lambda n: int(n.rsplit("_", 1)[-1])):
            group = data[name]
            observations = group["obs"]

            missing = [key for key in obs_keys if key not in observations]
            if missing:
                raise KeyError(f"{name} has no observation term(s) {missing}; it has {sorted(observations.keys())}")

            columns = [np.asarray(observations[key][()], dtype=np.float32) for key in obs_keys]
            episodes.append(
                Episode(
                    name=name,
                    obs=np.concatenate(columns, axis=1),
                    actions=np.asarray(group["actions"][()], dtype=np.float32),
                )
            )

    if not episodes:
        raise ValueError(f"{path} contains no episodes")
    return episodes


def chunk_episode(episode: Episode, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair each observation with the next ``horizon`` actions.

    The last few steps of an episode have fewer than ``horizon`` actions left.
    Rather than drop them -- they are the end of the task, the most valuable part
    of a demonstration -- the final action is repeated to fill the chunk. A
    policy learning "hold still once finished" from that padding is correct
    behaviour, not an artefact.

    Returns:
        Observations ``(T, obs_dim)`` and action chunks ``(T, horizon, action_dim)``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")

    steps = len(episode)
    indices = np.arange(steps)[:, None] + np.arange(horizon)[None, :]
    indices = np.minimum(indices, steps - 1)  # clamp: repeat the last action
    return episode.obs, episode.actions[indices]


@dataclass
class DemoDataset:
    """Observations paired with action chunks, ready for a training loop."""

    obs: np.ndarray  # (N, obs_dim)
    action_chunks: np.ndarray  # (N, horizon, action_dim)
    episode_names: list[str]

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[1]

    @property
    def action_dim(self) -> int:
        return self.action_chunks.shape[2]

    @property
    def horizon(self) -> int:
        return self.action_chunks.shape[1]

    def __len__(self) -> int:
        return len(self.obs)

    def format(self) -> str:
        return (
            f"{len(self)} samples from {len(self.episode_names)} episode(s): "
            f"obs {self.obs_dim}, actions {self.action_dim} x {self.horizon} steps"
        )


def build_dataset(episodes: list[Episode], horizon: int) -> DemoDataset:
    """Chunk every episode and stack them into one training set."""
    observations, chunks = [], []
    for episode in episodes:
        episode_obs, episode_chunks = chunk_episode(episode, horizon)
        observations.append(episode_obs)
        chunks.append(episode_chunks)

    return DemoDataset(
        obs=np.concatenate(observations, axis=0),
        action_chunks=np.concatenate(chunks, axis=0),
        episode_names=[episode.name for episode in episodes],
    )


def split_episodes(
    episodes: list[Episode],
    validation_episodes: int = 1,
    seed: int = 0,
) -> tuple[list[Episode], list[Episode]]:
    """Hold out whole episodes for validation.

    By episode and not by sample: consecutive steps of one demonstration are
    nearly identical, so a random split would put near-duplicates of the
    validation samples into training and report a validation loss that means
    nothing.

    Raises:
        ValueError: if holding that many out would leave nothing to train on.
    """
    if validation_episodes < 0:
        raise ValueError(f"validation_episodes must not be negative, got {validation_episodes}")
    if validation_episodes >= len(episodes):
        raise ValueError(
            f"cannot hold out {validation_episodes} of {len(episodes)} episodes and still have any to train on"
        )

    order = np.random.default_rng(seed).permutation(len(episodes))
    held_out = {int(i) for i in order[:validation_episodes]}

    train = [episode for i, episode in enumerate(episodes) if i not in held_out]
    validation = [episode for i, episode in enumerate(episodes) if i in held_out]
    return train, validation
