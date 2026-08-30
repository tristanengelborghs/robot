"""Batching and checkpoints, shared by the training scripts.

Small, but the checkpoint half is worth getting right: a policy saved without
the normalisation it was trained under is not a policy, it is a set of weights
that expects inputs nobody can reconstruct. Every checkpoint here carries the
architecture, the observation normaliser and the action normaliser together, so
loading one cannot silently produce a policy that acts on differently-scaled
numbers than it was trained on.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from harness.demo_data import Normalizer
from harness.diffusion import DiffusionConfig, DiffusionPolicy

# Bumped whenever the checkpoint layout changes, so an old file fails loudly
# rather than loading into a mismatched policy.
CHECKPOINT_VERSION = 1


def iterate_batches(count: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    """Yield index arrays covering ``count`` items once.

    The last batch is short rather than dropped. Dropping it is the usual
    convention and quietly discards up to a batch of data every epoch, which on
    a hundred demonstrations is not nothing.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    order = rng.permutation(count) if shuffle else np.arange(count)
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def save_checkpoint(
    path: str | Path,
    policy: DiffusionPolicy,
    obs_normalizer: Normalizer,
    action_normalizer: Normalizer,
    extra: dict | None = None,
) -> Path:
    """Write weights, architecture and both normalisers as one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "config": asdict(policy.cfg),
            "state_dict": policy.state_dict(),
            "obs_mean": obs_normalizer.mean,
            "obs_scale": obs_normalizer.scale,
            "action_mean": action_normalizer.mean,
            "action_scale": action_normalizer.scale,
            "extra": extra or {},
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[DiffusionPolicy, Normalizer, Normalizer, dict]:
    """Rebuild a policy and the normalisation it was trained under.

    Raises:
        ValueError: if the checkpoint was written by a different layout version.
            Loading weights into a mismatched architecture usually succeeds and
            then behaves strangely, which is far worse than refusing.
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)

    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"checkpoint version {version}, expected {CHECKPOINT_VERSION}; retrain or migrate it")

    policy = DiffusionPolicy(DiffusionConfig(**payload["config"]))
    policy.load_state_dict(payload["state_dict"])
    policy.to(device)

    obs_normalizer = Normalizer(mean=payload["obs_mean"], scale=payload["obs_scale"])
    action_normalizer = Normalizer(mean=payload["action_mean"], scale=payload["action_scale"])
    return policy, obs_normalizer, action_normalizer, payload.get("extra", {})


def pick_device(requested: str = "auto") -> torch.device:
    """CUDA on the box, MPS on the laptop, CPU if neither is there."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
