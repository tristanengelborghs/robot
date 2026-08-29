"""Reading the HDF5 demonstration files Isaac Lab records.

Component 2's deliverable is partly "a script that prints each dataset's
observation and action shapes" and partly "a visualization of end-effector
position over time". Neither needs a simulator: a recorded dataset is just an
HDF5 file, so both live on the laptop where they can be run against demos
recorded weeks ago without renting anything.

The layout comes from Isaac Lab's own ``HDF5DatasetFileHandler`` (vendored at
reference/isaaclab_source/) and is robomimic-compatible:

.. code-block:: text

    /                     attrs: format_version
    /data                 attrs: total, env_args (JSON: env_name, type)
    /data/demo_0          attrs: num_samples, success, seed
    /data/demo_0/actions              (T, action_dim)
    /data/demo_0/obs/<term>           (T, ...)      one dataset per obs term
    /data/demo_0/initial_state/...    the reset state a replay starts from
    /data/demo_0/states/...           per-step state, for replay validation

``format_version`` is worth checking rather than assuming: version 0 files store
quaternions ``wxyz`` and version 1 files store them ``xyzw``, so anything that
reads an orientation out of a dataset has to know which it is holding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Observation terms that hold the end-effector position, best first. The task
# config names it eef_pos; a dataset from another task may not.
EE_POS_KEYS = ("eef_pos", "ee_pos", "ee_frame_pos", "end_effector_pos")


@dataclass(frozen=True)
class EpisodeSummary:
    """Shapes and metadata for one recorded episode."""

    name: str
    num_samples: int
    success: bool | None
    seed: int | None
    action_shape: tuple[int, ...] | None
    obs_shapes: dict[str, tuple[int, ...]]

    def format(self) -> str:
        outcome = {True: "success", False: "FAILED", None: "unlabelled"}[self.success]
        seed = "" if self.seed is None else f", seed {self.seed}"

        lines = [f"{self.name}: {self.num_samples} steps, {outcome}{seed}"]
        lines.append(f"  actions {self.action_shape}")
        for key, shape in sorted(self.obs_shapes.items()):
            lines.append(f"  obs/{key} {shape}")
        return "\n".join(lines)


@dataclass(frozen=True)
class DatasetSummary:
    """Shapes and metadata for a whole dataset file."""

    path: Path
    env_name: str | None
    format_version: int
    episodes: list[EpisodeSummary]

    @property
    def quaternion_order(self) -> str:
        """Which quaternion convention orientations in this file are stored in."""
        return "xyzw" if self.format_version >= 1 else "wxyz (legacy)"

    def format(self) -> str:
        head = [
            f"{self.path}",
            f"  env        {self.env_name or '(unnamed)'}",
            f"  version    {self.format_version} -- quaternions {self.quaternion_order}",
            f"  episodes   {len(self.episodes)}",
        ]
        if self.episodes:
            lengths = [ep.num_samples for ep in self.episodes]
            head.append(f"  steps      {sum(lengths)} total, {min(lengths)}-{max(lengths)} per episode")
        return "\n".join(head + [""] + [ep.format() for ep in self.episodes])


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("reading demonstration datasets needs h5py: run `make install`") from exc
    return h5py


def _flatten_shapes(group, prefix: str = "") -> dict[str, tuple[int, ...]]:
    """Depth-first walk of an HDF5 group, mapping ``a/b`` paths to shapes."""
    h5py = _require_h5py()
    shapes: dict[str, tuple[int, ...]] = {}
    for key in group:
        path = f"{prefix}{key}"
        item = group[key]
        if isinstance(item, h5py.Group):
            shapes.update(_flatten_shapes(item, prefix=f"{path}/"))
        else:
            shapes[path] = tuple(item.shape)
    return shapes


def summarize(path: str | Path) -> DatasetSummary:
    """Read every episode's shapes and metadata out of a dataset file."""
    h5py = _require_h5py()
    path = Path(path)
    episodes: list[EpisodeSummary] = []

    with h5py.File(path, "r") as handle:
        format_version = int(handle.attrs.get("format_version", 0))
        data = handle["data"]
        env_args = json.loads(data.attrs["env_args"]) if "env_args" in data.attrs else {}

        # demo_10 must not sort before demo_2.
        for name in sorted(data.keys(), key=_demo_order):
            group = data[name]
            obs_shapes = _flatten_shapes(group["obs"]) if "obs" in group else {}
            episodes.append(
                EpisodeSummary(
                    name=name,
                    num_samples=int(group.attrs.get("num_samples", 0)),
                    success=_maybe_bool(group.attrs.get("success")),
                    seed=_maybe_int(group.attrs.get("seed")),
                    action_shape=tuple(group["actions"].shape) if "actions" in group else None,
                    obs_shapes=obs_shapes,
                )
            )

    return DatasetSummary(
        path=path,
        env_name=env_args.get("env_name") or None,
        format_version=format_version,
        episodes=episodes,
    )


def ee_positions(path: str | Path, episode: str, key: str | None = None) -> tuple[str, np.ndarray]:
    """End-effector position over time for one episode.

    Args:
        path: The dataset file.
        episode: Episode group name, e.g. ``demo_0``.
        key: Observation term to read. Defaults to the first of
            :data:`EE_POS_KEYS` the episode actually has.

    Returns:
        The term name used, and a ``(T, 3)`` array.

    Raises:
        KeyError: if no end-effector term can be found, naming what was there
            instead -- a dataset from an unfamiliar task is the likely cause,
            and guessing silently would produce a plot of the wrong quantity.
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        group = handle["data"][episode]
        obs = group["obs"]
        candidates = [k for k in (EE_POS_KEYS if key is None else (key,)) if k in obs]
        if not candidates:
            raise KeyError(
                f"{episode} has no end-effector position term "
                f"(looked for {list(EE_POS_KEYS if key is None else (key,))}, found {sorted(obs.keys())})"
            )
        name = candidates[0]
        values = np.asarray(obs[name][()], dtype=np.float64)

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"obs/{name} has shape {values.shape}, expected (T, 3)")
    return name, values


def _demo_order(name: str) -> tuple[int, str]:
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix), name) if suffix.isdigit() else (2**31, name)


def _maybe_bool(value) -> bool | None:
    return None if value is None else bool(value)


def _maybe_int(value) -> int | None:
    return None if value is None else int(value)
