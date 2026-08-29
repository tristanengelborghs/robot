"""Print the shapes in a recorded demonstration file, and plot the end-effector.

Component 2's two laptop-side deliverables in one command::

    PYTHONPATH=. .venv/bin/python scripts/inspect_dataset.py datasets/stack.hdf5
    PYTHONPATH=. .venv/bin/python scripts/inspect_dataset.py datasets/stack.hdf5 --plot

No simulator, no GPU box: an HDF5 file is an HDF5 file, so a dataset recorded
months ago stays inspectable with the instance stopped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import dataset  # noqa: E402


def plot_ee_trajectories(path: Path, episodes: list[str], out: Path) -> Path:
    """Plot x, y and z of the end-effector against time, one row per axis."""
    import matplotlib

    matplotlib.use("Agg")  # this runs over ssh and in CI as often as not
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    term = None
    for name in episodes:
        term, values = dataset.ee_positions(path, name)
        steps = range(len(values))
        for axis, label in zip(axes, "xyz"):
            axis.plot(steps, values[:, "xyz".index(label)], linewidth=1.0, label=name)

    for axis, label in zip(axes, "xyz"):
        axis.set_ylabel(f"{label} [m]")
        axis.grid(alpha=0.3)
    axes[0].set_title(f"end-effector position over time -- obs/{term} -- {path.name}")
    axes[-1].set_xlabel("environment step (20 Hz)")
    if len(episodes) > 1:
        axes[0].legend(fontsize="small", ncol=min(len(episodes), 5))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="path to a recorded .hdf5 file")
    parser.add_argument("--plot", action="store_true", help="also write an end-effector trajectory plot")
    parser.add_argument("--plot-file", type=Path, default=None, help="where to write it (default: alongside the data)")
    parser.add_argument("--episode", action="append", default=None, help="limit to these episodes (repeatable)")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"no such dataset: {args.dataset}", file=sys.stderr)
        return 1

    summary = dataset.summarize(args.dataset)
    print(summary.format())

    if not summary.episodes:
        print("\nno episodes in this file -- nothing to plot", file=sys.stderr)
        return 1

    if args.plot:
        episodes = args.episode or [ep.name for ep in summary.episodes]
        out = args.plot_file or args.dataset.with_suffix(".ee.png")
        try:
            written = plot_ee_trajectories(args.dataset, episodes, out)
        except KeyError as exc:
            print(f"\ncannot plot: {exc}", file=sys.stderr)
            return 1
        print(f"\nwrote {written}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
