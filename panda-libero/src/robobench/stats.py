"""Is the difference real?

The 2026 audit of manipulation benchmarks found only about a fifth of published
LIBERO improvements are provably significant from the reported numbers. Two
success rates and no interval is not a result. This module turns the per-episode
logs written by `robobench.eval` into an interval and a p-value.

    python -m robobench.stats results/baseline/libero_object_episodes.jsonl \
                              results/my_arch/libero_object_episodes.jsonl

Episodes are paired on (task_id, init_state_idx) — both policies see the identical
starting configuration — so the default test is a paired sign-flip permutation test,
which is considerably more powerful than comparing two independent means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_episodes(path: str | Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no episodes in {path}")
    return rows


def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
                 ) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _key(row: dict) -> Tuple:
    return (row["suite"], row["task_id"], row.get("init_state_idx", row["episode"]))


def pair(rows_a: List[dict], rows_b: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    a_map: Dict[Tuple, int] = {_key(r): r["success"] for r in rows_a}
    b_map: Dict[Tuple, int] = {_key(r): r["success"] for r in rows_b}
    shared = sorted(set(a_map) & set(b_map))
    if not shared:
        raise ValueError(
            "no shared (task, init state) pairs — the two runs used different "
            "evaluation configurations, so they are not comparable."
        )
    a = np.array([a_map[k] for k in shared], dtype=float)
    b = np.array([b_map[k] for k in shared], dtype=float)
    return a, b, shared


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 20_000, seed: int = 0) -> float:
    """Two-sided sign-flip test on the paired differences. Returns p."""
    rng = np.random.default_rng(seed)
    d = b - a
    observed = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    # +1 in numerator and denominator: the observed assignment is itself a valid
    # permutation, which keeps the test from ever reporting p = 0.
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                        alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compare(path_a: str, path_b: str, n_perm: int = 20_000, seed: int = 0) -> dict:
    rows_a, rows_b = load_episodes(path_a), load_episodes(path_b)
    a, b, shared = pair(rows_a, rows_b)

    lo_a, hi_a = bootstrap_ci(a, seed=seed)
    lo_b, hi_b = bootstrap_ci(b, seed=seed)
    d_lo, d_hi = paired_bootstrap_ci(a, b, seed=seed)
    p = paired_permutation_test(a, b, n_perm=n_perm, seed=seed)

    n_tasks = len({k[1] for k in shared})
    return {
        "n_paired_episodes": len(shared),
        "n_tasks": n_tasks,
        "a": {"path": str(path_a), "success_rate": float(a.mean()), "ci95": [lo_a, hi_a]},
        "b": {"path": str(path_b), "success_rate": float(b.mean()), "ci95": [lo_b, hi_b]},
        "delta": float(b.mean() - a.mean()),
        "delta_ci95": [d_lo, d_hi],
        "p_value": p,
        "significant_at_0.05": bool(p < 0.05),
        # Both policies fail / both succeed carry no information for a paired test;
        # only the discordant pairs do. A tiny count here means the comparison is
        # underpowered no matter what the p-value says.
        "discordant_pairs": int(np.sum(a != b)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two runs from their per-episode logs.")
    ap.add_argument("baseline", help="episodes.jsonl of the baseline run")
    ap.add_argument("candidate", help="episodes.jsonl of the new architecture")
    ap.add_argument("--permutations", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print the raw result dict")
    args = ap.parse_args()

    r = compare(args.baseline, args.candidate, args.permutations, args.seed)
    if args.json:
        print(json.dumps(r, indent=2))
        return

    a, b = r["a"], r["b"]
    print(f"\n  paired on {r['n_paired_episodes']} episodes across {r['n_tasks']} tasks")
    print(f"  discordant pairs: {r['discordant_pairs']}  (the only ones that carry signal)\n")
    print(f"  baseline   {a['success_rate'] * 100:6.2f}%   95% CI [{a['ci95'][0] * 100:.1f}, {a['ci95'][1] * 100:.1f}]")
    print(f"  candidate  {b['success_rate'] * 100:6.2f}%   95% CI [{b['ci95'][0] * 100:.1f}, {b['ci95'][1] * 100:.1f}]")
    print(f"\n  delta      {r['delta'] * 100:+6.2f} pp   95% CI [{r['delta_ci95'][0] * 100:+.1f}, {r['delta_ci95'][1] * 100:+.1f}]")
    print(f"  p          {r['p_value']:.4f}  (paired sign-flip permutation, two-sided)")

    if r["significant_at_0.05"]:
        print("\n  -> distinguishable from noise at alpha = 0.05.")
    else:
        print("\n  -> NOT distinguishable from noise. More seeds or more episodes, or the gain isn't there.")


if __name__ == "__main__":
    main()
