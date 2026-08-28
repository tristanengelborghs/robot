"""Seeding, provenance, logging. Small, but this is what makes a result auditable."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN autotuning picks different kernels run-to-run; disabling it costs a
        # little throughput and buys reproducible numbers.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def git_provenance() -> Dict[str, str]:
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "dirty": _run("git", "status", "--porcelain") != "",
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(module: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


class JsonlWriter:
    """Append-only per-episode / per-step log.

    Per-episode outcomes (not just the aggregate success rate) are what let anyone
    run a significance test on your result. Cheap to write, impossible to retrofit.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def human(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)
