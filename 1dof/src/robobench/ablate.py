"""Input ablations — the cheapest credibility you can buy.

The 2026 benchmark audit found that a 90M model with no language encoder at all
matches state of the art on LIBERO's easier suites. So before claiming your
architecture uses the instruction, or the wrist camera, or proprioception, run the
same architecture with that input destroyed and show the score collapses.

The ablation is applied identically in training and in rollout evaluation, so what
you measure is a model that genuinely never had access to the input.
"""

from __future__ import annotations

from typing import Dict

import torch

MODES = ("none", "no_lang", "no_wrist", "no_agentview", "no_vision", "no_proprio")


def apply(batch: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
    if mode == "none":
        return batch
    if mode not in MODES:
        raise ValueError(f"unknown ablation '{mode}', expected one of {MODES}")

    out = dict(batch)
    out["images"] = dict(batch["images"])

    if mode == "no_lang":
        out["lang"] = torch.zeros_like(batch["lang"])
    elif mode == "no_proprio":
        out["proprio"] = torch.zeros_like(batch["proprio"])
    elif mode == "no_vision":
        for cam in out["images"]:
            out["images"][cam] = torch.zeros_like(out["images"][cam])
    elif mode == "no_wrist" and "wrist" in out["images"]:
        out["images"]["wrist"] = torch.zeros_like(out["images"]["wrist"])
    elif mode == "no_agentview" and "agentview" in out["images"]:
        out["images"]["agentview"] = torch.zeros_like(out["images"]["agentview"])

    return out
