"""The contract every architecture implements.

Implement `forward` and you get the training loop, the evaluator, action chunking,
normalization and the statistics for free. Override `loss` / `predict` only if your
method needs a different objective (diffusion, flow matching, a VAE term, ...).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class ObsSpec:
    """Everything a model needs to know about its inputs, resolved from the dataset."""

    cameras: List[str] = field(default_factory=lambda: ["agentview", "wrist"])
    image_size: int = 128
    proprio_dim: int = 9
    lang_dim: int = 512


class BasePolicy(nn.Module, abc.ABC):
    registry_name: str = "base"

    # Whether the evaluator loads a checkpoint with `strict=True`. A policy that
    # stores only what training changed (see `trainable_state_dict`) sets this
    # False; the evaluator then checks that every key the checkpoint lacks is one
    # the freshly built model already carries frozen.
    strict_load: bool = True

    def __init__(self, obs_spec: ObsSpec, action_dim: int, chunk_size: int):
        super().__init__()
        self.obs_spec = obs_spec
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    # -- required ---------------------------------------------------------

    @abc.abstractmethod
    def forward(
        self,
        images: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang: torch.Tensor,
        text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        images   dict[camera -> uint8 tensor (B, 3, H, W)]
        proprio  float32 (B, proprio_dim)
        lang     float32 (B, lang_dim)   frozen instruction embedding
        text     the B instruction strings, or None. A model with its own text
                 encoder reads these and ignores `lang`; the `no_lang` ablation
                 blanks them to "" alongside zeroing `lang`.

        returns  float32 (B, chunk_size, action_dim), actions in normalized [-1, 1]
        """

    # -- provided ---------------------------------------------------------

    def loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Masked L1 over the action chunk. Padded chunk steps contribute nothing."""
        pred = self(batch["images"], batch["proprio"], batch["lang"], text=batch.get("text"))
        per_step = (pred - batch["actions"]).abs().mean(dim=-1)  # (B, chunk)
        mask = batch["mask"]
        denom = mask.sum().clamp(min=1.0)
        loss = (per_step * mask).sum() / denom
        with torch.no_grad():
            gripper_err = (pred[..., -1] - batch["actions"][..., -1]).abs()
            gripper_err = (gripper_err * mask).sum() / denom
        return {"loss": loss, "l1": loss.detach(), "gripper_l1": gripper_err}

    @torch.no_grad()
    def predict(
        self,
        images: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang: torch.Tensor,
        text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Inference entry point used by the evaluator. Returns (B, chunk, action_dim)."""
        return self(images, proprio, lang, text=text)

    # -- checkpointing ----------------------------------------------------

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """What a checkpoint stores, and what the EMA tracks. Everything, by default.

        A policy built on a frozen pretrained backbone overrides this to return only
        what training changed, and sets `strict_load = False` so the evaluator
        rebuilds the backbone from its source and loads the delta on top. Saving a
        500M-parameter backbone twice per checkpoint, and averaging it every step,
        is otherwise the price of a two-line model.
        """
        return self.state_dict()
