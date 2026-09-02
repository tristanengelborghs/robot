"""Copy this file to start a new architecture.

    cp src/robobench/models/template.py src/robobench/models/my_arch.py

Then:
  1. rename the class and change @register_model("my_arch")
  2. add the import to robobench/models/__init__.py
  3. train it:  python -m robobench.train --config configs/resnet_film_bc.yaml model.name=my_arch

You only have to implement `forward`. The masked-L1 objective, action chunking,
normalization, rollout evaluation and the significance testing all come from the
harness, so anything that changes in the results table changes because of your
architecture and not because of the surrounding code.

If your method needs a different objective — diffusion, flow matching, a VAE term —
override `loss()` and `predict()` as shown at the bottom.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from robobench.models.base import BasePolicy, ObsSpec
from robobench.models.modules import MLP, ImageNormalize, ResNet18Encoder, SpatialSoftmax
from robobench.registry import register_model


@register_model("template")
class TemplatePolicy(BasePolicy):
    """A minimal working policy: encode, concatenate, decode a chunk with an MLP.

    It trains and it runs, but it is weaker than `resnet_film_bc` — it has no
    language conditioning inside the vision tower and no attention. Treat it as a
    scaffold to gut, not as a baseline.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        action_dim: int,
        chunk_size: int,
        # --- your hyperparameters go here; they come straight from the YAML under
        #     `model:` (everything except `name`), so no plumbing is required.
        hidden_dim: int = 256,
        num_keypoints: int = 32,
    ):
        super().__init__(obs_spec, action_dim, chunk_size)
        self.cameras = list(obs_spec.cameras)

        self.image_norm = ImageNormalize()
        self.vision = nn.ModuleList([ResNet18Encoder(width=32) for _ in self.cameras])
        self.pool = nn.ModuleList(
            [SpatialSoftmax(v.out_channels, num_keypoints) for v in self.vision]
        )

        feat_dim = len(self.cameras) * num_keypoints * 2 + obs_spec.proprio_dim + obs_spec.lang_dim
        self.head = MLP([feat_dim, hidden_dim, hidden_dim, chunk_size * action_dim])

    def forward(
        self,
        images: Dict[str, torch.Tensor],   # camera -> uint8 (B, 3, H, W)
        proprio: torch.Tensor,             # (B, proprio_dim)
        lang: torch.Tensor,                # (B, lang_dim) frozen embedding
        text=None,                         # list of B instruction strings; unused here
    ) -> torch.Tensor:                     # -> (B, chunk_size, action_dim) in [-1, 1]
        feats = []
        for i, cam in enumerate(self.cameras):
            feats.append(self.pool[i](self.vision[i](self.image_norm(images[cam]))))
        feats += [proprio, lang]

        out = self.head(torch.cat(feats, dim=-1))
        return out.reshape(out.shape[0], self.chunk_size, self.action_dim)

    # ------------------------------------------------------------------
    # Only needed for non-regression objectives. Delete if L1 is fine.
    #
    # def loss(self, batch):
    #     """Must return a dict containing key "loss". Extra keys are logged."""
    #     noise = torch.randn_like(batch["actions"])
    #     ...
    #     return {"loss": loss, "denoise_mse": mse.detach()}
    #
    # @torch.no_grad()
    # def predict(self, images, proprio, lang, text=None):
    #     """Called once per chunk at rollout time. Run your sampler here."""
    #     ...
    #     return actions  # (B, chunk_size, action_dim)
