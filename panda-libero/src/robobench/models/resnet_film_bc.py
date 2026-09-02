"""ResNetFiLM-BC — the reference architecture and the baseline to beat.

Roughly 27M parameters, trained from scratch, no pretraining of any kind. It is
deliberately conventional: per-camera ResNet-18 with FiLM language conditioning,
spatial-softmax pooling, a small transformer that reads the observation tokens and
writes an action chunk. Nothing here is a research contribution — that is the point.
It exists so a new architecture has an honest, same-budget reference to beat.

The published from-scratch reference on LIBERO (Diffusion Policy, ~150M params) sits
around 78% on SPATIAL and 93% on OBJECT. That is the column to compete in; the ~98%
numbers belong to 3-7B models pretrained on Open-X and are not the same experiment.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from robobench.models.base import BasePolicy, ObsSpec
from robobench.models.modules import (
    MLP,
    ImageNormalize,
    ResNet18Encoder,
    SpatialSoftmax,
)
from robobench.registry import register_model


@register_model("resnet_film_bc")
class ResNetFiLMBC(BasePolicy):
    def __init__(
        self,
        obs_spec: ObsSpec,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
        num_keypoints: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        vision_width: int = 64,
        share_vision_encoder: bool = False,
        use_film: bool = True,
    ):
        super().__init__(obs_spec, action_dim, chunk_size)
        self.hidden_dim = hidden_dim
        self.cameras = list(obs_spec.cameras)
        self.share_vision_encoder = share_vision_encoder

        self.image_norm = ImageNormalize()

        # Language: a frozen sentence embedding projected into model width. Frozen
        # because re-training a text encoder on 10 instructions only memorizes them.
        self.lang_proj = MLP([obs_spec.lang_dim, hidden_dim, hidden_dim])
        cond_dim = hidden_dim if use_film else None

        n_encoders = 1 if share_vision_encoder else len(self.cameras)
        self.vision = nn.ModuleList(
            [ResNet18Encoder(cond_dim=cond_dim, width=vision_width) for _ in range(n_encoders)]
        )
        feat_ch = self.vision[0].out_channels
        self.pool = nn.ModuleList(
            [SpatialSoftmax(feat_ch, num_keypoints) for _ in range(n_encoders)]
        )
        self.vis_proj = nn.ModuleList(
            [MLP([num_keypoints * 2, hidden_dim, hidden_dim]) for _ in range(len(self.cameras))]
        )

        self.proprio_proj = MLP([obs_spec.proprio_dim, hidden_dim, hidden_dim])

        # One learned type embedding per token slot (per-camera, proprio, language),
        # so the transformer can tell the streams apart.
        self.num_obs_tokens = len(self.cameras) + 2
        self.token_type = nn.Parameter(torch.zeros(1, self.num_obs_tokens, hidden_dim))
        nn.init.normal_(self.token_type, std=0.02)

        # Learned queries, one per chunk step — the action decoder reads these out.
        self.chunk_queries = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        nn.init.normal_(self.chunk_queries, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and only warns; off explicitly.
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        nn.init.zeros_(self.action_head.bias)
        nn.init.normal_(self.action_head.weight, std=0.01)

    def _encode_camera(self, idx: int, img: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        enc_idx = 0 if self.share_vision_encoder else idx
        feat = self.vision[enc_idx](self.image_norm(img), cond)
        return self.vis_proj[idx](self.pool[enc_idx](feat))

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang: torch.Tensor,
        text: Optional[List[str]] = None,  # language arrives as the frozen embedding
    ) -> torch.Tensor:
        lang_tok = self.lang_proj(lang)

        tokens = [self._encode_camera(i, images[cam], lang_tok) for i, cam in enumerate(self.cameras)]
        tokens.append(self.proprio_proj(proprio))
        tokens.append(lang_tok)

        obs = torch.stack(tokens, dim=1) + self.token_type  # (B, n_obs, D)
        queries = self.chunk_queries.expand(obs.shape[0], -1, -1)

        seq = torch.cat([obs, queries], dim=1)
        seq = self.transformer(seq)

        decoded = self.out_norm(seq[:, self.num_obs_tokens :])
        return self.action_head(decoded)
