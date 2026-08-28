"""Reusable blocks. Nothing here is novel — it is the standard visuomotor stack, kept
in one place so a new architecture can borrow the parts it does not want to rethink."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNormalize(nn.Module):
    """uint8 (B,3,H,W) -> normalized float. Kept inside the model so the evaluator
    can hand raw frames straight from the simulator with no preprocessing drift."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return (x - self.mean) / self.std


class FiLM(nn.Module):
    """Feature-wise linear modulation: condition a conv stage on the instruction."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)  # starts as identity

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class SpatialSoftmax(nn.Module):
    """Feature map -> (x, y) expected keypoint locations.

    Standard in visuomotor BC (robomimic, Diffusion Policy). Keeps spatial precision
    that global average pooling throws away, which matters a lot for grasping.
    """

    def __init__(self, in_channels: int, num_keypoints: int = 64, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.num_keypoints = num_keypoints
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.proj(x)
        _, k, h, w = x.shape
        flat = (x.reshape(b * k, h * w) / self.temperature).softmax(dim=-1)

        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        exp_x = (flat * pos_x.reshape(1, -1)).sum(dim=-1)
        exp_y = (flat * pos_y.reshape(1, -1)).sum(dim=-1)
        return torch.stack([exp_x, exp_y], dim=-1).reshape(b, k * 2)


def _gn(channels: int, max_groups: int = 16) -> nn.GroupNorm:
    """GroupNorm, not BatchNorm.

    BC batches are small and highly correlated within a batch (consecutive timesteps
    from the same demo), which makes BatchNorm statistics unstable and creates a
    train/eval mismatch at rollout time where batch size is 1.
    """
    groups = math.gcd(channels, max_groups)
    return nn.GroupNorm(max(1, groups), channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.n1 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.n2 = _gn(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), _gn(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        out = F.relu(self.n1(self.conv1(x)), inplace=True)
        out = self.n2(self.conv2(out))
        return F.relu(out + idt, inplace=True)


class ResNet18Encoder(nn.Module):
    """ResNet-18 (GroupNorm) with optional per-stage FiLM conditioning.

    Trained from scratch by default. Keeping ImageNet weights out of the picture is
    what makes a "no pretraining" column honest; flip `pretrained` only if you mean it.
    """

    STAGES = ((64, 1), (128, 2), (256, 2), (512, 2))

    def __init__(self, cond_dim: Optional[int] = None, width: int = 64, in_ch: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 7, 2, 3, bias=False),
            _gn(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        chans = width
        stages, films = [], []
        for out_mult, stride in self.STAGES:
            out_ch = int(out_mult * width / 64)
            stages.append(nn.Sequential(BasicBlock(chans, out_ch, stride), BasicBlock(out_ch, out_ch, 1)))
            films.append(FiLM(cond_dim, out_ch) if cond_dim else None)
            chans = out_ch
        self.stages = nn.ModuleList(stages)
        self.films = nn.ModuleList([f if f is not None else nn.Identity() for f in films])
        self.use_film = cond_dim is not None
        self.out_channels = chans

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.stem(x)
        for stage, film in zip(self.stages, self.films):
            x = stage(x)
            if self.use_film and cond is not None:
                x = film(x, cond)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[-1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class MLP(nn.Module):
    def __init__(self, sizes, act=nn.GELU, out_act=False):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 or out_act:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
