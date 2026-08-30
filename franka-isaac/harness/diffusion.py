"""A diffusion policy, written out rather than imported.

The idea in one paragraph. A policy has to represent "what would a demonstrator
do here", and that is often not a single action: at the moment of reaching for a
cube, going left around it and going right around it are both good, and their
average is a hand driven into the cube. Regression to the mean is exactly what an
MSE-trained policy does. A diffusion policy sidesteps it by learning to *sample*
from the distribution of good actions instead of predicting its centre. It does
that by learning to undo noise: take a demonstrated action chunk, add a known
amount of Gaussian noise, and train a network to predict the noise that was
added. At inference you start from pure noise and subtract predicted noise
repeatedly until an action chunk falls out.

Three deliberate simplifications against the original paper, all of which trade
capacity for legibility:

* **An MLP denoiser, not a 1D convolutional U-Net.** The paper's U-Net earns its
  keep on image observations and long horizons. On a 30-dimensional state vector
  and an 8-step chunk, a couple of wide residual layers fit the same function
  and can be read in one sitting.
* **Predicting the noise, not the action.** These are equivalent
  parameterisations; noise prediction is the one where the loss is plain MSE
  against a known target, which makes it obvious when training is working.
* **DDPM sampling, not DDIM.** More steps at inference, less to explain. The
  sampler is fifteen lines and matches the training objective exactly.

Nothing here imports Isaac Lab, so the whole file is testable on a laptop --
including a real overfitting test that trains for a few hundred steps and checks
the policy reproduces a memorised action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DiffusionConfig:
    """Sizes and schedule. Defaults suit a state vector and a short chunk."""

    obs_dim: int
    action_dim: int
    horizon: int

    hidden_dim: int = 512
    hidden_layers: int = 3
    time_embed_dim: int = 64

    # Denoising steps. 100 is plenty for an 8x7 action chunk; the paper uses
    # this range too, and inference cost is linear in it.
    num_steps: int = 100
    # Cosine schedule: it spends less of its budget at the very noisy end than a
    # linear one, which matters when the number of steps is small.
    cosine_offset: float = 0.008


class NoiseSchedule:
    """The forward process: how much signal survives at each step.

    ``alpha_bar[t]`` is the fraction of the original signal remaining at step
    ``t``, running from approximately 1 (untouched) to approximately 0 (pure
    noise). Everything else is derived from it, which is why this is the only
    thing worth testing carefully -- a schedule that does not reach noise at the
    end, or that starts already noisy, breaks training in ways the loss curve
    does not reveal.
    """

    def __init__(self, num_steps: int, cosine_offset: float = 0.008, device: torch.device | None = None):
        if num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}")

        self.num_steps = num_steps
        steps = torch.arange(num_steps + 1, dtype=torch.float64, device=device) / num_steps
        # Nichol & Dhariwal's cosine schedule.
        alpha_bar = torch.cos((steps + cosine_offset) / (1 + cosine_offset) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0001, 0.9999)
        self.betas = betas.float()
        self.alphas = (1.0 - self.betas).float()
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        # The variance of the true reverse step, given the clean action. DDPM
        # offers a choice between this and beta_t; beta_t is simpler and
        # noticeably noisier, and on a short chunk that noise is the difference
        # between a usable action and a jittery one.
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=self.alpha_bar.dtype), self.alpha_bar[:-1]])
        self.posterior_variance = (self.betas * (1 - alpha_bar_prev) / (1 - self.alpha_bar)).float()

    def to(self, device: torch.device) -> NoiseSchedule:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The forward process in closed form: jump straight to step ``t``.

        ``x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise``

        Being able to jump is what makes training cheap: each sample picks one
        random ``t`` rather than simulating the whole chain.
        """
        alpha_bar = self.alpha_bar[t].view(-1, *([1] * (clean.dim() - 1)))
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Embed the denoising step, the way transformers embed position.

    The network has to behave differently at "almost pure noise" and "nearly
    clean", so the step index is an input. Feeding the raw integer would make the
    network learn its own scaling; a sinusoidal embedding hands it a range of
    frequencies to pick from and trains noticeably faster.
    """
    half = dim // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    if dim % 2:  # odd sizes need one column of padding
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class ResidualBlock(nn.Module):
    """Linear, activation, linear, plus a skip.

    The skip is not decoration: the denoiser's job at small ``t`` is nearly the
    identity, and a residual path lets it represent that without having to learn
    it in the weights.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DiffusionPolicy(nn.Module):
    """Predicts the noise that was added to an action chunk.

    Conditioning is by concatenation: the observation and the step embedding are
    joined to the flattened noisy chunk and pushed through an MLP. FiLM
    conditioning would be the next thing to try if capacity ever became the
    limit; concatenation is enough here and is one line instead of a module.
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg

        chunk_dim = cfg.horizon * cfg.action_dim
        input_dim = chunk_dim + cfg.obs_dim + cfg.time_embed_dim

        self.input_proj = nn.Linear(input_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(ResidualBlock(cfg.hidden_dim) for _ in range(cfg.hidden_layers))
        self.output_proj = nn.Linear(cfg.hidden_dim, chunk_dim)

    def forward(self, noisy_actions: torch.Tensor, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise in ``noisy_actions``.

        Args:
            noisy_actions: ``(batch, horizon, action_dim)``.
            obs: ``(batch, obs_dim)``, already normalised.
            t: ``(batch,)`` denoising step indices.

        Returns:
            Predicted noise, shaped like ``noisy_actions``.
        """
        batch = noisy_actions.shape[0]
        flat = noisy_actions.reshape(batch, -1)
        time_embedding = sinusoidal_embedding(t, self.cfg.time_embed_dim)

        hidden = self.input_proj(torch.cat([flat, obs, time_embedding], dim=1))
        for block in self.blocks:
            hidden = block(hidden)

        return self.output_proj(hidden).reshape_as(noisy_actions)


def diffusion_loss(
    policy: DiffusionPolicy,
    schedule: NoiseSchedule,
    obs: torch.Tensor,
    actions: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The whole training objective: predict the noise you added.

    One random denoising step per sample, rather than all of them, which is what
    makes each gradient step cheap.
    """
    batch = actions.shape[0]
    t = torch.randint(0, schedule.num_steps, (batch,), device=actions.device, generator=generator)
    noise = torch.randn(actions.shape, device=actions.device, generator=generator)

    noisy = schedule.add_noise(actions, noise, t)
    predicted = policy(noisy, obs, t)

    return torch.nn.functional.mse_loss(predicted, noise)


@torch.no_grad()
def sample_actions(
    policy: DiffusionPolicy,
    schedule: NoiseSchedule,
    obs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Denoise from pure noise to an action chunk.

    The reverse process, one step at a time: estimate the noise, remove the part
    of it this step is responsible for, then add a little back -- except on the
    final step, where adding noise would just dirty the answer.

    Returns:
        ``(batch, horizon, action_dim)``, in whatever space the policy was
        trained on. Callers holding a normaliser must undo it.
    """
    was_training = policy.training
    policy.eval()

    cfg = policy.cfg
    batch = obs.shape[0]
    shape = (batch, cfg.horizon, cfg.action_dim)
    actions = torch.randn(shape, device=obs.device, generator=generator)

    for step in reversed(range(schedule.num_steps)):
        t = torch.full((batch,), step, device=obs.device, dtype=torch.long)
        predicted_noise = policy(actions, obs, t)

        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bar[step]
        # The posterior mean of x_{t-1} given x_t and the predicted noise.
        mean = (actions - (1 - alpha) / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()

        if step > 0:
            noise = torch.randn(shape, device=obs.device, generator=generator)
            actions = mean + schedule.posterior_variance[step].sqrt() * noise
        else:
            # No noise on the last step: adding any would only dirty the answer.
            actions = mean

    policy.train(was_training)
    return actions
