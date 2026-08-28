"""Policy and value networks for the teacher-student pipeline.

The actor-critic is asymmetric on purpose: the critic consumes privileged
simulator state (object pose, contact forces, randomized physics params) that
the actor never sees. The critic exists only to reduce gradient variance during
training and is thrown away at deployment, so feeding it privileged state is
free. The actor must be restricted to observations available on the real
robot — any privileged input that leaks into it works perfectly in simulation
and yields a policy that cannot run on hardware, a failure that shows up months
later, not in training curves. Keeping the two input streams as separate
constructor arguments makes such a leak visible at the call site instead. The
same seam is what later lets a student policy be distilled from the actor
alone (DAgger on actor observations) without touching or retraining the critic.

log_std is a state-independent learnable parameter: exploration noise is a
training-schedule quantity, and making it a function of the observation lets
the policy "explain away" bad actions by claiming high variance in hard states.
It is clamped to [-5, 2] because a std below exp(-5) makes log-probs of nearby
actions explode (KL spikes, NaN updates) and above exp(2) the policy is
indistinguishable from noise on a normalized action space.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["ActorCritic", "StudentPolicy"]

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _mlp(sizes: Sequence[int], activation, out_gain: float) -> nn.Sequential:
    """Orthogonal-initialized MLP: gain sqrt(2) on hidden layers, `out_gain` on the head.

    The head gain is the part that matters: 0.01 on a policy head keeps the
    initial action distribution near zero-mean so early exploration is set by
    log_std alone, and 1.0 on a value head keeps initial value estimates at the
    scale of the returns they will regress to.
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        last = i == len(sizes) - 2
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(lin.weight, gain=out_gain if last else math.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if not last:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Asymmetric actor-critic with a diagonal Gaussian policy."""

    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation=nn.ELU,
    ) -> None:
        super().__init__()
        self.actor = _mlp([actor_obs_dim, *hidden, act_dim], activation, out_gain=0.01)
        self.critic = _mlp([critic_obs_dim, *hidden, 1], activation, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def _dist(self, actor_obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(actor_obs)
        # Clamp at use rather than in-place after the optimizer step: gradients
        # flow while inside the range and are cut exactly when saturated.
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or take the mean) action for rollout collection. No grad."""
        dist = self._dist(actor_obs)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.critic(critic_obs).squeeze(-1)
        return action, logp, value

    def evaluate(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-prob, entropy, and value of stored transitions, with grad."""
        dist = self._dist(actor_obs)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(critic_obs).squeeze(-1)
        return logp, entropy, value


class StudentPolicy(nn.Module):
    """Deterministic MLP obs -> action, the DAgger distillation target.

    Deliberately has no critic and no noise: the student regresses the
    teacher's mean action from deployable observations only, so its interface
    is exactly what runs on the robot.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation=nn.ELU,
    ) -> None:
        super().__init__()
        self.net = _mlp([obs_dim, *hidden, act_dim], activation, out_gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
