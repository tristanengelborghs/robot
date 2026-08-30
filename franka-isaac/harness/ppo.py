"""PPO, written out rather than imported.

The parts of PPO that are actually subtle are all arithmetic, and all of them
fail silently. An advantage estimator with the wrong bootstrapping still trains,
just worse. A ratio computed against the current policy instead of the old one
turns the clipped objective into an unclipped one, and the run looks fine until
it collapses. Normalising advantages across the wrong axis costs a few percent
that nobody attributes to anything. So the arithmetic lives here, tested against
hand-computed values, and the training script is left with nothing but a loop.

What PPO is doing, briefly. Policy gradients want to increase the probability of
actions that did better than expected. Doing that naively lets one batch move
the policy far enough that the data it was estimated from no longer describes it,
and the run diverges. PPO's answer is a trust region enforced by clipping: the
objective stops improving once the new policy's probability ratio leaves
``[1-eps, 1+eps]``, so a single update cannot move the policy much regardless of
how attractive the gradient looks.

Two design notes, because both are choices rather than facts:

* **A state-independent action standard deviation.** A learned per-state sigma
  is more expressive and much easier to get wrong -- it collapses early, the
  policy stops exploring, and the failure looks like a bad reward. A single
  learned vector is what most continuous-control PPO implementations use.
* **Advantages normalised per batch, not per minibatch.** Per minibatch is
  common and slightly wrong: it makes the update depend on how the batch was
  split.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class PPOConfig:
    """Hyper-parameters, with the defaults that usually work on manipulation."""

    obs_dim: int
    action_dim: int

    hidden_dim: int = 256
    hidden_layers: int = 2

    # Discount. Episodes here are a few hundred steps, so 0.99 keeps the horizon
    # comfortably longer than the task.
    gamma: float = 0.99
    # GAE trade-off: 0 is one-step TD (low variance, biased), 1 is Monte Carlo
    # (unbiased, high variance). 0.95 is the usual compromise.
    gae_lambda: float = 0.95

    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0

    learning_rate: float = 3e-4
    epochs_per_batch: int = 5
    minibatches: int = 4

    # Starting exploration, as log standard deviation. exp(-1) is about 0.37,
    # which on actions clipped to [-1, 1] explores without thrashing.
    init_log_std: float = -1.0


def mlp(input_dim: int, hidden_dim: int, layers: int, output_dim: int) -> nn.Sequential:
    """A plain tanh MLP. Tanh rather than ReLU: it is the usual choice for
    continuous-control PPO, where bounded activations make the value function
    better behaved early on."""
    modules: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
    for _ in range(layers - 1):
        modules += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class ActorCritic(nn.Module):
    """A Gaussian policy and a value function, with separate trunks.

    Shared trunks save parameters and make the two objectives fight over the
    same features; separate ones cost almost nothing at this size.
    """

    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg
        self.actor = mlp(cfg.obs_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.action_dim)
        self.critic = mlp(cfg.obs_dim, cfg.hidden_dim, cfg.hidden_layers, 1)
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), cfg.init_log_std))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action, and report its log-probability and the state value.

        Deliberately gradient-free: this is data collection, not part of the
        update. The log-probabilities it returns are the *old* policy's, the
        fixed reference the clipped ratio is measured against -- keeping them
        attached to a graph would both waste memory and invite the mistake of
        differentiating through the behaviour policy.

        Returns:
            The action, its summed log-probability, and the critic's estimate.
        """
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score already-taken actions under the *current* policy."""
        dist = self.distribution(obs)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), self.value(obs)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation, walked backwards through time.

    Args:
        rewards: ``(steps, envs)``.
        values: ``(steps, envs)`` critic estimates for the observed states.
        dones: ``(steps, envs)``, 1 where the episode ended *on that step*.
        last_values: ``(envs,)`` bootstrap value for the state after the last.
        gamma: Discount.
        gae_lambda: Bias/variance trade-off.

    Returns:
        Advantages and value targets, both ``(steps, envs)``.

    The ``1 - done`` factors are the part worth staring at: at an episode
    boundary neither the next value nor the running advantage may leak backwards
    across it, or the critic learns to predict the start of the next episode from
    the end of the previous one.
    """
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_values)

    for t in reversed(range(steps)):
        next_values = last_values if t == steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_values * not_done - values[t]
        running = delta + gamma * gae_lambda * not_done * running
        advantages[t] = running

    return advantages, advantages + values


def normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Centre and scale, guarding the degenerate case.

    A batch where every advantage is identical has zero standard deviation --
    rare, but it happens on a reward that has not started firing yet, and
    dividing by it turns the first useful batch into NaNs.
    """
    return (values - values.mean()) / (values.std() + eps)


def ppo_losses(
    policy: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The clipped surrogate objective, a value loss and an entropy bonus.

    ``old_log_probs`` must come from the policy that *collected* the data. Using
    freshly computed ones makes every ratio exactly 1, which silently removes the
    trust region and leaves an objective that trains and then collapses.
    """
    log_probs, entropy, values = policy.evaluate(obs, actions)

    ratio = (log_probs - old_log_probs).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages
    # Minimum, then negated: the pessimistic bound is what stops an update from
    # exploiting an advantage estimate it has already moved away from.
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = torch.nn.functional.mse_loss(values, returns)
    entropy_loss = -entropy.mean()

    total = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()
        approx_kl = (old_log_probs - log_probs).mean().item()

    return total, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.mean().item(),
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
    }


class RolloutBuffer:
    """Fixed-size storage for one batch of on-policy experience.

    On-policy is the operative word: this is filled, used for a handful of
    epochs, and thrown away. Anything kept across a policy update is off-policy
    data that PPO's ratio is not valid for.
    """

    def __init__(self, steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.steps = steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros(steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(steps, num_envs, device=device)
        self.rewards = torch.zeros(steps, num_envs, device=device)
        self.dones = torch.zeros(steps, num_envs, device=device)
        self.values = torch.zeros(steps, num_envs, device=device)
        self._cursor = 0

    def add(self, obs, action, log_prob, reward, done, value) -> None:
        if self._cursor >= self.steps:
            raise IndexError(f"the buffer holds {self.steps} steps and is already full")

        index = self._cursor
        self.obs[index] = obs
        self.actions[index] = action
        self.log_probs[index] = log_prob
        self.rewards[index] = reward
        self.dones[index] = done.float()
        self.values[index] = value
        self._cursor += 1

    @property
    def full(self) -> bool:
        return self._cursor >= self.steps

    def reset(self) -> None:
        self._cursor = 0

    def flatten(self, advantages: torch.Tensor, returns: torch.Tensor):
        """Collapse (steps, envs) into one batch for the update."""
        return (
            self.obs.reshape(-1, self.obs.shape[-1]),
            self.actions.reshape(-1, self.actions.shape[-1]),
            self.log_probs.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
        )
