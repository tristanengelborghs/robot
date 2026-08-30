"""PPO's arithmetic, against values worked out by hand.

Every bug these guard against is silent. Advantage estimation that leaks across
an episode boundary still trains. A ratio computed against the current policy
rather than the collecting one removes the trust region entirely and looks fine
until the run collapses. None of it raises, none of it shows up in a loss curve,
and all of it is checkable in milliseconds without a simulator.
"""

import dataclasses

import pytest
import torch

from harness.ppo import (
    ActorCritic,
    PPOConfig,
    RolloutBuffer,
    compute_gae,
    normalize,
    ppo_losses,
)

CONFIG = PPOConfig(obs_dim=4, action_dim=2, hidden_dim=32, hidden_layers=1)


def test_gae_of_a_perfect_critic_is_zero():
    # If the critic already predicts the discounted return exactly, there is
    # nothing to learn: every advantage should vanish.
    rewards = torch.zeros(3, 1)
    values = torch.zeros(3, 1)
    dones = torch.zeros(3, 1)

    advantages, returns = compute_gae(rewards, values, dones, torch.zeros(1), gamma=0.99, gae_lambda=0.95)

    assert advantages == pytest.approx(torch.zeros(3, 1))
    assert returns == pytest.approx(torch.zeros(3, 1))


def test_gae_on_a_single_step_is_the_td_error():
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.5]])
    dones = torch.zeros(1, 1)
    last = torch.tensor([2.0])

    advantages, returns = compute_gae(rewards, values, dones, last, gamma=0.9, gae_lambda=0.95)

    # delta = r + gamma * next_value - value = 1 + 0.9 * 2 - 0.5
    assert advantages[0, 0] == pytest.approx(2.3)
    assert returns[0, 0] == pytest.approx(2.8)


def test_lambda_zero_is_one_step_temporal_difference():
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[0.0], [0.0]])
    dones = torch.zeros(2, 1)

    advantages, _ = compute_gae(rewards, values, dones, torch.zeros(1), gamma=0.5, gae_lambda=0.0)

    # With lambda = 0 nothing propagates backwards: each step is its own delta.
    assert advantages[0, 0] == pytest.approx(1.0)
    assert advantages[1, 0] == pytest.approx(1.0)


def test_lambda_one_is_the_discounted_monte_carlo_return():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.zeros(3, 1)
    dones = torch.zeros(3, 1)

    advantages, _ = compute_gae(rewards, values, dones, torch.zeros(1), gamma=0.5, gae_lambda=1.0)

    assert advantages[2, 0] == pytest.approx(1.0)
    assert advantages[1, 0] == pytest.approx(1.5)  # 1 + 0.5
    assert advantages[0, 0] == pytest.approx(1.75)  # 1 + 0.5 + 0.25


def test_an_episode_boundary_stops_the_advantage_leaking_backwards():
    """The single most important line in the estimator.

    Without the ``1 - done`` factors the critic learns to predict the beginning
    of the next episode from the end of the previous one, which is not a
    prediction about anything.
    """
    rewards = torch.tensor([[1.0], [100.0]])
    values = torch.zeros(2, 1)
    dones = torch.tensor([[1.0], [0.0]])  # the first step ends an episode

    advantages, _ = compute_gae(rewards, values, dones, torch.zeros(1), gamma=0.99, gae_lambda=0.95)

    # Step 0 ended an episode, so the 100 that follows must not reach it.
    assert advantages[0, 0] == pytest.approx(1.0)
    assert advantages[1, 0] == pytest.approx(100.0)


def test_bootstrapping_uses_the_value_after_the_last_step():
    # Truncation is not termination: an episode cut off by the batch boundary
    # must have its remaining value estimated, not treated as zero.
    rewards = torch.zeros(1, 1)
    values = torch.zeros(1, 1)
    dones = torch.zeros(1, 1)

    advantages, _ = compute_gae(rewards, values, dones, torch.tensor([10.0]), gamma=1.0, gae_lambda=1.0)
    assert advantages[0, 0] == pytest.approx(10.0)


def test_gae_handles_several_environments_independently():
    rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.zeros(2, 2)
    dones = torch.zeros(2, 2)

    advantages, _ = compute_gae(rewards, values, dones, torch.zeros(2), gamma=0.5, gae_lambda=1.0)

    assert advantages[0, 0] == pytest.approx(1.0)
    assert advantages[0, 1] == pytest.approx(0.5)


def test_normalisation_centres_and_scales():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normalized = normalize(values)
    assert normalized.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert normalized.std().item() == pytest.approx(1.0, abs=1e-2)


def test_normalising_a_constant_batch_does_not_explode():
    # Happens on a reward that has not started firing yet; without the epsilon
    # the first useful batch would be NaNs.
    result = normalize(torch.full((8,), 3.0))
    assert torch.isfinite(result).all()


def test_an_unchanged_policy_has_a_ratio_of_one():
    torch.manual_seed(0)
    policy = ActorCritic(CONFIG)
    obs = torch.randn(16, CONFIG.obs_dim)
    actions, log_probs, _ = policy.act(obs)

    _, stats = ppo_losses(
        policy, obs, actions, log_probs, torch.randn(16), torch.randn(16), CONFIG
    )

    # Same policy that collected the data: no divergence, nothing clipped.
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-5)
    assert stats["clip_fraction"] == pytest.approx(0.0)


def test_clipping_engages_once_the_policy_has_moved():
    torch.manual_seed(0)
    policy = ActorCritic(CONFIG)
    obs = torch.randn(64, CONFIG.obs_dim)
    actions, log_probs, _ = policy.act(obs)

    # Pretend the collecting policy assigned very different probabilities.
    stale = log_probs - 2.0

    _, stats = ppo_losses(policy, obs, actions, stale, torch.ones(64), torch.zeros(64), CONFIG)
    assert stats["clip_fraction"] > 0.5, "a policy this far from the data must be clipped"


def test_the_objective_saturates_beyond_the_clip():
    """The trust region, demonstrated.

    Past the clip the objective stops responding to the ratio entirely: a policy
    a hundred times more likely to take the action scores exactly the same as one
    a thousand times more likely. That saturation is the whole mechanism -- it is
    what stops one attractive batch moving the policy arbitrarily far.

    Note what this does *not* say: a clipped step can still score better than a
    small one, because the cap is (1 + eps) times the advantage rather than the
    value of standing still. Asserting otherwise was the mistake this test was
    written with first.
    """
    torch.manual_seed(0)
    cfg = dataclasses.replace(CONFIG, value_coef=0.0, entropy_coef=0.0)
    policy = ActorCritic(cfg)
    obs = torch.randn(32, cfg.obs_dim)
    actions, log_probs, _ = policy.act(obs)
    advantages = torch.ones(32)
    returns = torch.zeros(32)

    big, _ = ppo_losses(policy, obs, actions, log_probs - 5.0, advantages, returns, cfg)
    bigger, _ = ppo_losses(policy, obs, actions, log_probs - 50.0, advantages, returns, cfg)

    assert big.item() == pytest.approx(bigger.item(), abs=1e-5), "beyond the clip the ratio stops mattering"
    # And the value it saturates at is exactly the cap.
    assert big.item() == pytest.approx(-(1 + cfg.clip_ratio) * advantages.mean().item(), abs=1e-4)


def test_a_negative_advantage_is_clipped_from_the_other_side():
    # The pessimistic bound is symmetric: an action that did badly cannot have
    # its probability driven down without limit either.
    torch.manual_seed(0)
    cfg = dataclasses.replace(CONFIG, value_coef=0.0, entropy_coef=0.0)
    policy = ActorCritic(cfg)
    obs = torch.randn(32, cfg.obs_dim)
    actions, log_probs, _ = policy.act(obs)
    advantages = -torch.ones(32)
    returns = torch.zeros(32)

    small, _ = ppo_losses(policy, obs, actions, log_probs + 5.0, advantages, returns, cfg)
    smaller, _ = ppo_losses(policy, obs, actions, log_probs + 50.0, advantages, returns, cfg)

    assert small.item() == pytest.approx(smaller.item(), abs=1e-5)
    assert small.item() == pytest.approx((1 - cfg.clip_ratio) * 1.0, abs=1e-4)


def test_the_losses_are_finite_and_differentiable():
    policy = ActorCritic(CONFIG)
    obs = torch.randn(8, CONFIG.obs_dim)
    actions, log_probs, _ = policy.act(obs)

    loss, stats = ppo_losses(policy, obs, actions, log_probs, torch.randn(8), torch.randn(8), CONFIG)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    assert stats["entropy"] > 0


def test_the_policy_reports_matching_shapes():
    policy = ActorCritic(CONFIG)
    obs = torch.randn(5, CONFIG.obs_dim)
    actions, log_probs, values = policy.act(obs)

    assert actions.shape == (5, CONFIG.action_dim)
    assert log_probs.shape == (5,)
    assert values.shape == (5,)


def test_the_buffer_fills_and_reports_when_it_is_full():
    buffer = RolloutBuffer(steps=2, num_envs=3, obs_dim=4, action_dim=2, device=torch.device("cpu"))
    assert not buffer.full

    for _ in range(2):
        buffer.add(
            torch.zeros(3, 4), torch.zeros(3, 2), torch.zeros(3),
            torch.zeros(3), torch.zeros(3, dtype=torch.bool), torch.zeros(3),
        )

    assert buffer.full
    with pytest.raises(IndexError, match="already full"):
        buffer.add(
            torch.zeros(3, 4), torch.zeros(3, 2), torch.zeros(3),
            torch.zeros(3), torch.zeros(3, dtype=torch.bool), torch.zeros(3),
        )


def test_flattening_collapses_steps_and_environments():
    buffer = RolloutBuffer(steps=3, num_envs=4, obs_dim=5, action_dim=2, device=torch.device("cpu"))
    obs, actions, log_probs, advantages, returns = buffer.flatten(
        torch.zeros(3, 4), torch.zeros(3, 4)
    )

    assert obs.shape == (12, 5)
    assert actions.shape == (12, 2)
    assert log_probs.shape == (12,)
    assert advantages.shape == (12,)
    assert returns.shape == (12,)


def test_it_can_learn_a_trivial_bandit():
    """Train on a one-step problem with a known optimum.

    The reward is highest when the action is zero, so a working PPO should pull
    the policy mean towards zero. It is a weak test of a strong claim, but it is
    the only one here that exercises the loop end to end, and it would catch a
    sign error in the objective that every other test tolerates.
    """
    torch.manual_seed(0)
    cfg = PPOConfig(obs_dim=2, action_dim=1, hidden_dim=64, hidden_layers=1, entropy_coef=0.0)
    policy = ActorCritic(cfg)
    optimiser = torch.optim.Adam(policy.parameters(), lr=3e-3)

    obs = torch.zeros(256, cfg.obs_dim)
    start = policy.actor(obs[:1]).abs().item()

    for _ in range(60):
        actions, log_probs, values = policy.act(obs)
        rewards = -(actions.squeeze(-1) ** 2)  # best at zero

        advantages = normalize(rewards - values.detach())
        for _ in range(cfg.epochs_per_batch):
            loss, _ = ppo_losses(policy, obs, actions, log_probs, advantages, rewards, cfg)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimiser.step()

    end = policy.actor(obs[:1]).abs().item()
    assert end < start, f"the policy mean should move towards zero, went {start:.3f} -> {end:.3f}"
