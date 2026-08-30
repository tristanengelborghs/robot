"""The diffusion policy, including whether it can actually learn anything.

Most of these are shape and schedule checks, which catch the errors that would
otherwise show up as a loss curve that goes down while the policy does nothing
useful. The one that earns its runtime is the last: it trains a small policy to
memorise a fixed action and checks it reproduces it. That is a slow test by unit
standards, a few seconds, and it is the only one that would notice if the
denoiser and the sampler disagreed about what the network predicts -- a mistake
that leaves every other test passing.
"""

import math

import pytest
import torch

from harness.diffusion import (
    DiffusionConfig,
    DiffusionPolicy,
    NoiseSchedule,
    diffusion_loss,
    sample_actions,
    sinusoidal_embedding,
)

CONFIG = DiffusionConfig(obs_dim=6, action_dim=3, horizon=4, hidden_dim=64, hidden_layers=2, num_steps=20)


def test_the_schedule_runs_from_clean_to_noise():
    schedule = NoiseSchedule(num_steps=100)

    # alpha_bar is the fraction of signal left: nearly all at the start, nearly
    # none at the end. A schedule that fails either end breaks training in a way
    # the loss curve does not show.
    assert schedule.alpha_bar[0] > 0.99
    assert schedule.alpha_bar[-1] < 0.02


def test_the_schedule_decreases_monotonically():
    schedule = NoiseSchedule(num_steps=100)
    differences = schedule.alpha_bar[1:] - schedule.alpha_bar[:-1]
    assert (differences <= 0).all(), "signal must never come back as noise is added"


def test_betas_stay_in_range():
    schedule = NoiseSchedule(num_steps=100)
    assert (schedule.betas > 0).all()
    assert (schedule.betas < 1).all()


def test_a_schedule_of_no_steps_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        NoiseSchedule(num_steps=0)


def test_adding_noise_at_the_first_step_barely_changes_the_action():
    schedule = NoiseSchedule(num_steps=100)
    clean = torch.ones(4, 3, 2)
    noise = torch.randn(4, 3, 2)

    barely = schedule.add_noise(clean, noise, torch.zeros(4, dtype=torch.long))
    assert torch.allclose(barely, clean, atol=0.15)


def test_adding_noise_at_the_last_step_destroys_the_action():
    schedule = NoiseSchedule(num_steps=100)
    clean = torch.ones(64, 3, 2)
    noise = torch.randn(64, 3, 2)

    destroyed = schedule.add_noise(clean, noise, torch.full((64,), 99, dtype=torch.long))
    # What is left should look like the noise, not like the ones.
    assert (destroyed - noise).abs().mean() < (destroyed - clean).abs().mean()


def test_noise_is_added_per_sample_at_its_own_step():
    # Every sample in a batch gets its own t; broadcasting the wrong way would
    # silently apply one sample's noise level to the whole batch.
    schedule = NoiseSchedule(num_steps=100)
    clean = torch.ones(2, 3, 2)
    noise = torch.zeros(2, 3, 2)
    t = torch.tensor([0, 99])

    noised = schedule.add_noise(clean, noise, t)
    assert noised[0].mean() > noised[1].mean(), "the later step must retain less signal"


def test_the_time_embedding_distinguishes_steps():
    embedding = sinusoidal_embedding(torch.arange(10), dim=16)
    assert embedding.shape == (10, 16)
    assert not torch.allclose(embedding[0], embedding[1])
    assert torch.isfinite(embedding).all()


def test_an_odd_embedding_size_still_works():
    assert sinusoidal_embedding(torch.arange(3), dim=7).shape == (3, 7)


def test_the_policy_predicts_noise_shaped_like_its_input():
    policy = DiffusionPolicy(CONFIG)
    noisy = torch.randn(5, CONFIG.horizon, CONFIG.action_dim)
    obs = torch.randn(5, CONFIG.obs_dim)
    t = torch.randint(0, CONFIG.num_steps, (5,))

    assert policy(noisy, obs, t).shape == noisy.shape


def test_the_policy_output_depends_on_the_observation():
    # A policy ignoring its conditioning would train to a plausible loss and be
    # useless, so check the observation actually reaches the output.
    torch.manual_seed(0)
    policy = DiffusionPolicy(CONFIG)
    noisy = torch.randn(1, CONFIG.horizon, CONFIG.action_dim)
    t = torch.zeros(1, dtype=torch.long)

    first = policy(noisy, torch.zeros(1, CONFIG.obs_dim), t)
    second = policy(noisy, torch.ones(1, CONFIG.obs_dim), t)
    assert not torch.allclose(first, second)


def test_the_policy_output_depends_on_the_denoising_step():
    torch.manual_seed(0)
    policy = DiffusionPolicy(CONFIG)
    noisy = torch.randn(1, CONFIG.horizon, CONFIG.action_dim)
    obs = torch.randn(1, CONFIG.obs_dim)

    early = policy(noisy, obs, torch.zeros(1, dtype=torch.long))
    late = policy(noisy, obs, torch.full((1,), CONFIG.num_steps - 1, dtype=torch.long))
    assert not torch.allclose(early, late)


def test_the_loss_is_a_finite_scalar():
    policy = DiffusionPolicy(CONFIG)
    schedule = NoiseSchedule(CONFIG.num_steps)
    loss = diffusion_loss(policy, schedule, torch.randn(8, CONFIG.obs_dim), torch.randn(8, 4, 3))

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_sampling_returns_a_chunk_of_the_right_shape():
    policy = DiffusionPolicy(CONFIG)
    schedule = NoiseSchedule(CONFIG.num_steps)
    actions = sample_actions(policy, schedule, torch.randn(3, CONFIG.obs_dim))

    assert actions.shape == (3, CONFIG.horizon, CONFIG.action_dim)
    assert torch.isfinite(actions).all()


def test_sampling_leaves_the_policy_in_the_mode_it_found_it():
    # Sampling switches to eval internally; forgetting to switch back would
    # silently disable dropout and norm updates for the rest of training.
    policy = DiffusionPolicy(CONFIG)
    policy.train()
    sample_actions(policy, NoiseSchedule(CONFIG.num_steps), torch.randn(1, CONFIG.obs_dim))
    assert policy.training


def test_it_can_memorise_one_action():
    """Train on a single fixed pair and check the policy reproduces it.

    This is the test that would catch the denoiser and the sampler disagreeing
    about what the network predicts -- noise against action, or a sign error in
    the posterior mean. Every shape test passes under those bugs; this one does
    not.
    """
    torch.manual_seed(0)
    cfg = DiffusionConfig(obs_dim=4, action_dim=2, horizon=2, hidden_dim=128, hidden_layers=2, num_steps=50)
    policy = DiffusionPolicy(cfg)
    schedule = NoiseSchedule(cfg.num_steps)

    obs = torch.ones(1, cfg.obs_dim)
    target = torch.tensor([[[0.7, -0.4], [0.2, 0.9]]])

    optimiser = torch.optim.Adam(policy.parameters(), lr=2e-3)
    batch_obs = obs.repeat(64, 1)
    batch_target = target.repeat(64, 1, 1)

    first_loss = None
    for step in range(2500):
        loss = diffusion_loss(policy, schedule, batch_obs, batch_target)
        if step == 0:
            first_loss = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert loss.item() < first_loss, "the loss must come down"

    # Average several samples: DDPM is stochastic, and the claim is that the
    # distribution it learned is centred on the demonstrated action.
    drawn = sample_actions(policy, schedule, obs.repeat(32, 1)).mean(dim=0)
    error = (drawn - target[0]).abs().max().item()
    assert error < 0.2, f"sampled {drawn.tolist()} against {target[0].tolist()}"


def test_the_posterior_variance_is_smaller_than_beta():
    # The reason for preferring it: the same reverse step with less injected
    # noise, which on a short action chunk is the difference between a usable
    # action and a jittery one.
    schedule = NoiseSchedule(num_steps=100)
    assert (schedule.posterior_variance[1:] <= schedule.betas[1:] + 1e-8).all()
    assert schedule.posterior_variance[0] >= 0.0


def test_the_schedule_moves_to_a_device():
    schedule = NoiseSchedule(10).to(torch.device("cpu"))
    assert schedule.alpha_bar.device.type == "cpu"
    assert schedule.posterior_variance.device.type == "cpu"


def test_cosine_schedule_is_gentler_than_linear_at_the_start():
    # The point of the cosine schedule: it does not spend its early steps
    # destroying the signal, which matters when there are few steps.
    schedule = NoiseSchedule(num_steps=100)
    quarter = schedule.alpha_bar[25].item()
    assert quarter > 0.5, f"a quarter of the way in, most signal should remain, got {quarter:.2f}"
    assert not math.isnan(quarter)
