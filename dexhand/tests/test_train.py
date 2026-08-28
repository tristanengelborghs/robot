"""Training-loop glue that the PPO unit tests cannot see."""

from __future__ import annotations

import numpy as np
import torch

from dexhand.networks import ActorCritic
from dexhand.ppo import RunningNorm
from dexhand.train import bootstrap_truncations


def _setup(critic_value: float):
    policy = ActorCritic(4, 4, 2, hidden=(8, 8))
    with torch.no_grad():  # pin the critic to a known constant
        for p in policy.parameters():
            p.zero_()
        policy.critic[-1].bias.fill_(critic_value)
    norm = RunningNorm((4,))
    norm.update(np.zeros((2, 4)))
    return policy, norm


def test_truncation_is_bootstrapped_and_termination_is_not():
    """A time-limit cut is not a failure: its last reward must carry
    gamma*V(final_obs). A drop IS a failure and must carry nothing. Storing
    the merged done flag treats both alike, and with returns O(100) the
    critic then learns that value vanishes at t=200 for no observable reason."""
    policy, norm = _setup(critic_value=50.0)
    rew = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    done = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    final = {"critic": np.zeros(4, dtype=np.float32)}
    infos = [
        {"terminated": False, "final_obs": final},   # truncated
        {"terminated": True, "final_obs": final},    # dropped
        {},                                          # still running
    ]
    out = bootstrap_truncations(rew, done, infos, policy, norm, gamma=0.9, device="cpu")
    assert out[0] == np.float32(1.0 + 0.9 * 50.0)
    assert out[1] == 1.0
    assert out[2] == 1.0
    assert rew[0] == 1.0, "input reward array was mutated"
