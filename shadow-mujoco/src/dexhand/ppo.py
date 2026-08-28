"""PPO with GAE for training the teacher, plus running observation normalization.

Everything here is textbook PPO; the module exists because the three details
that actually decide whether the trained policy works are invisible in a code
review of the caller:

(1) GAE must cut the bootstrap at episode boundaries:
    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t),
    with the recursive trace masked by the same (1 - done_t). Forgetting the
    mask blends value estimates across resets. Training still runs, curves
    still go up, and the policy quietly converges to something worse — the
    classic silent PPO bug, which is why the tests pin GAE to hand-computed
    values instead of comparing against another implementation.

(2) Normalization statistics are part of the policy. If the mean/var used at
    evaluation or deployment differ from the ones the policy was trained
    under, every observation is shifted through a distribution the network
    never saw. Nothing errors; the policy is just bad. Hence RunningNorm
    round-trips exactly through state_dict().

(3) Reproducibility: minibatch shuffling is seeded from torch's RNG, so a run
    with a fixed torch seed is bit-identical — without this, "did my change
    alter behaviour?" cannot be answered by diffing losses.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

__all__ = ["RunningNorm", "RolloutBuffer", "PPO"]


class RunningNorm:
    """Running mean/var over numpy batches via parallel-variance merging.

    Uses the Chan et al. pairwise merge so the accumulated statistics equal
    the exact population mean/var of everything ever passed to update() —
    no decay, no bias from a warm-start count. Exactness matters because the
    tests compare against numpy on the concatenated stream, which is the only
    way to notice a wrong merge formula (a wrong one still produces plausible
    numbers).
    """

    def __init__(self, shape, eps: float = 1e-8) -> None:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.eps = float(eps)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0.0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        if self.count == 0.0:
            self.mean, self.var, self.count = batch_mean, batch_var, float(batch_count)
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        m2 = self.var * self.count + batch_var * batch_count
        m2 += np.square(delta) * self.count * batch_count / total
        self.mean = self.mean + delta * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = (x - self.mean) / np.sqrt(self.var + self.eps)
        # +-10 sigma clip: a single corrupted observation (NaN-adjacent physics
        # step, un-reset sensor) must not blow up the policy input.
        return np.clip(out, -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])


class RolloutBuffer:
    """Fixed-horizon on-policy storage for `n_envs` parallel environments.

    Convention: everything passed to add() at step t belongs to the same
    transition — done_t is the flag *returned by* env.step at t, meaning the
    episode ended on that transition and obs_{t+1} is a reset. The value stored
    at t+1 after a done is therefore V(reset state); the (1 - done_t) mask is
    what keeps it out of the advantage at t.
    """

    def __init__(self, horizon: int, n_envs: int, actor_obs_dim: int, critic_obs_dim: int, act_dim: int) -> None:
        self.horizon, self.n_envs = horizon, n_envs
        self.actor_obs = np.zeros((horizon, n_envs, actor_obs_dim), dtype=np.float32)
        self.critic_obs = np.zeros((horizon, n_envs, critic_obs_dim), dtype=np.float32)
        self.actions = np.zeros((horizon, n_envs, act_dim), dtype=np.float32)
        self.logps = np.zeros((horizon, n_envs), dtype=np.float32)
        self.rewards = np.zeros((horizon, n_envs), dtype=np.float32)
        self.dones = np.zeros((horizon, n_envs), dtype=np.float32)
        self.values = np.zeros((horizon, n_envs), dtype=np.float32)
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0
        self.advantages = None
        self.returns = None

    def add(self, actor_obs, critic_obs, action, logp, reward, done, value) -> None:
        if self.ptr >= self.horizon:
            raise RuntimeError("RolloutBuffer is full; call reset() before reuse")
        t = self.ptr
        self.actor_obs[t] = np.asarray(actor_obs, dtype=np.float32).reshape(self.n_envs, -1)
        self.critic_obs[t] = np.asarray(critic_obs, dtype=np.float32).reshape(self.n_envs, -1)
        self.actions[t] = np.asarray(action, dtype=np.float32).reshape(self.n_envs, -1)
        self.logps[t] = np.asarray(logp, dtype=np.float32).reshape(self.n_envs)
        self.rewards[t] = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        self.dones[t] = np.asarray(done, dtype=np.float32).reshape(self.n_envs)
        self.values[t] = np.asarray(value, dtype=np.float32).reshape(self.n_envs)
        self.ptr += 1

    def compute_gae(self, last_value, last_done, gamma: float = 0.99, lam: float = 0.95) -> None:
        """GAE(gamma, lam) with the bootstrap cut at dones.

        `last_value` is V(obs after the final stored step); `last_done` is the
        done flag from that final step (i.e. it must equal dones[-1] — it is an
        explicit argument so the bootstrap contract is visible at the call
        site, mirroring last_value). Both are per-env arrays.
        """
        if self.ptr != self.horizon:
            raise RuntimeError(f"buffer has {self.ptr}/{self.horizon} steps; fill it before compute_gae")
        last_value = np.asarray(last_value, dtype=np.float32).reshape(self.n_envs)
        last_done = np.asarray(last_done, dtype=np.float32).reshape(self.n_envs)

        adv = np.zeros((self.horizon, self.n_envs), dtype=np.float32)
        trace = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_value, nonterminal = last_value, 1.0 - last_done
            else:
                next_value, nonterminal = self.values[t + 1], 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            trace = delta + gamma * lam * nonterminal * trace
            adv[t] = trace
        self.advantages = adv
        self.returns = adv + self.values

    def minibatches(self, n: int, rng: np.random.Generator) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield `n` shuffled flat minibatches as dicts of torch tensors."""
        if self.advantages is None or self.returns is None:
            raise RuntimeError("call compute_gae before minibatches")
        total = self.horizon * self.n_envs
        flat = {
            "actor_obs": self.actor_obs.reshape(total, -1),
            "critic_obs": self.critic_obs.reshape(total, -1),
            "actions": self.actions.reshape(total, -1),
            "logps": self.logps.reshape(total),
            "values": self.values.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
        }
        for chunk in np.array_split(rng.permutation(total), n):
            yield {k: torch.as_tensor(v[chunk]) for k, v in flat.items()}


class PPO:
    """Clipped-surrogate PPO update over a filled RolloutBuffer."""

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        clip: float = 0.2,
        epochs: int = 4,
        n_minibatches: int = 4,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        target_kl: Optional[float] = None,
    ) -> None:
        self.policy = policy
        self.clip = clip
        self.epochs = epochs
        self.n_minibatches = n_minibatches
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        # Seed the shuffle from torch's RNG stream: a torch.manual_seed'd run
        # then determines every source of randomness in the update.
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        metrics: Dict[str, list] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": [],
        }

        for _ in range(self.epochs):
            epoch_kls = []
            for mb in buffer.minibatches(self.n_minibatches, rng):
                adv = mb["advantages"]
                # Per-minibatch normalization: keeps the surrogate's scale
                # independent of the reward scale, so lr and clip mean the
                # same thing across tasks.
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                logp, entropy, value = self.policy.evaluate(
                    mb["actor_obs"], mb["critic_obs"], mb["actions"]
                )
                logratio = logp - mb["logps"]
                ratio = logratio.exp()
                with torch.no_grad():
                    # k3 estimator: unbiased-ish, always non-negative, unlike
                    # the naive (-logratio).mean() which can go negative and
                    # mask a diverging update.
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip).float().mean()

                policy_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip),
                ).mean()

                # Value clipped against the values that generated the rollout:
                # an unconstrained value step shifts the advantage baseline of
                # the very next update.
                v_clipped = mb["values"] + torch.clamp(value - mb["values"], -self.clip, self.clip)
                value_loss = 0.5 * torch.max(
                    (value - mb["returns"]).square(),
                    (v_clipped - mb["returns"]).square(),
                ).mean()

                ent = entropy.mean()
                loss = policy_loss - self.ent_coef * ent + self.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(ent.item())
                metrics["approx_kl"].append(approx_kl.item())
                metrics["clip_frac"].append(clip_frac.item())
                epoch_kls.append(approx_kl.item())

            if self.target_kl is not None and float(np.mean(epoch_kls)) > self.target_kl:
                break

        return {k: float(np.mean(v)) for k, v in metrics.items()}
