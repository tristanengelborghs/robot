"""Tests for the PPO infrastructure.

Every test here guards a failure mode that does not raise: GAE that leaks
value estimates across episode resets, normalization stats that drift between
training and evaluation, an update step whose gradients point somewhere
useless, and nondeterminism that makes loss diffs meaningless. All of these
leave the code running and the curves plausible — the tests are the only
place they become visible.
"""

import time

import numpy as np
import pytest
import torch

from dexhand.networks import ActorCritic
from dexhand.ppo import PPO, RolloutBuffer, RunningNorm


def _fill(buffer, actor_obs, critic_obs, actions, logps, rewards, dones, values):
    for t in range(buffer.horizon):
        buffer.add(actor_obs[t], critic_obs[t], actions[t], logps[t], rewards[t], dones[t], values[t])


class TestGAE:
    def test_matches_hand_computed_values_with_mid_sequence_done(self):
        """A wrong done convention (off-by-one, or no mask at all) still trains
        and still converges — just to a worse policy. Pinning GAE to numbers
        worked out by hand with delta = r + gamma*V(s')*(1-done) - V(s) is the
        only check that catches it."""
        horizon, n_envs = 4, 2
        gamma, lam = 0.5, 0.5  # gamma*lam = 0.25, chosen so the arithmetic is exact by hand

        rewards = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], np.float32)
        values = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0], [0.5, 1.0]], np.float32)
        dones = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], np.float32)
        last_value = np.array([0.5, 2.0], np.float32)
        last_done = dones[-1]

        buf = RolloutBuffer(horizon, n_envs, 1, 1, 1)
        zeros = np.zeros((horizon, n_envs, 1), np.float32)
        _fill(buf, zeros, zeros, zeros, np.zeros((horizon, n_envs)), rewards, dones, values)
        buf.compute_gae(last_value, last_done, gamma=gamma, lam=lam)

        # Env 0 (done at t=1):                    Env 1 (no done):
        #  t=3: d = 1 + .5*.5 - .5 = .75; A = .75      d = 4 + .5*2 - 1 = 4;    A = 4
        #  t=2: d = .75;  A = .75 + .25*.75 = .9375    d = 3 + .5 - 1 = 2.5;    A = 2.5 + .25*4 = 3.5
        #  t=1: d = 1 + 0 - .5 = .5; A = .5  <-- CUT   d = 2 + .5 - 1 = 1.5;    A = 1.5 + .25*3.5 = 2.375
        #  t=0: d = .75;  A = .75 + .25*.5 = .875      d = 1 + .5 - 1 = .5;     A = .5 + .25*2.375 = 1.09375
        expected = np.array(
            [[0.875, 1.09375], [0.5, 2.375], [0.9375, 3.5], [0.75, 4.0]], np.float32
        )
        np.testing.assert_allclose(buf.advantages, expected, rtol=1e-6)
        np.testing.assert_allclose(buf.returns, expected + values, rtol=1e-6)

    def test_nothing_after_a_done_reaches_advantages_before_it(self):
        """The bootstrap cut, tested as a property: perturbing every reward and
        value after env 0's done must leave its pre-done advantages bit-identical.
        If any leak exists — through the delta's V(s') term or the recursive
        trace — this fails even for done conventions that happen to pass a
        single hand-computed example."""
        horizon, n_envs = 4, 2
        rewards = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], np.float32)
        values = np.full((horizon, n_envs), 0.7, np.float32)
        dones = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], np.float32)

        def gae(rw, vl, last_value):
            buf = RolloutBuffer(horizon, n_envs, 1, 1, 1)
            zeros = np.zeros((horizon, n_envs, 1), np.float32)
            _fill(buf, zeros, zeros, zeros, np.zeros((horizon, n_envs)), rw, dones, vl)
            buf.compute_gae(last_value, dones[-1])
            return buf.advantages

        adv_a = gae(rewards, values, np.array([0.3, 0.3], np.float32))

        rewards_b, values_b = rewards.copy(), values.copy()
        rewards_b[2:, 0] += 1000.0
        values_b[2:, 0] += 1000.0
        adv_b = gae(rewards_b, values_b, np.array([9999.0, 0.3], np.float32))

        np.testing.assert_array_equal(adv_a[:2, 0], adv_b[:2, 0])
        # Sanity that the perturbation was not a no-op: env 0's post-done
        # advantages (inside the perturbed region) did change.
        assert not np.allclose(adv_a[2:, 0], adv_b[2:, 0])


class TestRunningNorm:
    def test_matches_numpy_on_concatenated_stream(self):
        """The parallel-variance merge has plausible-looking wrong variants
        (missing cross term, biased warm-start count). Only agreement with
        numpy over the full concatenated stream rules them out."""
        rng = np.random.default_rng(0)
        chunks = [
            rng.normal(3.0, 2.0, size=(5, 3)),
            rng.normal(-1.0, 0.5, size=(50, 3)),
            rng.normal(0.0, 10.0, size=(17, 3)),
        ]
        rn = RunningNorm(3)
        for c in chunks:
            rn.update(c)
        allx = np.concatenate(chunks, axis=0)
        np.testing.assert_allclose(rn.mean, allx.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(rn.var, allx.var(axis=0), rtol=1e-10)

        x = rng.normal(size=(7, 3))
        expected = np.clip((x - allx.mean(0)) / np.sqrt(allx.var(0) + rn.eps), -10, 10)
        np.testing.assert_allclose(rn.normalize(x), expected, rtol=1e-5)

    def test_state_dict_round_trip_is_exact(self):
        """Deployment loads these stats into a fresh process. Any drift — a
        dtype cast, a re-derived count — shifts every observation the policy
        ever sees, with no error raised. The round-trip must be bit-exact."""
        rng = np.random.default_rng(1)
        rn = RunningNorm(4)
        for _ in range(5):
            rn.update(rng.normal(2.0, 3.0, size=(rng.integers(1, 40), 4)))

        rn2 = RunningNorm(4)
        rn2.load_state_dict(rn.state_dict())
        x = rng.normal(size=(11, 4))
        np.testing.assert_array_equal(rn.normalize(x), rn2.normalize(x))

        # Loaded stats must keep accumulating identically, not just normalizing.
        extra = rng.normal(size=(13, 4))
        rn.update(extra)
        rn2.update(extra)
        np.testing.assert_array_equal(rn.mean, rn2.mean)
        np.testing.assert_array_equal(rn.var, rn2.var)

    def test_normalize_clips_outliers(self):
        rn = RunningNorm(2)
        rn.update(np.zeros((10, 2)) + np.arange(10)[:, None])  # nonzero var
        out = rn.normalize(np.full((1, 2), 1e9))
        assert np.all(out <= 10.0)


def _rollout_synthetic(policy, buf, rng, horizon, n_envs, actor_dim, critic_dim):
    """Fill a buffer from random observations, with reward correlated to the
    action so advantages carry a real signal."""
    done_row = horizon // 2
    for t in range(horizon):
        aobs = rng.normal(size=(n_envs, actor_dim)).astype(np.float32)
        cobs = rng.normal(size=(n_envs, critic_dim)).astype(np.float32)
        with torch.no_grad():
            a, logp, v = policy.act(torch.as_tensor(aobs), torch.as_tensor(cobs))
        a_np = a.numpy()
        reward = -np.square(a_np - aobs[:, : a_np.shape[1]]).sum(-1)
        done = np.full(n_envs, 1.0 if t == done_row else 0.0, np.float32)
        buf.add(aobs, cobs, a_np, logp.numpy(), reward, done, v.numpy())
    buf.compute_gae(rng.normal(size=n_envs).astype(np.float32), np.zeros(n_envs, np.float32))


def _full_batch_pg_loss(policy, buf, clip=0.2):
    """The clipped surrogate over the whole buffer, against the stored logps.
    Before any update it is ~0 by construction (ratio == 1, normalized
    advantages have zero mean), so 'update reduces it' means 'clearly < 0'."""
    total = buf.horizon * buf.n_envs
    aobs = torch.as_tensor(buf.actor_obs.reshape(total, -1))
    cobs = torch.as_tensor(buf.critic_obs.reshape(total, -1))
    acts = torch.as_tensor(buf.actions.reshape(total, -1))
    old_logp = torch.as_tensor(buf.logps.reshape(total))
    adv = torch.as_tensor(buf.advantages.reshape(total))
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    with torch.no_grad():
        logp, _, _ = policy.evaluate(aobs, cobs, acts)
        ratio = (logp - old_logp).exp()
        loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
    return float(loss)


class TestPPOUpdate:
    def test_update_runs_and_reduces_policy_loss(self):
        """If gradients are detached somewhere, the sign of the surrogate is
        flipped, or old/new logps are swapped, the update still 'runs' — the
        surrogate just fails to go down. Measured on the full batch against
        the pre-update snapshot, where the starting loss is ~0."""
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        policy = ActorCritic(4, 6, 2, hidden=(32, 32))
        ppo = PPO(policy, lr=1e-3, epochs=6, n_minibatches=4)

        buf = RolloutBuffer(32, 8, 4, 6, 2)
        _rollout_synthetic(policy, buf, rng, 32, 8, 4, 6)

        before = _full_batch_pg_loss(policy, buf)
        metrics = ppo.update(buf)
        after = _full_batch_pg_loss(policy, buf)

        assert abs(before) < 1e-3  # ratio==1 baseline; if not, the comparison is meaningless
        assert after < before - 0.01
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            assert np.isfinite(metrics[key])

    def test_target_kl_stops_epochs(self):
        """target_kl exists to abort a diverging update; if the early stop is
        dead code, a huge-lr update runs all epochs and the KL explodes far
        past the threshold instead of stopping within one epoch of it."""
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        policy = ActorCritic(4, 6, 2, hidden=(32, 32))
        ppo = PPO(policy, lr=1.0, epochs=50, n_minibatches=4, target_kl=0.05)

        buf = RolloutBuffer(32, 8, 4, 6, 2)
        _rollout_synthetic(policy, buf, rng, 32, 8, 4, 6)
        ppo.update(buf)
        # Observe the stop through Adam's step counter: lr=1.0 blows past
        # target_kl within the first epoch (4 minibatch steps), so anything
        # near 50 epochs * 4 = 200 steps means the early stop is dead code.
        steps = int(next(iter(ppo.optimizer.state.values()))["step"])
        assert steps <= 8

    def test_same_torch_seed_gives_identical_losses(self):
        """Bit-level reproducibility is what makes a loss diff between two
        runs attributable to a code change. Any unseeded randomness — the
        minibatch shuffle is the classic offender — breaks this silently."""

        def run():
            torch.manual_seed(7)
            rng = np.random.default_rng(3)
            policy = ActorCritic(3, 5, 2, hidden=(16, 16))
            ppo = PPO(policy, epochs=2, n_minibatches=2)
            buf = RolloutBuffer(8, 4, 3, 5, 2)
            _rollout_synthetic(policy, buf, rng, 8, 4, 3, 5)
            return ppo.update(buf)

        m1, m2 = run(), run()
        assert m1.keys() == m2.keys()
        for k in m1:
            assert m1[k] == m2[k], f"{k}: {m1[k]} != {m2[k]}"


class _PointMass:
    """Deterministic 1-D point mass: state = position, action nudges it,
    reward = -|pos - target| with target 0. Episodes are 64 steps, all envs
    synchronized, auto-reset on done. Deliberately trivial: if PPO cannot
    solve this in seconds, the bug is in PPO, not the task."""

    def __init__(self, n_envs, ep_len=64, seed=0):
        self.n_envs, self.ep_len = n_envs, ep_len
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(n_envs, np.float32)
        self.t = 0

    def reset(self):
        self.pos = self.rng.uniform(-1.0, 1.0, self.n_envs).astype(np.float32)
        self.t = 0
        return self.pos[:, None].copy()

    def step(self, action):
        self.pos = np.clip(self.pos + 0.25 * np.clip(action, -1.0, 1.0), -3.0, 3.0).astype(np.float32)
        reward = -np.abs(self.pos).astype(np.float32)
        self.t += 1
        done = np.full(self.n_envs, float(self.t >= self.ep_len), np.float32)
        obs = self.reset() if self.t >= self.ep_len else self.pos[:, None].copy()
        return obs, reward, done


def _eval_return(policy, n_envs=32, seed=1234):
    env = _PointMass(n_envs, seed=seed)
    obs = env.reset()
    total = np.zeros(n_envs, np.float64)
    for _ in range(env.ep_len):
        with torch.no_grad():
            a, _, _ = policy.act(torch.as_tensor(obs), torch.as_tensor(obs), deterministic=True)
        obs, r, _ = env.step(a.numpy()[:, 0])
        total += r
    return float(total.mean())


class TestLearning:
    def test_ppo_learns_point_mass(self):
        """The end-to-end check: rollout -> GAE -> update, wired together, must
        actually improve a policy. Unit tests can all pass while a sign error
        at one interface (advantages negated, logps stale, obs misaligned by
        one step) makes the full loop learn nothing — only a learning curve
        catches that class of bug."""
        torch.manual_seed(0)
        n_envs, horizon = 16, 64
        env = _PointMass(n_envs, ep_len=horizon, seed=0)
        policy = ActorCritic(1, 1, 1, hidden=(32, 32))
        ppo = PPO(policy, lr=1e-3, epochs=4, n_minibatches=4)

        init_ret = _eval_return(policy)
        # Untrained head is near-zero -> the mass barely moves -> return is
        # roughly -ep_len * E|start| ~= -32. If this guard fails, the baseline
        # is not bad and the improvement assertion below proves nothing.
        assert init_ret < -15.0

        start = time.perf_counter()
        obs = env.reset()
        for _ in range(50):
            buf = RolloutBuffer(horizon, n_envs, 1, 1, 1)
            for _t in range(horizon):
                with torch.no_grad():
                    a, logp, v = policy.act(torch.as_tensor(obs), torch.as_tensor(obs))
                a_np = a.numpy()
                next_obs, r, done = env.step(a_np[:, 0])
                buf.add(obs, obs, a_np, logp.numpy(), r, done, v.numpy())
                obs = next_obs
            with torch.no_grad():
                _, _, last_v = policy.act(torch.as_tensor(obs), torch.as_tensor(obs))
            buf.compute_gae(last_v.numpy(), done)
            ppo.update(buf)
        elapsed = time.perf_counter() - start

        final_ret = _eval_return(policy)
        print(f"\npoint-mass learning: init={init_ret:.1f} final={final_ret:.1f} train={elapsed:.1f}s")
        # Comfortable margin, not a tuned threshold: a policy that learned
        # anything at all clears +10 over a ~-32 baseline; a broken loop
        # hovers at the baseline (or below it).
        assert final_ret > init_ret + 10.0
        assert elapsed < 60.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
