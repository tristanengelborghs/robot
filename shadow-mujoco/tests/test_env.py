"""Environment behaviour on the toy hand: no downloads, no GPU, seconds to run.

The cases here are the silent failures of RL environments: a reward that can be
farmed without acting, physics that differs between "identical" resets, an
observation that leaks privileged state into the student view, termination that
never triggers. Each of these trains without erroring and produces a policy
that learned the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from dexhand import rotations as rot
from dexhand.env import EnvConfig, ReorientEnv
from dexhand.hands import get_hand
from dexhand.randomize import DRConfig
from dexhand.scene import build_scene


@pytest.fixture(scope="module")
def env():
    return ReorientEnv("toy", EnvConfig(episode_len=50), seed=0)


def test_spaces_and_obs_shapes(env):
    obs, _ = env.reset(seed=0)
    for key in ("actor", "critic", "student"):
        assert obs[key].shape == env.observation_space[key].shape
        assert np.isfinite(obs[key]).all()
    assert env.action_space.shape == (6,)


def test_student_obs_has_no_cube_pose():
    """The student view exists to prove the pipeline works without privileged
    state. If cube pose leaks in, distillation 'works' in sim and the student
    is untrainable from real sensors. Checked by teleporting the cube (away
    from every fingertip, so tactile cannot change either) and asserting the
    student view is bit-identical while the actor view is not."""
    import mujoco

    e = ReorientEnv("toy", EnvConfig(episode_len=50, dr=DRConfig(enabled=False)), seed=0)
    e.reset(seed=0)
    a = e.scene.cube_qpos_adr
    e.data.qpos[a:a + 3] = [0.0, 0.0, 2.0]      # far above the hand: no contacts
    mujoco.mj_forward(e.model, e.data)
    before = e._obs()
    e.data.qpos[a:a + 3] = [0.5, 0.5, 2.0]
    e.data.qpos[a + 3:a + 7] = [0.0, 1.0, 0.0, 0.0]
    mujoco.mj_forward(e.model, e.data)
    after = e._obs()
    assert np.array_equal(before["student"], after["student"]), "cube pose leaked into the student view"
    assert not np.array_equal(before["actor"], after["actor"])


def test_deterministic_given_seed():
    def rollout():
        e = ReorientEnv("toy", EnvConfig(episode_len=30), seed=3)
        o, _ = e.reset(seed=3)
        acts = np.random.default_rng(9).uniform(-1, 1, (30, e.nu))
        rews, obs_sum = [], 0.0
        for a in acts:
            o, r, te, tr, _ = e.step(a)
            rews.append(r)
            obs_sum += float(o["actor"].sum())
            if te or tr:
                break
        return np.array(rews), obs_sum

    (r1, s1), (r2, s2) = rollout(), rollout()
    assert np.array_equal(r1, r2) and s1 == s2


def test_dr_actually_changes_physics():
    """Two resets with different seeds must produce different dynamics, or DR
    is a no-op and the 'robust' policy was trained on a single world."""
    e = ReorientEnv("toy", EnvConfig(episode_len=30), seed=0)
    e.reset(seed=0)
    m0 = float(e.model.body_mass[e.scene.cube_body_id])
    f0 = e.model.geom_friction[:, 0].copy()
    e.reset(seed=1)
    m1 = float(e.model.body_mass[e.scene.cube_body_id])
    f1 = e.model.geom_friction[:, 0].copy()
    assert m0 != m1 or not np.allclose(f0, f1)


def test_dr_does_not_random_walk():
    """Two hundred resets must stay inside the configured band for EVERY
    randomized field. In-place perturbation without a pristine snapshot
    compounds multiplicatively (x1.3 then x1.3 then ...) and drifts training
    physics arbitrarily far from nominal within a few hundred episodes."""
    from dexhand.scene import CUBE_MASS

    e = ReorientEnv("toy", EnvConfig(episode_len=10), seed=0)
    pristine_friction = e.dr._pristine["geom_friction"][:, 0]
    pristine_damping = e.dr._pristine["dof_damping"]
    pristine_gain = e.dr._pristine["actuator_gainprm"][:, 0]
    cfg = e.cfg.dr
    for i in range(200):
        e.reset(seed=i)
        m = float(e.model.body_mass[e.scene.cube_body_id])
        assert CUBE_MASS * cfg.cube_mass[0] * 0.99 <= m <= CUBE_MASS * cfg.cube_mass[1] * 1.01
        ratio = e.model.geom_friction[:, 0] / pristine_friction
        assert (ratio >= cfg.friction[0] * 0.99).all() and (ratio <= cfg.friction[1] * 1.01).all(), \
            f"reset {i}: friction ratio {ratio.min():.3f}..{ratio.max():.3f} left the band"
        ratio = e.model.dof_damping / pristine_damping
        assert (ratio >= cfg.damping[0] * 0.99).all() and (ratio <= cfg.damping[1] * 1.01).all()
        ratio = e.model.actuator_gainprm[:, 0] / pristine_gain
        assert (ratio >= cfg.actuator_gain[0] * 0.99).all() and (ratio <= cfg.actuator_gain[1] * 1.01).all()


def test_dr_mass_reaches_the_dynamics():
    """Writing body_mass without mj_setConst leaves body_subtreemass stale, so
    a 'heavier' cube still accelerates like the nominal one — DR that changes a
    number and not the physics."""
    e = ReorientEnv("toy", EnvConfig(episode_len=10), seed=0)
    e.reset(seed=0)
    cb = e.scene.cube_body_id
    assert e.model.body_subtreemass[cb] == pytest.approx(e.model.body_mass[cb])
    assert e.model.body_inertia[cb][0] / e.model.body_mass[cb] == pytest.approx(
        e.dr._pristine["body_inertia"][cb][0] / e.dr._pristine["body_mass"][cb])


def test_goals_are_born_outside_the_success_threshold():
    """Regression for the live-caught reward exploit: with max_goal_angle under
    the success threshold, every sampled goal was instantly 'achieved' and the
    first smoke run reported 50 successes/episode at iteration 1 for a policy
    that had learned nothing."""
    cfg = EnvConfig(episode_len=50, max_goal_angle=0.3, success_thresh=0.4)
    e = ReorientEnv("toy", cfg, seed=0)
    for i in range(20):
        e.reset(seed=i)
        _, cube_quat = e._cube_pose()
        assert rot.rot_dist(cube_quat, e._goal) > cfg.success_thresh


def test_success_resamples_goal_and_pays_once():
    e = ReorientEnv("toy", EnvConfig(episode_len=50), seed=0)
    e.reset(seed=0)
    _, cube_quat = e._cube_pose()
    e._set_goal(cube_quat.copy())  # force imminent success
    obs, r, te, tr, info = e.step(np.zeros(e.nu))
    assert info["is_success"]
    assert info["successes"] == 1
    assert r > e.cfg.success_bonus * 0.5
    assert rot.rot_dist(e._cube_pose()[1], e._goal) > e.cfg.success_thresh, \
        "the new goal must not already be satisfied"


def test_drop_terminates_with_penalty():
    e = ReorientEnv("toy", EnvConfig(episode_len=200), seed=0)
    e.reset(seed=0)
    a = e.scene.cube_qpos_adr
    e.data.qpos[a:a + 3] = [0.0, 0.0, -0.5]  # teleport the cube to the floor
    obs, r, te, tr, info = e.step(np.zeros(e.nu))
    assert te and info["dropped"]
    assert r < 0


def test_episode_truncates_at_length():
    e = ReorientEnv("toy", EnvConfig(episode_len=5), seed=0)
    e.reset(seed=0)
    for i in range(5):
        obs, r, te, tr, info = e.step(np.zeros(e.nu))
    assert tr and not te


def test_action_smoothing_limits_ctrl_slew(env):
    """The EMA is a safety property: a full-range action flip must not command
    a full-range ctrl flip within one step."""
    env.reset(seed=0)
    env.step(np.ones(env.nu))
    c_hi = env.data.ctrl.copy()
    env.step(-np.ones(env.nu))
    c_lo = env.data.ctrl.copy()
    full_swing = (env.scene.ctrl_range[:, 1] - env.scene.ctrl_range[:, 0])
    assert (np.abs(c_hi - c_lo) < 0.6 * full_swing).all()


def test_solver_options_survive_composition():
    """MjSpec attachment silently drops the child's <option> on conflict; the
    hand was tuned with elliptic cones and impratio 10, and losing them makes
    every grasp softer with no error anywhere."""
    s = build_scene(get_hand("toy"))
    import mujoco
    assert s.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert s.model.opt.impratio == 10.0
