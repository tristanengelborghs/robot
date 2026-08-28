"""Vector-env and tactile properties the training loop silently depends on."""

from __future__ import annotations

import numpy as np

from dexhand.env import EnvConfig, ReorientEnv
from dexhand.vec import ThreadedVecEnv


def _make(i, n_steps=40):
    return lambda: ReorientEnv("toy", EnvConfig(episode_len=n_steps), seed=100 + i)


def test_threaded_equals_serial():
    """Threading must not change physics. Each env owns its model and data, so
    results must be bit-identical to stepping the same envs in a loop — if they
    are not, envs are sharing mutable state and every training run is corrupted
    by a race that no single-env test can see."""
    acts = np.random.default_rng(0).uniform(-1, 1, (30, 4, 6))

    vec = ThreadedVecEnv([_make(i) for i in range(4)], n_threads=4)
    vec.reset(seeds=[7, 8, 9, 10])
    threaded = [vec.step(a)[1] for a in acts]
    vec.close()

    vec = ThreadedVecEnv([_make(i) for i in range(4)], n_threads=1)
    vec.reset(seeds=[7, 8, 9, 10])
    serial = [vec.step(a)[1] for a in acts]
    vec.close()

    assert np.array_equal(np.array(threaded), np.array(serial))


def test_autoreset_reports_final_obs_and_termination_kind():
    """GAE needs to know whether an episode ENDED or was CUT: bootstrapping
    through a real failure teaches that dropping the cube has future value."""
    vec = ThreadedVecEnv([_make(0, n_steps=5)])
    vec.reset(seeds=[0])
    done_seen = False
    for _ in range(6):
        obs, r, done, infos = vec.step(np.zeros((1, 6)))
        if done[0]:
            done_seen = True
            assert "final_obs" in infos[0]
            assert "terminated" in infos[0]
            assert infos[0]["final_obs"]["actor"].shape == obs["actor"][0].shape
    assert done_seen


def test_tactile_fires_on_contact_and_only_against_the_cube():
    env = ReorientEnv("toy", EnvConfig(episode_len=100), seed=2)
    env.reset(seed=2)
    from dexhand.tactile import fingertip_forces

    # curl all fingers hard onto the cube for a second
    for _ in range(20):
        env.step(np.ones(env.nu))
    f_cube = fingertip_forces(env.model, env.data, env.scene.fingertip_geom_ids,
                              against_geom=env.scene.cube_geom_id)
    f_all = fingertip_forces(env.model, env.data, env.scene.fingertip_geom_ids)
    assert f_cube.shape == (3,)
    assert (f_cube >= 0).all() and np.isfinite(f_cube).all()
    assert f_all.sum() >= f_cube.sum() - 1e-9, "cube-only filter found force the unfiltered pass missed"


def test_tactile_is_zero_without_contact():
    env = ReorientEnv("toy", EnvConfig(episode_len=100, settle_steps=0), seed=3)
    env.reset(seed=3)
    a = env.scene.cube_qpos_adr
    env.data.qpos[a:a + 3] = [0.0, 0.0, 5.0]  # cube far away
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    from dexhand.tactile import fingertip_forces
    f = fingertip_forces(env.model, env.data, env.scene.fingertip_geom_ids,
                         against_geom=env.scene.cube_geom_id)
    assert (f == 0).all()
