"""The GPU reward path must agree with the tested NumPy reference.

`rewards.py` is a second implementation of `harness/catch.py` -- vectorised,
torch, and what actually runs on thousands of environments. Duplication is only
safe if something forces the two to agree, and this is that something. It runs
on the laptop because `rewards.py` deliberately imports no isaaclab; it is
loaded straight from its file so the package __init__ (which does import
isaaclab, via the env) is never touched.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from harness import catch as ref

torch = pytest.importorskip("torch")

_PATH = (pathlib.Path(__file__).resolve().parent.parent
         / "source/catching/catching/tasks/direct/catch/rewards.py")
_spec = importlib.util.spec_from_file_location("catch_rewards", _PATH)
rw = importlib.util.module_from_spec(_spec)
# Registered before execution: @dataclass resolves annotations through
# sys.modules[cls.__module__], and an unregistered module makes that None.
sys.modules[_spec.name] = rw
_spec.loader.exec_module(rw)

N = 512
NDOF = 20


def _cfgs(stage: ref.Stage):
    """One config, expressed both ways. Built field by field so a field added
    on one side and forgotten on the other fails here."""
    a = ref.CatchConfig(stage=stage)
    b = rw.CatchCfg(
        is_drop_stage=stage is ref.Stage.DROP,
        is_pop_stage=stage is ref.Stage.POP,
        settle_steps=a.settle_steps, episode_len=a.episode_len,
        spawn_height=a.spawn_height, catch_radius=a.catch_radius,
        secure_speed=a.secure_speed, min_contacts=a.min_contacts,
        hold_steps=a.hold_steps, drop_below=a.drop_below,
        drop_lateral=a.drop_lateral, max_force=a.max_force,
        launch_deadline=a.launch_deadline, track_scale=a.track_scale,
        secure_bonus=a.secure_bonus, hold_scale=a.hold_scale,
        apex_scale=a.apex_scale, drop_penalty=a.drop_penalty,
        force_penalty=a.force_penalty, action_rate_penalty=a.action_rate_penalty,
        ctrl_penalty=a.ctrl_penalty,
    )
    return a, b


def _states(seed=0):
    """Random states spanning every branch: settling, airborne, held, dropped,
    out of bounds, over-force, out of time."""
    rng = np.random.default_rng(seed)
    palm = np.tile(np.array([0.0, 0.0, 0.30]), (N, 1))
    return dict(
        palm=palm,
        ball_pos=palm + rng.uniform(-0.3, 0.3, (N, 3)),
        ball_vel=rng.uniform(-2.5, 2.5, (N, 3)),
        contacts=rng.integers(0, 6, N),
        max_force=rng.uniform(0.0, 60.0, N),
        action=rng.uniform(-1, 1, (N, NDOF)),
        prev_action=rng.uniform(-1, 1, (N, NDOF)),
        steps=rng.integers(0, 420, N),
        held_steps=rng.integers(0, 40, N),
        flights=rng.integers(0, 2, N),
        peak=palm[:, 2] + rng.uniform(0.0, 0.2, N),
    )


def test_phase_and_reason_codes_match():
    assert rw.SETTLE == ref.PHASE_CODES.index("settle")
    assert rw.FLIGHT == ref.PHASE_CODES.index("flight")
    assert rw.SECURED == ref.PHASE_CODES.index("secured")
    assert rw.TIMEOUT == ref.REASON_CODES.index("timeout")
    assert rw.NEVER_LAUNCHED == ref.REASON_CODES.index("never_launched")


@pytest.mark.parametrize("stage", [ref.Stage.DROP, ref.Stage.TOSS, ref.Stage.POP])
def test_phase_reward_and_termination_agree(stage):
    cfg_np, cfg_t = _cfgs(stage)
    s = _states(seed=hash(stage.value) % 1000)

    t = lambda x, d=torch.float32: torch.as_tensor(x, dtype=d)  # noqa: E731
    phase_t = rw.phase_of(t(s["steps"], torch.long), t(s["ball_pos"]), t(s["ball_vel"]),
                          t(s["contacts"], torch.long), t(s["palm"]),
                          t(s["flights"], torch.long), cfg_t)
    terms_t = rw.reward_terms(t(s["ball_pos"]), t(s["ball_vel"]), t(s["contacts"], torch.long),
                              t(s["max_force"]), t(s["palm"]), t(s["action"]),
                              t(s["prev_action"]), phase_t, t(s["held_steps"], torch.long),
                              t(s["peak"]), cfg_t)
    total_t = rw.total(terms_t).numpy()
    reason_t = rw.terminate(t(s["ball_pos"]), t(s["palm"]), t(s["max_force"]),
                            t(s["steps"], torch.long), t(s["held_steps"], torch.long),
                            t(s["flights"], torch.long), cfg_t).numpy()

    for i in range(N):
        ball = ref.BallState(pos=s["ball_pos"][i], vel=s["ball_vel"][i],
                             contacts=int(s["contacts"][i]), max_force=float(s["max_force"][i]))
        hand = ref.HandState(palm_pos=s["palm"][i], action=s["action"][i])
        ep = ref.Episode(steps=int(s["steps"][i]), flights_completed=int(s["flights"][i]),
                         held_steps=int(s["held_steps"][i]), peak_height=float(s["peak"][i]))
        ep.phase = ref.phase_of(ball, hand, ep, cfg_np)

        assert ref.phase_code(ep.phase) == phase_t[i].item(), f"phase, env {i}"
        got = ref.total(ref.reward_terms(ball, hand, ep, cfg_np, s["prev_action"][i]))
        assert got == pytest.approx(float(total_t[i]), abs=1e-4), f"reward, env {i}"
        _, reason = ref.terminate(ball, hand, ep, cfg_np)
        assert ref.reason_code(reason) == reason_t[i], f"termination, env {i}"


def test_intercept_prediction_agrees():
    cfg_np, cfg_t = _cfgs(ref.Stage.TOSS)
    s = _states(seed=7)
    pos = torch.as_tensor(s["ball_pos"], dtype=torch.float32)
    vel = torch.as_tensor(s["ball_vel"], dtype=torch.float32)
    plane = torch.as_tensor(s["palm"][:, 2], dtype=torch.float32)
    xy, valid = rw.predict_intercept_xy(pos, vel, plane)
    tt, tvalid = rw.time_to_plane(pos, vel, plane)
    from harness import ballistics as bal
    for i in range(N):
        hit = bal.predict_intercept(s["ball_pos"][i], s["ball_vel"][i], float(s["palm"][i, 2]))
        assert bool(valid[i]) == (hit is not None), f"validity, env {i}"
        assert bool(tvalid[i]) == (hit is not None), f"time validity, env {i}"
        if hit is not None:
            assert xy[i].numpy() == pytest.approx(hit[0][:2], abs=1e-3), f"xy, env {i}"
            assert float(tt[i]) == pytest.approx(hit[1], abs=1e-3), f"time, env {i}"


@pytest.mark.parametrize("stage", [ref.Stage.DROP, ref.Stage.TOSS])
@pytest.mark.parametrize("hz,height", [(20, 0.05), (50, 0.05), (120, 0.05),
                                       (240, 0.08), (240, 0.30), (1000, 0.05)])
def test_the_two_validators_agree(stage, hz, height):
    """The reference validator is readable; the torch one is the only one the
    environment can reach. The first smoke run shipped a config the reference
    would have rejected, because nothing called it. They must not drift."""
    cfg_np, cfg_t = _cfgs(stage)
    cfg_np.ctrl_dt = cfg_t.ctrl_dt = 1.0 / hz
    cfg_np.spawn_height = cfg_t.spawn_height = height

    def raised(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    assert raised(cfg_np.validate) == raised(cfg_t.validate), (
        f"{stage.value} at {hz} Hz, {height*100:.0f} cm: validators disagree")
