"""Task logic: the geometry of a catch, the failure reasons, and the reward's
degenerate optimum -- the one that would otherwise be discovered on the GPU."""

import numpy as np
import pytest

from harness import ballistics as bal
from harness.catch import (BallState, CatchConfig, Curriculum, Episode, HandState,
                           Phase, Stage, is_secured, phase_of, reward_terms,
                           terminate, total)

PALM = np.array([0.0, 0.0, 0.30])


def hand(action=None):
    return HandState(palm_pos=PALM, action=np.zeros(20) if action is None else action)


def held_ball(dz=0.04, speed=0.0, contacts=3):
    return BallState(pos=PALM + np.array([0.0, 0.0, dz]),
                     vel=np.array([0.0, 0.0, -speed]), contacts=contacts)


# -- configuration ---------------------------------------------------------

def test_validate_rejects_twenty_hertz():
    """A 5 cm toss is four control steps at 20 Hz. Refuse it up front."""
    with pytest.raises(ValueError, match="control steps of flight"):
        CatchConfig(stage=Stage.TOSS, ctrl_dt=0.05, spawn_height=0.05).validate()


def test_validate_accepts_one_hundred_hertz():
    CatchConfig(stage=Stage.TOSS, ctrl_dt=0.01, spawn_height=0.05).validate()


def test_validate_rejects_a_ball_that_outruns_the_control_step():
    cfg = CatchConfig(stage=Stage.TOSS, ctrl_dt=0.01, spawn_height=0.30,
                      ball_radius=0.01, min_flight_steps=1)
    with pytest.raises(ValueError, match="per control step"):
        cfg.validate()


# -- what counts as a catch ------------------------------------------------

def test_a_grazing_ball_is_not_a_catch():
    cfg = CatchConfig()
    grazed = BallState(pos=PALM + np.array([0.0, 0.0, 0.04]),
                       vel=np.array([0.0, 0.0, -2.0]), contacts=1)
    assert not is_secured(grazed, hand(), cfg)


def test_a_ball_held_slow_and_close_is_a_catch():
    assert is_secured(held_ball(), hand(), CatchConfig())


def test_a_ball_below_the_palm_is_not_held_however_slow():
    cfg = CatchConfig()
    under = BallState(pos=PALM - np.array([0.0, 0.0, 0.02]), vel=np.zeros(3), contacts=3)
    assert not is_secured(under, hand(), cfg)


def test_phases():
    cfg = CatchConfig()
    ep = Episode(steps=0)
    assert phase_of(held_ball(), hand(), ep, cfg) is Phase.SETTLE

    ep = Episode(steps=cfg.settle_steps)
    assert phase_of(held_ball(), hand(), ep, cfg) is Phase.READY, "held, but nothing has flown"

    airborne = BallState(pos=PALM + np.array([0.0, 0.0, 0.10]),
                         vel=np.array([0.0, 0.0, -1.0]), contacts=0)
    assert phase_of(airborne, hand(), ep, cfg) is Phase.FLIGHT

    ep.flights_completed = 1
    assert phase_of(held_ball(), hand(), ep, cfg) is Phase.SECURED


# -- termination -----------------------------------------------------------

@pytest.mark.parametrize("ball, ep, reason", [
    (BallState(PALM - np.array([0., 0., 0.2]), np.zeros(3)), Episode(steps=50), "dropped"),
    (BallState(PALM + np.array([0.4, 0., 0.05]), np.zeros(3)), Episode(steps=50), "out_of_bounds"),
    (BallState(PALM + np.array([0., 0., 0.04]), np.zeros(3), 3, 99.0), Episode(steps=50), "excess_force"),
    (held_ball(), Episode(steps=50, held_steps=30, flights_completed=1), "caught"),
    (held_ball(), Episode(steps=150), "never_launched"),
    (held_ball(), Episode(steps=400, flights_completed=1), "timeout"),
])
def test_termination_reasons(ball, ep, reason):
    cfg = CatchConfig(stage=Stage.TOSS)
    done, got = terminate(ball, hand(), ep, cfg)
    assert done and got == reason


def test_a_live_episode_is_not_terminated():
    done, reason = terminate(held_ball(), hand(), Episode(steps=50, flights_completed=1),
                             CatchConfig(stage=Stage.TOSS))
    assert not done and reason is None


# -- the degenerate optimum ------------------------------------------------

def _rollout(states, cfg, ep=None):
    """Score a scripted episode. `states` yields (ball, launched) per step."""
    ep = ep or Episode()
    prev = np.zeros(20)
    ret = 0.0
    for ball, launched in states:
        ep.steps += 1
        if launched:
            ep.flights_completed = 1
        ep.phase = phase_of(ball, hand(), ep, cfg)
        ep.held_steps = ep.held_steps + 1 if ep.phase is Phase.SECURED else 0
        ret += total(reward_terms(ball, hand(), ep, cfg, prev))
    return ret, ep


def _never_throw(cfg):
    return [(held_ball(), False) for _ in range(cfg.launch_deadline)]


def _throw_and_catch(cfg):
    """Rise, fall back to the palm, get caught, get held."""
    v0 = bal.launch_speed_for(cfg.spawn_height)
    pos, vel = PALM + np.array([0.0, 0.0, 0.01]), np.array([0.0, 0.0, v0])
    steps = [(held_ball(), False) for _ in range(cfg.settle_steps)]
    launched = False
    while True:
        pos, vel = bal.integrate(pos, vel, cfg.ctrl_dt)
        if pos[2] <= PALM[2] + 0.01 and vel[2] < 0:
            break
        steps.append((BallState(pos.copy(), vel.copy(), contacts=0), launched))
        launched = True
    steps += [(held_ball(), True) for _ in range(cfg.hold_steps)]
    return steps


def test_never_throwing_scores_strictly_worse_than_catching():
    """The failure this reward is designed against.

    Pay per step for holding and penalise the drop, and the optimal policy on a
    launch stage is to sit still forever: no risk, endless reward. Hold reward
    is therefore gated on a completed flight, so the do-nothing episode collects
    nothing at all.
    """
    cfg = CatchConfig(stage=Stage.TOSS, ctrl_dt=0.01, spawn_height=0.05)
    cfg.validate()
    idle, idle_ep = _rollout(_never_throw(cfg), cfg)
    caught, caught_ep = _rollout(_throw_and_catch(cfg), cfg)

    assert idle == pytest.approx(0.0), "idling must not accumulate reward"
    assert caught > idle
    assert caught_ep.flights_completed == 1
    assert terminate(held_ball(), hand(), idle_ep, cfg)[1] == "never_launched"


def test_the_flight_is_not_a_reward_desert():
    """Every airborne step must carry signal, or the flight has no gradient."""
    cfg = CatchConfig(stage=Stage.TOSS, ctrl_dt=0.01, spawn_height=0.05)
    ep = Episode(steps=cfg.settle_steps + 1, phase=Phase.FLIGHT)
    airborne = BallState(PALM + np.array([0.0, 0.0, 0.05]),
                         np.array([0.0, 0.0, -0.5]), contacts=0)
    assert reward_terms(airborne, hand(), ep, cfg, np.zeros(20))["track"] > 0


def test_tracking_pays_more_when_the_hand_is_under_the_ball():
    """The dense term must actually rank a good intercept above a bad one."""
    cfg = CatchConfig(stage=Stage.TOSS)
    ep = Episode(steps=cfg.settle_steps + 1, phase=Phase.FLIGHT)
    under = BallState(PALM + np.array([0.0, 0.0, 0.05]), np.array([0., 0., -0.5]))
    off = BallState(PALM + np.array([0.15, 0.0, 0.05]), np.array([0., 0., -0.5]))
    r_under = reward_terms(under, hand(), ep, cfg, np.zeros(20))["track"]
    r_off = reward_terms(off, hand(), ep, cfg, np.zeros(20))["track"]
    assert r_under > r_off


# -- curriculum ------------------------------------------------------------

def test_curriculum_holds_until_the_window_fills():
    cur = Curriculum(cfg=CatchConfig(), window=20)
    for _ in range(19):
        cur.record(True)
    assert not cur.update()


def test_curriculum_widens_on_measured_success_and_resets_the_window():
    cfg = CatchConfig(spawn_height=0.05)
    cur = Curriculum(cfg=cfg, window=20, step=0.02)
    for _ in range(20):
        cur.record(True)
    assert cur.update()
    assert cfg.spawn_height == pytest.approx(0.07)
    assert cur.success_rate == 0.0, "window clears so the next promotion is earned again"


def test_curriculum_does_not_widen_below_the_bar():
    cfg = CatchConfig(spawn_height=0.05)
    cur = Curriculum(cfg=cfg, window=20, promote_at=0.6)
    for i in range(20):
        cur.record(i < 5)
    assert not cur.update()
    assert cfg.spawn_height == pytest.approx(0.05)


def test_curriculum_stops_at_the_ceiling():
    cfg = CatchConfig(spawn_height=0.30, max_spawn_height=0.30)
    cur = Curriculum(cfg=cfg, window=20)
    for _ in range(20):
        cur.record(True)
    assert not cur.update()
