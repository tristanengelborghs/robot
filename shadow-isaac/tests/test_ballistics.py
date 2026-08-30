"""Ballistics against closed form, and against flying the ball step by step."""

import numpy as np
import pytest

from harness import ballistics as bal


def test_apex_and_launch_speed_round_trip():
    for h in (0.02, 0.05, 0.2, 1.0):
        assert bal.apex_height(bal.launch_speed_for(h)) == pytest.approx(h)


def test_a_falling_ball_gains_no_height():
    assert bal.apex_height(-3.0) == 0.0
    assert bal.flight_time(-3.0) == 0.0


def test_integrate_matches_the_analytic_apex():
    """Fly a 5 cm toss at 1 kHz and check where it actually tops out."""
    v0 = bal.launch_speed_for(0.05)
    pos, vel = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, v0])
    peak = 0.0
    for _ in range(1000):
        pos, vel = bal.integrate(pos, vel, 1e-3)
        peak = max(peak, pos[2])
        if vel[2] < 0 and pos[2] < 0:
            break
    assert peak == pytest.approx(0.05, abs=1e-4)


def test_predict_intercept_agrees_with_flying_there():
    pos = np.array([0.10, -0.05, 0.30])
    vel = np.array([0.20, 0.10, 1.50])
    plane = 0.25
    point, t = bal.predict_intercept(pos, vel, plane)

    p, v = pos.copy(), vel.copy()
    dt = 1e-4
    flown = 0.0
    while flown < t:
        p, v = bal.integrate(p, v, dt)
        flown += dt
    assert p[2] == pytest.approx(plane, abs=2e-3)
    assert point[:2] == pytest.approx(p[:2], abs=2e-3)
    assert v[2] < 0, "the intercept must be the descending crossing"


def test_intercept_is_none_when_the_apex_falls_short():
    # rising at 0.5 m/s from z=0 reaches ~1.3 cm; a plane at 20 cm is unreachable
    assert bal.predict_intercept(np.zeros(3), np.array([0.0, 0.0, 0.5]), 0.20) is None


def test_intercept_below_the_plane_and_falling_never_arrives():
    assert bal.predict_intercept(np.array([0.0, 0.0, 0.1]),
                                 np.array([0.0, 0.0, -1.0]), 0.20) is None


def test_travel_per_control_step_is_the_rate_argument():
    """The number that rules out 20 Hz: a 5 cm toss lands at ~1 m/s."""
    v = bal.launch_speed_for(0.05)
    assert bal.travel_per_control_step(v, 0.05) == pytest.approx(0.0495, abs=1e-3)
    assert bal.travel_per_control_step(v, 0.01) == pytest.approx(0.0099, abs=1e-3)
