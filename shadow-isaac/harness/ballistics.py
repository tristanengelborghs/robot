"""Closed-form ballistics for a ball in free flight.

Everything here is analytic, so it runs on the laptop and needs no simulator.
That matters twice over. It is the fake plant the task logic is tested against,
and it is a *prediction* the policy is given: between release and catch the
hand has no influence on the ball whatsoever, so the only thing worth learning
during flight is where the ball is going to arrive and when.

That prediction is what fills the hole in the middle of the reward. Without it
the flight phase carries no gradient at all -- the policy acts, the ball
ignores it, and the outcome lands many steps later. With it, every step of the
flight has a dense, honest signal: reduce the distance between the palm and the
point the ball will cross palm height.

Conventions: z is up, gravity is positive-valued and applied downward, and no
drag. Drag on a light ball over a 20 cm toss changes the landing point by well
under a millimetre, which is far below the noise the contact solver adds.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

G = 9.81


def apex_height(v_up: float, g: float = G) -> float:
    """Height gained above the launch point. Zero for a ball already falling."""
    return max(v_up, 0.0) ** 2 / (2.0 * g)


def launch_speed_for(height: float, g: float = G) -> float:
    """The upward speed that reaches `height`. Inverse of apex_height."""
    if height < 0.0:
        raise ValueError(f"height must be non-negative, got {height}")
    return float(np.sqrt(2.0 * g * height))


def flight_time(v_up: float, g: float = G) -> float:
    """Seconds from launch until the ball returns to the launch height."""
    return 2.0 * max(v_up, 0.0) / g


def time_to_plane(z: float, v_up: float, z_plane: float, g: float = G) -> Optional[float]:
    """Seconds until the ball crosses `z_plane` on the way DOWN, or None.

    None means the ball never gets there: it is already below the plane and
    rising away from it, or its apex falls short. Callers must treat None as
    "there is no catch to prepare for", not as zero -- a zero would tell the
    policy the ball is arriving this instant and yank the hand.

    The descending root is always the larger one, which is why the sign is
    fixed rather than chosen by inspection. Picking the smaller root gives the
    ascending crossing, which for a ball thrown up from inside the hand is
    ~0 seconds away and is never the crossing you want to catch at.
    """
    disc = v_up * v_up + 2.0 * g * (z - z_plane)
    if disc < 0.0:
        return None
    t = (v_up + np.sqrt(disc)) / g
    return float(t) if t > 0.0 else None


def predict_intercept(pos: np.ndarray, vel: np.ndarray, z_plane: float,
                      g: float = G) -> Optional[Tuple[np.ndarray, float]]:
    """Where and when the ball crosses `z_plane` descending.

    Returns (xyz point on the plane, seconds until it arrives), or None if it
    never arrives. Horizontal motion is unaccelerated, so the xy is a straight
    extrapolation over the same interval.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    t = time_to_plane(float(pos[2]), float(vel[2]), z_plane, g)
    if t is None:
        return None
    point = np.array([pos[0] + vel[0] * t, pos[1] + vel[1] * t, z_plane])
    return point, t


def integrate(pos: np.ndarray, vel: np.ndarray, dt: float,
              g: float = G) -> Tuple[np.ndarray, np.ndarray]:
    """Exact free-flight update over dt. The fake plant the tests fly."""
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    acc = np.array([0.0, 0.0, -g])
    return pos + vel * dt + 0.5 * acc * dt * dt, vel + acc * dt


def travel_per_control_step(v_impact: float, ctrl_dt: float) -> float:
    """How far the ball moves between two decisions. The rate sanity check.

    A control period that lets the ball cross its own diameter is a control
    period the policy cannot aim a catch through, however good the policy is.
    `catch.CatchConfig.validate` refuses such a configuration rather than
    letting it show up later as a policy that mysteriously will not learn.
    """
    return abs(v_impact) * ctrl_dt
