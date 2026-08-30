"""A scripted catch: an open hand that closes as the ball arrives.

This is the gate before any GPU hours go into PPO. If a controller holding
ground-truth ball state cannot catch, no policy will, and the fault is in the
environment -- unreachable `is_secured` thresholds, a ball that bounces out, a
palm facing the wrong way -- not in the learning. `franka-isaac` uses
`stack_sm.py` the same way, and for the same reason.

The one genuinely fiddly part is turning "curl the fingers" into an action, and
it is fiddly for a reason worth stating: the action space is each joint's range
mapped onto [-1, 1], and the Shadow Hand's ranges do not share a sign
convention. FFJ1 runs [0, 1.571], so curling it means +1. THJ0 runs [-1.571, 0],
so flexing the thumb means **-1**. A grasp written directly in action space
gets the thumb backwards and nothing complains: the hand closes four fingers
and sticks its thumb out.

So the poses below are in radians, per joint, and are converted through the
limits actually reported by the simulator. `tests/test_grasp.py` pins the
conversion against the limit table read off the asset on the box.

No isaaclab import, so it is testable on a laptop.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

#: Target angle (radians) per joint-name suffix, for a closed cradle. Chosen to
#: curl the fingers without splaying them: the J3/J4 abduction joints stay near
#: neutral, because spreading the fingers is how a ball falls through them.
GRASP_POSE: Dict[str, float] = {
    "WRJ1": 0.0, "WRJ0": 0.0,
    "FFJ3": 0.0, "MFJ3": 0.0, "RFJ3": 0.0, "LFJ3": 0.0, "LFJ4": 0.15,
    "FFJ2": 1.30, "MFJ2": 1.30, "RFJ2": 1.30, "LFJ2": 1.30,
    "FFJ1": 1.30, "MFJ1": 1.30, "RFJ1": 1.30, "LFJ1": 1.30,
    # Tendon-coupled, so never commanded -- but needed to write a reset pose,
    # which sets joint positions directly rather than actuator targets.
    "FFJ0": 1.30, "MFJ0": 1.30, "RFJ0": 1.30, "LFJ0": 1.30,
    # Thumb opposition. THJ0's range is [-1.571, 0], so flexion is negative --
    # the sign trap this module exists to keep out of the action space.
    "THJ4": 0.90, "THJ3": 0.60, "THJ2": 0.00, "THJ1": 0.35, "THJ0": -0.80,
}

#: Open, waiting -- and genuinely CUPPED, not flat.
#:
#: The first version held the fingers nearly straight (0.25 rad). Measured on
#: the box, that puts the fingertips 16.9 cm above the palm, so a ball spawned
#: clear of them falls 21 cm and arrives at ~2 m/s onto the *tips* -- a convex
#: surface it rolls straight off. The ball was held for a mean of 1.7 steps per
#: episode against the 60 consecutive it needs.
#:
#: Curling to ~0.7 rad does two things at once: it lowers the fingertips, so the
#: ball starts nearer and lands slower, and it turns the hand into a concave
#: pocket the ball can settle into. It also leaves room to close further --
#: GRASP_POSE is 1.3 rad, so there is still travel left for the catch itself.
OPEN_POSE: Dict[str, float] = {
    "WRJ1": 0.0, "WRJ0": 0.0,
    "FFJ3": 0.0, "MFJ3": 0.0, "RFJ3": 0.0, "LFJ3": 0.0, "LFJ4": 0.10,
    "FFJ2": 0.70, "MFJ2": 0.70, "RFJ2": 0.70, "LFJ2": 0.70,
    "FFJ1": 0.70, "MFJ1": 0.70, "RFJ1": 0.70, "LFJ1": 0.70,
    "FFJ0": 0.70, "MFJ0": 0.70, "RFJ0": 0.70, "LFJ0": 0.70,
    "THJ4": 0.60, "THJ3": 0.40, "THJ2": 0.00, "THJ1": 0.20, "THJ0": -0.40,
}


def _suffix(name: str) -> str:
    return name.rsplit("_", 1)[-1]


def pose_to_action(pose: Dict[str, float], joint_names: Sequence[str],
                   lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Radians -> the [-1, 1] action the environment expects.

    Exactly inverts `CatchEnv._apply_action`, which maps
    ``target = lo + 0.5 * (a + 1) * (hi - lo)``. A target outside a joint's
    range is clipped rather than silently extrapolated, so a pose written for
    one hand cannot quietly command another past its limits.
    """
    if not (len(joint_names) == len(lower) == len(upper)):
        raise ValueError(
            f"{len(joint_names)} names, {len(lower)} lower and {len(upper)} upper "
            f"limits; all three describe the same joints and must agree")
    out = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        key = _suffix(name)
        if key not in pose:
            raise KeyError(f"no pose entry for joint {name!r} (suffix {key!r})")
        span = upper[i] - lower[i]
        if span <= 0.0:
            continue
        q = np.clip(pose[key], lower[i], upper[i])
        out[i] = np.clip(2.0 * (q - lower[i]) / span - 1.0, -1.0, 1.0)
    return out


def closure(time_to_arrival: float | None, lead_time: float) -> float:
    """How closed the hand should be, in [0, 1].

    A ramp rather than a trigger. Fingers are position-controlled through a
    spring (stiffness 1.0, damping 0.1 on this asset), so they take time to
    arrive; snapping the target from open to closed at the last instant means
    the hand is still opening when the ball lands. Starting the ramp
    `lead_time` before arrival gives the fingers somewhere to be.

    `None` -- the ball is not coming -- holds the hand open. Treating it as
    zero instead would read as "arriving now" and clench at nothing.
    """
    if lead_time <= 0.0:
        raise ValueError(f"lead_time must be positive, got {lead_time}")
    if time_to_arrival is None:
        return 0.0
    return float(np.clip((lead_time - time_to_arrival) / lead_time, 0.0, 1.0))


def action(time_to_arrival: float | None, open_a: np.ndarray, closed_a: np.ndarray,
           lead_time: float) -> np.ndarray:
    """Interpolate between the open and closed actions by closure."""
    c = closure(time_to_arrival, lead_time)
    return (1.0 - c) * open_a + c * closed_a


def pose_to_qpos(pose: Dict[str, float], joint_names: Sequence[str],
                 lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Radians per joint, clipped to range. For writing a reset state.

    Distinct from `pose_to_action`: a reset writes joint POSITIONS directly, so
    it covers all 24 joints including the four tendon-coupled ones an action
    can never command.
    """
    out = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        key = _suffix(name)
        if key not in pose:
            raise KeyError(f"no pose entry for joint {name!r} (suffix {key!r})")
        out[i] = float(np.clip(pose[key], lower[i], upper[i]))
    return out
