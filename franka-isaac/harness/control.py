"""Geometry and bookkeeping shared by the scripted controllers.

Both task controllers -- stacking and pick-and-lift -- need the same handful of
things: how a pose error becomes an action for an IK-relative task, what a yaw
error means for an object with square symmetry, and a record of which state
followed which. None of it is task-specific, so it lives here rather than in
whichever controller happened to be written first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

# A square cube looks the same every quarter turn, so a yaw error only ever
# needs to be driven into this half-window.
YAW_SYMMETRY = math.pi / 2


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle into ``[-pi, pi)``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_error(target: float, current: float) -> float:
    """Smallest rotation from ``current`` to ``target``, up to cube symmetry.

    A square cube presents an identical face every quarter turn, so aligning to
    within a quarter turn is aligning. Without this the controller would happily
    unwind 80 degrees to reach a grasp it was already in.
    """
    return wrap_to_pi(target - current + YAW_SYMMETRY / 2) % YAW_SYMMETRY - YAW_SYMMETRY / 2


def yaw_from_quat(quat: np.ndarray) -> float:
    """Yaw of a ``wxyz`` quaternion, in radians.

    Isaac Lab reports orientations as ``wxyz`` in the simulator (the ``xyzw``
    conversion happens only on the way into an HDF5 file), so this takes them in
    that order.
    """
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def horizontal_distance(vec: np.ndarray) -> float:
    """Length of ``vec`` in the xy plane, ignoring height."""
    return float(np.linalg.norm(vec[:2]))


@dataclass
class Transition:
    """One state change, for the log the deliverable asks for."""

    step: int
    frm: Enum
    to: Enum
    reason: str


@dataclass
class EpisodeLog:
    """What a single scripted episode did, for the state-transition log."""

    transitions: list[Transition]
    steps: int = 0
    succeeded: bool = False

    def format(self) -> str:
        lines = [f"{'step':>6}  {'from':<14} -> {'to':<14} reason"]
        lines += [f"{t.step:>6}  {t.frm.value:<14} -> {t.to.value:<14} {t.reason}" for t in self.transitions]
        lines.append(f"{self.steps} steps, {'success' if self.succeeded else 'FAILED'}")
        return "\n".join(lines)


def pose_action(
    pos_err: np.ndarray,
    yaw_err: float,
    *,
    gripper_open: bool,
    kp_pos: float,
    kp_yaw: float,
    max_pos_step: float,
    max_yaw_step: float,
    action_scale: float,
) -> np.ndarray:
    """Turn a pose error into the 7 numbers an IK-relative task wants.

    ``action_scale`` is divided back out because the environment multiplies the
    command by it before the differential IK controller sees it: the clip is then
    a limit on real commanded motion rather than on an arbitrary number. Only yaw
    is controlled; roll and pitch stay at zero.

    The gripper convention is Isaac Lab's: ``+1`` opens, ``-1`` closes.
    """
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip(kp_pos * pos_err / action_scale, -max_pos_step, max_pos_step)
    action[5] = np.clip(kp_yaw * yaw_err / action_scale, -max_yaw_step, max_yaw_step)
    action[6] = 1.0 if gripper_open else -1.0
    return action
