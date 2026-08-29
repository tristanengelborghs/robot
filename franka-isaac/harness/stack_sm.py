"""A scripted cube-stacking controller for ``Isaac-Stack-Cube-Franka-IK-Rel-v0``.

Component 2 of ../note.md needs recorded demonstrations before it can validate
the recording pipeline, and Component 5 needs a scripted controller before it
trusts a learned one. This is both: a state machine that stacks the three cubes
from ground-truth poses, so the record -> HDF5 -> replay path can be proven with
nobody sitting at a keyboard on a rented GPU.

It is deliberately pure NumPy. Everything hard about a scripted controller --
which state comes next, when a state has failed, how a pose error becomes an
action, the 90-degree yaw symmetry of a square cube -- is arithmetic, and gets
tested on the laptop against a fake plant (see tests/test_stack_sm.py). The GPU
box only has to supply real observations; see scripts/record_scripted.py.

The task's action space is 7 numbers, from stack_ik_rel_env_cfg.py:

* ``[0:3]`` end-effector position delta, in metres, in the environment frame
* ``[3:6]`` end-effector rotation delta as an axis-angle vector
* ``[6]``   gripper: ``+1`` open, ``-1`` close (``GripperState`` in Isaac Lab's
  own lift_cube_sm.py, and ``BinaryJointPositionAction`` closes on a negative)

Both pose halves are multiplied by ``DifferentialInverseKinematicsActionCfg
.scale = 0.5`` before they reach the differential IK controller, which is why
:attr:`StackConfig.action_scale` divides back out of every command: the numbers
in this file are the motion actually being asked for.

Only yaw is controlled. The Franka starts the episode in a top-down grasp pose
and the reset event randomises the cubes in yaw only, so roll and pitch are left
at zero rather than fought over -- and a rotation about the environment's z axis
and one about the gripper's own approach axis are then the same rotation, which
keeps this honest whichever frame the controller interprets the delta in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Cube 1 is the base of the stack, cube 2 goes on top of it, cube 3 on top of
# that: mdp.cubes_stacked requires cube_1.z < cube_2.z < cube_3.z. Indices here
# are zero-based into the cube arrays, so cube_1 is index 0.
STACK_PLAN: tuple[tuple[int, int], ...] = ((1, 0), (2, 1))

# A square cube looks the same every quarter turn, so a yaw error only ever
# needs to be driven into this half-window.
YAW_SYMMETRY = math.pi / 2


class State(Enum):
    """One step of a pick-and-place, plus the two terminal states.

    The order is the order they run in. Each state has a target pose, a
    completion test and a timeout; :class:`StackStateMachine` owns the
    transitions between them.
    """

    HOVER_PICK = "hover_pick"
    DESCEND_PICK = "descend_pick"
    CLOSE = "close"
    LIFT = "lift"
    HOVER_PLACE = "hover_place"
    DESCEND_PLACE = "descend_place"
    OPEN = "open"
    RETREAT = "retreat"
    SETTLE = "settle"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class StackConfig:
    """Geometry, gains and tolerances.

    The geometry constants are the environment's, not ours: ``cube_height``
    is the ``height_diff`` that ``mdp.cubes_stacked`` tests for, and
    ``action_scale`` is the config's arm-action scale.
    """

    # Geometry (metres).
    cube_height: float = 0.0468  # mdp.cubes_stacked height_diff
    hover_height: float = 0.10  # above the cube centre, before descending
    grasp_height: float = 0.005  # above the cube centre, when closing
    lift_height: float = 0.15  # above the cube centre, when carrying

    # Control.
    action_scale: float = 0.5  # DifferentialInverseKinematicsActionCfg.scale
    kp_pos: float = 0.8
    kp_yaw: float = 0.8
    max_pos_step: float = 0.10  # raw action units => 5 cm of commanded delta
    max_yaw_step: float = 0.20

    # Tolerances (metres / radians).
    xy_tol: float = 0.006
    pos_tol: float = 0.008
    yaw_tol: float = 0.06

    # Dwell times, in environment steps at 20 Hz.
    close_steps: int = 15
    open_steps: int = 12
    settle_steps: int = 20

    # Per-state patience. A state that cannot finish in this many steps has
    # failed, and the episode is abandoned rather than recorded.
    state_timeout: int = 250


@dataclass(frozen=True)
class Observation:
    """The slice of the environment the controller actually looks at.

    Every position is in the environment frame. That is what the task's
    ``eef_pos`` term reports (it subtracts ``env_origins``) and what the first
    three entries of each cube's block in the ``object`` term report, so the two
    are directly comparable -- which matters, because ``cube_positions`` in the
    same observation group is in the *world* frame and only coincides with these
    when there is a single environment at the origin.
    """

    eef_pos: np.ndarray  # (3,)
    eef_yaw: float
    cube_pos: np.ndarray  # (3, 3) -- cube_1, cube_2, cube_3
    cube_yaw: np.ndarray  # (3,)


@dataclass
class Transition:
    """One state change, for the log the deliverable asks for."""

    step: int
    frm: State
    to: State
    reason: str


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


class StackStateMachine:
    """Stacks three cubes, one state at a time.

    Usage is one :meth:`step` per environment step::

        sm = StackStateMachine()
        while not sm.finished:
            action = sm.step(observation)

    The machine is closed-loop on cube poses right up to the grasp, and then
    open-loop on the *recorded* grasp pose while carrying -- a carried cube sits
    inside the gripper, so its measured position stops being an independent
    signal about where the arm should go.
    """

    def __init__(self, cfg: StackConfig | None = None, plan: tuple[tuple[int, int], ...] = STACK_PLAN):
        self.cfg = cfg or StackConfig()
        self.plan = plan

        self.state = State.HOVER_PICK
        self.step_count = 0  # steps in the episode
        self.state_step = 0  # steps in the current state
        self.stage = 0  # index into self.plan
        self.transitions: list[Transition] = []

        # Yaw the gripper committed to at the grasp, and the height it lifted
        # from; both are frozen at grasp time because the cube's own pose is no
        # longer independent of the arm once it is held.
        self._grasp_yaw: float = 0.0
        self._carry_z: float = 0.0

    @property
    def finished(self) -> bool:
        return self.state in (State.DONE, State.FAILED)

    @property
    def failed(self) -> bool:
        return self.state is State.FAILED

    @property
    def pick_index(self) -> int:
        """Cube being moved in the current stage."""
        return self.plan[min(self.stage, len(self.plan) - 1)][0]

    @property
    def base_index(self) -> int:
        """Cube being stacked onto in the current stage."""
        return self.plan[min(self.stage, len(self.plan) - 1)][1]

    def step(self, obs: Observation) -> np.ndarray:
        """Advance one environment step and return a 7-dimensional action."""
        if self.finished:
            return self._action(np.zeros(3), 0.0, gripper_open=True)

        self.step_count += 1
        self.state_step += 1

        action = self._act(obs)

        if self.state_step >= self.cfg.state_timeout and self.state not in (State.DONE, State.FAILED):
            self._go(State.FAILED, f"timed out after {self.state_step} steps")

        return action

    def _act(self, obs: Observation) -> np.ndarray:
        cfg = self.cfg
        pick = obs.cube_pos[self.pick_index]
        base = obs.cube_pos[self.base_index]

        if self.state is State.HOVER_PICK:
            target = pick + np.array([0.0, 0.0, cfg.hover_height])
            yaw_err = yaw_error(obs.cube_yaw[self.pick_index], obs.eef_yaw)
            err = target - obs.eef_pos
            if _xy_norm(err) < cfg.xy_tol and abs(yaw_err) < cfg.yaw_tol:
                self._go(State.DESCEND_PICK, "aligned above cube")
            return self._action(err, yaw_err, gripper_open=True)

        if self.state is State.DESCEND_PICK:
            target = pick + np.array([0.0, 0.0, cfg.grasp_height])
            yaw_err = yaw_error(obs.cube_yaw[self.pick_index], obs.eef_yaw)
            err = target - obs.eef_pos
            if np.linalg.norm(err) < cfg.pos_tol:
                self._grasp_yaw = obs.eef_yaw
                self._carry_z = pick[2] + cfg.lift_height
                self._go(State.CLOSE, "at grasp pose")
            return self._action(err, yaw_err, gripper_open=True)

        if self.state is State.CLOSE:
            # Hold still while the fingers travel; commanding motion here is how
            # a grasp turns into a swipe.
            if self.state_step >= cfg.close_steps:
                self._go(State.LIFT, "gripper closed")
            return self._action(np.zeros(3), 0.0, gripper_open=False)

        if self.state is State.LIFT:
            err = np.array([0.0, 0.0, self._carry_z - obs.eef_pos[2]])
            if abs(err[2]) < cfg.pos_tol:
                self._go(State.HOVER_PLACE, "cube lifted clear")
            return self._action(err, 0.0, gripper_open=False)

        if self.state is State.HOVER_PLACE:
            target = base + np.array([0.0, 0.0, cfg.hover_height + cfg.cube_height])
            err = target - obs.eef_pos
            if _xy_norm(err) < cfg.xy_tol:
                self._go(State.DESCEND_PLACE, "above the base cube")
            return self._action(err, 0.0, gripper_open=False)

        if self.state is State.DESCEND_PLACE:
            # The carried cube's centre must end up one cube-height above the
            # base cube's centre, and the gripper holds it at grasp_height.
            target = base + np.array([0.0, 0.0, cfg.cube_height + cfg.grasp_height])
            err = target - obs.eef_pos
            if np.linalg.norm(err) < cfg.pos_tol:
                self._go(State.OPEN, "cube seated on the stack")
            return self._action(err, 0.0, gripper_open=False)

        if self.state is State.OPEN:
            if self.state_step >= cfg.open_steps:
                self._go(State.RETREAT, "gripper released")
            return self._action(np.zeros(3), 0.0, gripper_open=True)

        if self.state is State.RETREAT:
            # Clear of the stack before anything else moves: the success test
            # also requires the fingers be fully open, and dragging a finger
            # across the top cube is the classic way to lose a finished stack.
            err = np.array([0.0, 0.0, base[2] + cfg.lift_height - obs.eef_pos[2]])
            if abs(err[2]) < cfg.pos_tol:
                self.stage += 1
                if self.stage >= len(self.plan):
                    self._go(State.SETTLE, "all cubes placed")
                else:
                    self._go(State.HOVER_PICK, f"stage {self.stage} of {len(self.plan)}")
            return self._action(err, 0.0, gripper_open=True)

        if self.state is State.SETTLE:
            # Stand still with the gripper open and let the success termination
            # observe a stack that is not still being touched.
            if self.state_step >= cfg.settle_steps:
                self._go(State.DONE, "settled")
            return self._action(np.zeros(3), 0.0, gripper_open=True)

        raise AssertionError(f"unhandled state {self.state}")

    def _action(self, pos_err: np.ndarray, yaw_err: float, *, gripper_open: bool) -> np.ndarray:
        """Turn a pose error into the 7 numbers the environment wants."""
        cfg = self.cfg
        action = np.zeros(7, dtype=np.float32)
        # Divide the config's action scale back out, so the clip below is a
        # limit on real commanded motion rather than on an arbitrary number.
        action[:3] = np.clip(cfg.kp_pos * pos_err / cfg.action_scale, -cfg.max_pos_step, cfg.max_pos_step)
        action[5] = np.clip(cfg.kp_yaw * yaw_err / cfg.action_scale, -cfg.max_yaw_step, cfg.max_yaw_step)
        action[6] = 1.0 if gripper_open else -1.0
        return action

    def _go(self, to: State, reason: str) -> None:
        self.transitions.append(Transition(self.step_count, self.state, to, reason))
        self.state = to
        self.state_step = 0


def _xy_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec[:2]))


@dataclass
class EpisodeLog:
    """What a single scripted episode did, for the state-transition log."""

    transitions: list[Transition] = field(default_factory=list)
    steps: int = 0
    succeeded: bool = False

    def format(self) -> str:
        lines = [f"{'step':>6}  {'from':<14} -> {'to':<14} reason"]
        lines += [f"{t.step:>6}  {t.frm.value:<14} -> {t.to.value:<14} {t.reason}" for t in self.transitions]
        lines.append(f"{self.steps} steps, {'success' if self.succeeded else 'FAILED'}")
        return "\n".join(lines)
