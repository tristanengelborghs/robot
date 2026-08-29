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


# Layout of one cube's block in the task's `object` observation term, from
# mdp.object_obs: a position followed by a quaternion, the position already
# relative to the environment origin.
POSE_LENGTH = 7
CUBE_COUNT = 3


def observation_from_arrays(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    object_term: np.ndarray,
) -> Observation:
    """Build an :class:`Observation` from the task's raw observation arrays.

    This lives here rather than in scripts/record_scripted.py because it is
    arithmetic, and arithmetic is testable without a simulator. Slicing a 39-wide
    observation term into three poses is exactly the kind of off-by-seven that
    should not be discovered on a rented GPU.

    Only the first three pose blocks of ``object_term`` are read; the rest of
    that term is relative vectors between the cubes and the gripper, which the
    controller derives for itself from these.

    Args:
        eef_pos: End-effector position, ``(3,)``.
        eef_quat: End-effector orientation as a ``wxyz`` quaternion, ``(4,)``.
        object_term: The task's ``object`` observation term, at least
            ``CUBE_COUNT * POSE_LENGTH`` long.

    Raises:
        ValueError: if ``object_term`` is too short to hold three poses, which
            means it came from a task laid out differently to this one.
    """
    flat = np.asarray(object_term).reshape(-1)
    needed = CUBE_COUNT * POSE_LENGTH
    if flat.size < needed:
        raise ValueError(
            f"the `object` observation term is {flat.size} wide, expected at least {needed} "
            f"({CUBE_COUNT} cubes x {POSE_LENGTH} numbers of pose) -- is this the stacking task?"
        )

    poses = flat[:needed].reshape(CUBE_COUNT, POSE_LENGTH)
    return Observation(
        eef_pos=np.asarray(eef_pos, dtype=float).reshape(3),
        eef_yaw=yaw_from_quat(np.asarray(eef_quat).reshape(4)),
        cube_pos=poses[:, :3].astype(float).copy(),
        cube_yaw=np.array([yaw_from_quat(pose[3:7]) for pose in poses]),
    )


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

        # Frozen at grasp time, because a cube's pose stops being independent
        # of the arm the moment the arm is holding it: the yaw the gripper
        # committed to, and the height to carry it at.
        self._grasp_yaw: float = 0.0
        self._carry_height: float = 0.0

        # Every state maps to exactly one handler. Building the table here,
        # rather than dispatching on a chain of comparisons, means an
        # unhandled state is a KeyError naming the state rather than a silent
        # fall-through -- and tests/test_stack_sm.py checks the table is total.
        self._handlers = {
            State.HOVER_PICK: self._hover_pick,
            State.DESCEND_PICK: self._descend_pick,
            State.CLOSE: self._close,
            State.LIFT: self._lift,
            State.HOVER_PLACE: self._hover_place,
            State.DESCEND_PLACE: self._descend_place,
            State.OPEN: self._open,
            State.RETREAT: self._retreat,
            State.SETTLE: self._settle,
            State.DONE: self._terminal,
            State.FAILED: self._terminal,
        }

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
        """Advance one environment step and return a 7-dimensional action.

        The handler for the current state decides what to command and whether to
        move on. Timeouts are checked here rather than inside each handler, so
        that no state can forget to have one.
        """
        if self.finished:
            return self._hold_still()

        self.step_count += 1
        self.state_step += 1

        action = self._handlers[self.state](obs)

        if not self.finished and self.state_step >= self.cfg.state_timeout:
            self._go(State.FAILED, f"timed out after {self.state_step} steps")

        return action

    # -- one handler per state, in the order they run --------------------------
    #
    # Each takes the current observation, returns the action to command, and
    # transitions when its own completion test passes. They are deliberately
    # separate methods rather than branches of one function: a state's target
    # pose, its completion test and its reason for moving on belong together and
    # nowhere else.

    def _hover_pick(self, obs: Observation) -> np.ndarray:
        """Line up above the cube to be picked, gripper open, yaw matched."""
        cube = obs.cube_pos[self.pick_index]
        target = cube + np.array([0.0, 0.0, self.cfg.hover_height])
        position_error = target - obs.eef_pos
        yaw_err = yaw_error(obs.cube_yaw[self.pick_index], obs.eef_yaw)

        if horizontal_distance(position_error) < self.cfg.xy_tol and abs(yaw_err) < self.cfg.yaw_tol:
            self._go(State.DESCEND_PICK, "aligned above cube")

        return self._action(position_error, yaw_err, gripper_open=True)

    def _descend_pick(self, obs: Observation) -> np.ndarray:
        """Come down onto the cube, still open, still tracking its yaw."""
        cube = obs.cube_pos[self.pick_index]
        target = cube + np.array([0.0, 0.0, self.cfg.grasp_height])
        position_error = target - obs.eef_pos
        yaw_err = yaw_error(obs.cube_yaw[self.pick_index], obs.eef_yaw)

        if np.linalg.norm(position_error) < self.cfg.pos_tol:
            # Freeze what the arm will need while carrying. Once the cube is
            # held, its measured pose stops being an independent signal about
            # where the arm should go, so these cannot be read again later.
            self._grasp_yaw = obs.eef_yaw
            self._carry_height = cube[2] + self.cfg.lift_height
            self._go(State.CLOSE, "at grasp pose")

        return self._action(position_error, yaw_err, gripper_open=True)

    def _close(self, _obs: Observation) -> np.ndarray:
        """Hold still while the fingers travel.

        Commanding motion here is how a grasp turns into a swipe: the fingers
        take a few control steps to close, and anything that moves the hand
        during them drags the cube instead of gripping it.
        """
        if self.state_step >= self.cfg.close_steps:
            self._go(State.LIFT, "gripper closed")

        return self._hold_still(gripper_open=False)

    def _lift(self, obs: Observation) -> np.ndarray:
        """Raise the held cube clear of the table and of the other cubes."""
        position_error = np.array([0.0, 0.0, self._carry_height - obs.eef_pos[2]])

        if abs(position_error[2]) < self.cfg.pos_tol:
            self._go(State.HOVER_PLACE, "cube lifted clear")

        return self._action(position_error, 0.0, gripper_open=False)

    def _hover_place(self, obs: Observation) -> np.ndarray:
        """Carry the cube over the one it is going on top of."""
        base = obs.cube_pos[self.base_index]
        target = base + np.array([0.0, 0.0, self.cfg.hover_height + self.cfg.cube_height])
        position_error = target - obs.eef_pos

        if horizontal_distance(position_error) < self.cfg.xy_tol:
            self._go(State.DESCEND_PLACE, "above the base cube")

        return self._action(position_error, 0.0, gripper_open=False)

    def _descend_place(self, obs: Observation) -> np.ndarray:
        """Lower until the carried cube is seated on the stack.

        The carried cube must end up one cube-height above the base cube's
        centre, and the gripper is holding it ``grasp_height`` above its centre.
        """
        base = obs.cube_pos[self.base_index]
        target = base + np.array([0.0, 0.0, self.cfg.cube_height + self.cfg.grasp_height])
        position_error = target - obs.eef_pos

        if np.linalg.norm(position_error) < self.cfg.pos_tol:
            self._go(State.OPEN, "cube seated on the stack")

        return self._action(position_error, 0.0, gripper_open=False)

    def _open(self, _obs: Observation) -> np.ndarray:
        """Let go, and give the fingers time to actually be open."""
        if self.state_step >= self.cfg.open_steps:
            self._go(State.RETREAT, "gripper released")

        return self._hold_still(gripper_open=True)

    def _retreat(self, obs: Observation) -> np.ndarray:
        """Climb clear of the stack before anything else moves.

        Dragging a finger across the cube just placed is the classic way to lose
        a finished stack, and the success test also requires both fingers back
        at their open position.
        """
        base = obs.cube_pos[self.base_index]
        position_error = np.array([0.0, 0.0, base[2] + self.cfg.lift_height - obs.eef_pos[2]])

        if abs(position_error[2]) < self.cfg.pos_tol:
            self._finish_stage()

        return self._action(position_error, 0.0, gripper_open=True)

    def _settle(self, _obs: Observation) -> np.ndarray:
        """Stand still with the gripper open and let the stack be observed.

        The environment's success termination wants a stack that is not still
        being touched, held for several consecutive steps.
        """
        if self.state_step >= self.cfg.settle_steps:
            self._go(State.DONE, "settled")

        return self._hold_still(gripper_open=True)

    def _terminal(self, _obs: Observation) -> np.ndarray:
        """DONE and FAILED both do nothing, safely."""
        return self._hold_still(gripper_open=True)

    # -- transitions and command construction ---------------------------------

    def _finish_stage(self) -> None:
        """One cube is placed. Move to the next, or settle if that was the last."""
        self.stage += 1
        if self.stage >= len(self.plan):
            self._go(State.SETTLE, "all cubes placed")
        else:
            self._go(State.HOVER_PICK, f"stage {self.stage} of {len(self.plan)}")

    def _hold_still(self, *, gripper_open: bool = True) -> np.ndarray:
        """Command no motion at all, with the gripper in the given state."""
        return self._action(np.zeros(3), 0.0, gripper_open=gripper_open)

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


def horizontal_distance(vec: np.ndarray) -> float:
    """Length of ``vec`` in the xy plane, ignoring height."""
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
