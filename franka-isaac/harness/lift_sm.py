"""A scripted pick-and-lift controller for ``Isaac-Lift-Cube-Franka-IK-Rel-v0``.

The stacking controller with the placing half removed: approach, descend, close,
lift, hold. That is the whole task, and it is worth having as scripted code for
the same two reasons as before -- it produces demonstrations unattended, and if a
controller holding ground-truth poses cannot do the task then no policy trained
on the same observations is going to either.

Two things differ from :mod:`harness.stack_sm` and neither is cosmetic.

**The control rate is 50 Hz, not 20.** The lift task runs ``decimation=2`` at
``sim.dt=0.01``; the stacking task runs ``decimation=5``. Every gain, step limit
and dwell time here is in units of one control step, so reusing the stacking
numbers would move the arm two and a half times faster than intended.

**There is no end-effector pose in the observations.** The lift task's policy
group carries joint positions, the object position and the commanded goal, but
not where the hand is. The scene's ``ee_frame`` transformer still knows, so the
driver reads it from ``env.scene`` and passes it in; see scripts/record_lift.py.

The goal command is deliberately ignored. The task as shipped is "lift the block
to a commanded pose"; the task we want first is "pick the block up", so the
demonstrations lift and hold, and it is the success condition in the recorder --
ours, since the task ships without one -- that decides when they are done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from harness.control import Transition, horizontal_distance, pose_action, yaw_error


class State(Enum):
    """The states of one pick-and-lift, in the order they run."""

    HOVER = "hover"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    HOLD = "hold"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class LiftConfig:
    """Geometry, gains and tolerances, all in units of one 50 Hz control step.

    The geometry is measured rather than assumed: ``scripts/probe_lift.py``
    reports the cube settling at ``z = 0.021`` after spawning at ``0.055``, which
    is where ``grasp_height`` and the lift target come from.
    """

    # Geometry (metres), relative to the object's centre.
    hover_height: float = 0.10
    grasp_height: float = 0.005
    lift_height: float = 0.18  # well clear of the task's own 0.04 threshold

    # Control. The environment multiplies the commanded delta by action_scale
    # before the differential IK controller sees it.
    action_scale: float = 0.5
    kp_pos: float = 0.6
    kp_yaw: float = 0.6
    # At 50 Hz, 0.04 raw is 0.02 m of commanded delta per step: 1 m/s flat out,
    # and proportional control means it only approaches that far from the target.
    max_pos_step: float = 0.04
    max_yaw_step: float = 0.10

    # Tolerances (metres / radians).
    xy_tol: float = 0.006
    pos_tol: float = 0.008
    yaw_tol: float = 0.06

    # Dwell times, in control steps at 50 Hz.
    close_steps: int = 30  # 0.6s for the fingers to travel
    hold_steps: int = 40  # 0.8s of holding, so success is not a single frame

    # Per-state patience. Generous next to the 250-step episode because the
    # recorder disables the time limit; a state that cannot finish in this many
    # steps has failed and the episode is abandoned rather than recorded.
    state_timeout: int = 200


@dataclass(frozen=True)
class LiftObservation:
    """What the controller looks at, all in the environment frame.

    The driver is responsible for subtracting ``env_origins`` from both, so a
    single environment at the origin and one offset in a grid behave the same.
    """

    eef_pos: np.ndarray  # (3,)
    eef_yaw: float
    object_pos: np.ndarray  # (3,)
    object_yaw: float


@dataclass
class LiftOutcome:
    """Where an episode got to, for the transition log and for diagnosis."""

    transitions: list[Transition] = field(default_factory=list)
    steps: int = 0
    succeeded: bool = False
    peak_object_height: float = 0.0

    def format(self) -> str:
        lines = [f"{'step':>6}  {'from':<9} -> {'to':<9} reason"]
        lines += [f"{t.step:>6}  {t.frm.value:<9} -> {t.to.value:<9} {t.reason}" for t in self.transitions]
        lines.append(
            f"{self.steps} steps, peak object height {self.peak_object_height:.3f} m, "
            f"{'success' if self.succeeded else 'FAILED'}"
        )
        return "\n".join(lines)


class LiftStateMachine:
    """Approaches a cube, grasps it, lifts it and holds.

    One :meth:`step` per control step::

        controller = LiftStateMachine()
        while not controller.finished:
            action = controller.step(observation)
    """

    def __init__(self, cfg: LiftConfig | None = None):
        self.cfg = cfg or LiftConfig()

        self.state = State.HOVER
        self.step_count = 0
        self.state_step = 0
        self.transitions: list[Transition] = []
        self.peak_object_height = 0.0

        # Frozen at the grasp: once the cube is held its measured position stops
        # being an independent signal about where the arm should go, so the lift
        # target cannot be derived from it afterwards.
        self._carry_height = 0.0

        self._handlers = {
            State.HOVER: self._hover,
            State.DESCEND: self._descend,
            State.CLOSE: self._close,
            State.LIFT: self._lift,
            State.HOLD: self._hold,
            State.DONE: self._terminal,
            State.FAILED: self._terminal,
        }

    @property
    def finished(self) -> bool:
        return self.state in (State.DONE, State.FAILED)

    @property
    def succeeded(self) -> bool:
        return self.state is State.DONE

    def step(self, obs: LiftObservation) -> np.ndarray:
        """Advance one control step and return a 7-dimensional action."""
        self.peak_object_height = max(self.peak_object_height, float(obs.object_pos[2]))

        if self.finished:
            return self._still(gripper_open=False)

        self.step_count += 1
        self.state_step += 1

        action = self._handlers[self.state](obs)

        if not self.finished and self.state_step >= self.cfg.state_timeout:
            self._go(State.FAILED, f"timed out after {self.state_step} steps")

        return action

    def outcome(self) -> LiftOutcome:
        """A record of what this episode did."""
        return LiftOutcome(
            transitions=list(self.transitions),
            steps=self.step_count,
            succeeded=self.succeeded,
            peak_object_height=self.peak_object_height,
        )

    # -- one handler per state, in the order they run --------------------------

    def _hover(self, obs: LiftObservation) -> np.ndarray:
        """Line up above the cube, gripper open, yaw matched to it."""
        target = obs.object_pos + np.array([0.0, 0.0, self.cfg.hover_height])
        error = target - obs.eef_pos
        yaw_err = yaw_error(obs.object_yaw, obs.eef_yaw)

        if horizontal_distance(error) < self.cfg.xy_tol and abs(yaw_err) < self.cfg.yaw_tol:
            self._go(State.DESCEND, "aligned above the cube")

        return self._action(error, yaw_err, gripper_open=True)

    def _descend(self, obs: LiftObservation) -> np.ndarray:
        """Come down onto the cube, still open, still tracking its yaw."""
        target = obs.object_pos + np.array([0.0, 0.0, self.cfg.grasp_height])
        error = target - obs.eef_pos
        yaw_err = yaw_error(obs.object_yaw, obs.eef_yaw)

        if np.linalg.norm(error) < self.cfg.pos_tol:
            self._carry_height = float(obs.object_pos[2]) + self.cfg.lift_height
            self._go(State.CLOSE, "at the grasp pose")

        return self._action(error, yaw_err, gripper_open=True)

    def _close(self, _obs: LiftObservation) -> np.ndarray:
        """Hold still while the fingers travel.

        Commanding motion here is how a grasp becomes a swipe: the fingers take
        several control steps to close, and anything moving the hand during them
        drags the cube instead of gripping it.
        """
        if self.state_step >= self.cfg.close_steps:
            self._go(State.LIFT, "gripper closed")

        return self._still(gripper_open=False)

    def _lift(self, obs: LiftObservation) -> np.ndarray:
        """Raise the cube clear of the table."""
        error = np.array([0.0, 0.0, self._carry_height - obs.eef_pos[2]])

        if abs(error[2]) < self.cfg.pos_tol:
            self._go(State.HOLD, "cube lifted")

        return self._action(error, 0.0, gripper_open=False)

    def _hold(self, _obs: LiftObservation) -> np.ndarray:
        """Hold the cube up, so success is a held state and not one lucky frame.

        The recorder's success condition wants several consecutive steps above
        the height. A demonstration that clips the threshold on the way through
        and drops the cube is not a demonstration of picking something up.
        """
        if self.state_step >= self.cfg.hold_steps:
            self._go(State.DONE, "held")

        return self._still(gripper_open=False)

    def _terminal(self, _obs: LiftObservation) -> np.ndarray:
        """DONE and FAILED both hold still, gripper closed on whatever is there."""
        return self._still(gripper_open=False)

    # -- transitions and command construction ---------------------------------

    def _action(self, pos_err: np.ndarray, yaw_err: float, *, gripper_open: bool) -> np.ndarray:
        cfg = self.cfg
        return pose_action(
            pos_err,
            yaw_err,
            gripper_open=gripper_open,
            kp_pos=cfg.kp_pos,
            kp_yaw=cfg.kp_yaw,
            max_pos_step=cfg.max_pos_step,
            max_yaw_step=cfg.max_yaw_step,
            action_scale=cfg.action_scale,
        )

    def _still(self, *, gripper_open: bool) -> np.ndarray:
        return self._action(np.zeros(3), 0.0, gripper_open=gripper_open)

    def _go(self, to: State, reason: str) -> None:
        self.transitions.append(Transition(self.step_count, self.state, to, reason))
        self.state = to
        self.state_step = 0
