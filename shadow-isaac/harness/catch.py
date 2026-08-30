"""Catch-task logic: phases, reward, termination, curriculum.

Pure functions over plain dataclasses. No isaaclab import, no simulator, no
GPU -- so all of it is tested on the laptop, and the Isaac Lab environment on
the box is left holding nothing but the wiring. This is the same split
`franka-isaac` uses for `stack_sm.py`, and for the same reason: the hard logic
is where the bugs are, and the GPU box bills by the hour.

Three decisions here are load-bearing, and each one is a failure mode that
would otherwise show up as "the policy just does not learn".

Reward through the flight. Between release and catch the ball is ballistic and
    the hand cannot touch it. A reward that only pays out on the catch gives
    that entire stretch no gradient. So the flight is scored on *predicted*
    intercept error -- the distance between the palm and the point where the
    ball will cross palm height, which `ballistics.predict_intercept` gives in
    closed form. The policy cannot move the ball, but it can be in the right
    place when it arrives, and that is exactly what it is paid for.

Holding is not rewarded until the ball has flown. The obvious reward -- pay
    per step while the ball is held, penalise the drop -- has a degenerate
    optimum on the launch stages: never throw. Sitting still collects the hold
    reward forever at zero risk, the return curve rises, and the run looks
    healthy while the task is not being attempted. So hold reward is gated on
    `flights_completed`, and a stage that expects a launch terminates the
    episode if none happens before a deadline. `test_catch.py` pins both by
    asserting that the do-nothing policy scores strictly below a catch.

The control rate is validated, not assumed. A 5 cm toss lasts 0.2 s; at 20 Hz
    that is four control steps and the ball crosses its own diameter between
    decisions. No policy aims a catch through that. `CatchConfig.validate`
    refuses the configuration up front rather than letting it surface weeks
    later as a mystery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from . import ballistics as bal


class Stage(Enum):
    """Curriculum stages, in the order they must be learned."""

    DROP = "drop"      # ball released above the palm at rest; catch it
    TOSS = "toss"      # ball spawned with upward velocity; track apex, catch
    POP = "pop"        # the hand launches the ball itself, then catches it


#: Stable integer codes for Phase and for termination reasons. The vectorised
#: torch implementation that runs on the GPU cannot carry Python enums through
#: a tensor, and `tests/test_parity.py` compares the two implementations by
#: these codes. Append only -- reordering silently rewrites what a logged run
#: meant.
PHASE_CODES = ("settle", "ready", "flight", "secured", "done", "failed")
REASON_CODES = (None, "dropped", "out_of_bounds", "excess_force", "caught",
                "never_launched", "timeout")


def phase_code(phase: "Phase") -> int:
    return PHASE_CODES.index(phase.value)


def reason_code(reason: Optional[str]) -> int:
    return REASON_CODES.index(reason)


class Phase(Enum):
    SETTLE = "settle"      # post-reset, un-rewarded, let the scene come to rest
    READY = "ready"        # ball held or not yet released
    FLIGHT = "flight"      # airborne, no finger contact
    SECURED = "secured"    # caught after a flight, being held
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class BallState:
    pos: np.ndarray            # world xyz
    vel: np.ndarray            # world xyz
    contacts: int = 0          # fingertips currently touching the ball
    max_force: float = 0.0     # largest per-fingertip normal force, newtons


@dataclass(frozen=True)
class HandState:
    palm_pos: np.ndarray       # world xyz of the palm centre
    action: np.ndarray         # last action, in [-1, 1]


@dataclass
class CatchConfig:
    stage: Stage = Stage.DROP

    ctrl_dt: float = 0.01           # 100 Hz. See validate(); 20 Hz cannot work.
    episode_len: int = 400          # control steps (4 s at 100 Hz)
    settle_steps: int = 20

    ball_radius: float = 0.03
    spawn_height: float = 0.05      # above the palm; the curriculum widens this
    max_spawn_height: float = 0.30

    # -- catch geometry ---------------------------------------------------
    catch_radius: float = 0.08      # lateral distance from palm centre
    secure_speed: float = 0.25      # relative speed below which a hold counts
    min_contacts: int = 1           # fingertips needed to call it held
    hold_steps: int = 30            # held this long = success

    # -- termination ------------------------------------------------------
    drop_below: float = 0.10        # ball this far under the palm = dropped
    drop_lateral: float = 0.25
    max_force: float = 40.0         # newtons on one fingertip
    launch_deadline: int = 150      # control steps to get airborne (TOSS/POP)

    # -- reward -----------------------------------------------------------
    track_scale: float = 2.0        # dense intercept-error term during flight
    secure_bonus: float = 50.0
    hold_scale: float = 1.0
    apex_scale: float = 10.0        # POP only: pay for getting it airborne
    drop_penalty: float = 20.0
    force_penalty: float = 0.5
    action_rate_penalty: float = 0.01
    ctrl_penalty: float = 0.001

    # -- validation -------------------------------------------------------
    min_flight_steps: int = 20
    max_travel_fraction: float = 0.25   # of a ball DIAMETER per control step

    def validate(self) -> None:
        """Refuse a configuration the task cannot be learned in.

        Both checks are about the same thing from two directions: the policy
        needs enough decisions during the flight, and each decision must not
        let the ball jump past the hand.
        """
        v = bal.launch_speed_for(self.spawn_height)
        # A drop reaches the palm in half the time a toss does: it only falls.
        flight = (np.sqrt(2.0 * self.spawn_height / bal.G)
                  if self.stage is Stage.DROP else bal.flight_time(v))
        steps = flight / self.ctrl_dt
        if steps < self.min_flight_steps:
            raise ValueError(
                f"{steps:.1f} control steps of flight at {1/self.ctrl_dt:.0f} Hz for a "
                f"{self.spawn_height*100:.0f} cm toss; need >= {self.min_flight_steps}. "
                f"Raise the control rate or raise spawn_height.")
        travel = bal.travel_per_control_step(v, self.ctrl_dt)
        limit = self.max_travel_fraction * 2.0 * self.ball_radius
        if travel > limit:
            raise ValueError(
                f"ball travels {travel*100:.1f} cm per control step, more than "
                f"{self.max_travel_fraction:.0%} of its {2*self.ball_radius*100:.0f} cm "
                f"diameter ({limit*100:.1f} cm). Raise the control rate.")


@dataclass
class Episode:
    """Everything the reward needs that a single frame does not carry."""

    steps: int = 0
    flights_completed: int = 0
    held_steps: int = 0
    phase: Phase = Phase.SETTLE
    peak_height: float = 0.0


def relative_speed(ball: BallState, palm_vel: Optional[np.ndarray] = None) -> float:
    v = ball.vel if palm_vel is None else ball.vel - np.asarray(palm_vel)
    return float(np.linalg.norm(v))


def is_secured(ball: BallState, hand: HandState, cfg: CatchConfig) -> bool:
    """Held, not merely touched.

    Contact alone is not a catch -- a ball grazing a fingertip on its way past
    reports contact. It has to be slow, close, above the palm, and touched by
    more than one finger.
    """
    lateral = float(np.linalg.norm(ball.pos[:2] - hand.palm_pos[:2]))
    return (ball.contacts >= cfg.min_contacts
            and lateral <= cfg.catch_radius
            and ball.pos[2] > hand.palm_pos[2]
            and relative_speed(ball) <= cfg.secure_speed)


def phase_of(ball: BallState, hand: HandState, ep: Episode, cfg: CatchConfig) -> Phase:
    if ep.steps < cfg.settle_steps:
        return Phase.SETTLE
    if is_secured(ball, hand, cfg):
        return Phase.SECURED if ep.flights_completed > 0 else Phase.READY
    if ball.contacts == 0 and ball.pos[2] > hand.palm_pos[2]:
        return Phase.FLIGHT
    return Phase.READY


def reward_terms(ball: BallState, hand: HandState, ep: Episode, cfg: CatchConfig,
                 prev_action: np.ndarray) -> Dict[str, float]:
    """Named reward terms. Named, so ablating or logging one is free."""
    terms: Dict[str, float] = {}
    phase = ep.phase

    if phase is Phase.SETTLE:
        return {"settle": 0.0}

    if phase is Phase.FLIGHT:
        hit = bal.predict_intercept(ball.pos, ball.vel, float(hand.palm_pos[2]))
        if hit is not None:
            point, _ = hit
            err = float(np.linalg.norm(point[:2] - hand.palm_pos[:2]))
            terms["track"] = cfg.track_scale / (err + 0.05)
        if cfg.stage is Stage.POP:
            gained = max(0.0, ep.peak_height - float(hand.palm_pos[2]))
            terms["apex"] = cfg.apex_scale * min(gained / cfg.spawn_height, 1.0)

    if phase is Phase.SECURED:
        # Gated on a completed flight: never-throw earns nothing here.
        terms["hold"] = cfg.hold_scale
        if ep.held_steps == 1:
            terms["secure"] = cfg.secure_bonus

    if ball.max_force > cfg.max_force:
        terms["force"] = -cfg.force_penalty * (ball.max_force - cfg.max_force)

    terms["action_rate"] = -cfg.action_rate_penalty * float(
        np.sum((hand.action - prev_action) ** 2))
    terms["effort"] = -cfg.ctrl_penalty * float(np.sum(hand.action ** 2))
    return terms


def total(terms: Dict[str, float]) -> float:
    return float(sum(terms.values()))


def terminate(ball: BallState, hand: HandState, ep: Episode,
              cfg: CatchConfig) -> Tuple[bool, Optional[str]]:
    """(done, reason). The reason is the point -- a success rate with no
    failure breakdown says nothing about what to fix."""
    if ball.pos[2] < hand.palm_pos[2] - cfg.drop_below:
        return True, "dropped"
    if float(np.linalg.norm(ball.pos[:2] - hand.palm_pos[:2])) > cfg.drop_lateral:
        return True, "out_of_bounds"
    if ball.max_force > cfg.max_force:
        return True, "excess_force"
    if ep.held_steps >= cfg.hold_steps and ep.flights_completed > 0:
        return True, "caught"
    if (cfg.stage is not Stage.DROP and ep.flights_completed == 0
            and ep.steps >= cfg.launch_deadline):
        return True, "never_launched"
    if ep.steps >= cfg.episode_len:
        return True, "timeout"
    return False, None


@dataclass
class Curriculum:
    """Widen the toss on measured success, not on a step schedule.

    A step schedule traverses the curriculum at whatever pace was tuned on the
    machine it was tuned on; a smoke run on a laptop and a long run on the box
    then see different tasks under the same config. `shadow-mujoco` learned
    this one already.
    """

    cfg: CatchConfig
    promote_at: float = 0.6
    step: float = 0.02
    window: int = 20
    _recent: list = field(default_factory=list)

    def record(self, caught: bool) -> None:
        self._recent.append(bool(caught))
        if len(self._recent) > self.window:
            self._recent.pop(0)

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._recent)) if self._recent else 0.0

    def update(self) -> bool:
        """Widen if the window clears the bar. Returns True if it moved."""
        if len(self._recent) < self.window or self.success_rate < self.promote_at:
            return False
        new = min(self.cfg.spawn_height + self.step, self.cfg.max_spawn_height)
        if new == self.cfg.spawn_height:
            return False
        self.cfg.spawn_height = new
        self._recent.clear()
        return True
