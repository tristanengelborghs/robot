"""The pick-and-lift controller, against a plant made of arithmetic.

Same approach as tests/test_stack_sm.py and for the same reason: the ordering
claims in a state machine -- grasp before lift, hold before declaring success,
give up rather than push forever -- need no contact dynamics to check, and
checking them here is the difference between finding a sign error in a unit test
and finding it in an hour of rented L4.

The plant is generous: perfect tracking with a first-order lag and a cube that
snaps to a closed gripper. It cannot show that the controller will succeed in
Isaac Lab. It shows the controller is internally consistent -- that its targets
are reachable by the actions it emits, and that following it ends with the cube
genuinely off the table.
"""

import numpy as np
import pytest

from harness.control import wrap_to_pi
from harness.lift_sm import LiftConfig, LiftObservation, LiftStateMachine, State

# Measured from the real task by scripts/probe_lift.py.
TABLE_Z = 0.021
LIFT_THRESHOLD = 0.04  # the task's own object_is_lifted height


class FakePlant:
    """A point gripper and one cube, with no physics worth the name."""

    def __init__(self, object_pos, object_yaw=0.0, eef_pos=(0.463, 0.0, 0.385), tracking=0.5):
        self.object_pos = np.array(object_pos, dtype=float)
        self.object_yaw = float(object_yaw)
        self.eef_pos = np.array(eef_pos, dtype=float)
        self.eef_yaw = 0.0
        self.cfg = LiftConfig()
        self.tracking = tracking
        self.closed = False
        self.holding = False

    def observe(self) -> LiftObservation:
        return LiftObservation(
            eef_pos=self.eef_pos.copy(),
            eef_yaw=self.eef_yaw,
            object_pos=self.object_pos.copy(),
            object_yaw=self.object_yaw,
        )

    def step(self, action) -> LiftObservation:
        action = np.asarray(action, dtype=float)
        self.eef_pos += action[:3] * self.cfg.action_scale * self.tracking
        self.eef_yaw = wrap_to_pi(self.eef_yaw + action[5] * self.cfg.action_scale * self.tracking)

        closing = action[6] < 0
        if closing and not self.closed:
            grasp_point = self.eef_pos - np.array([0.0, 0.0, self.cfg.grasp_height])
            self.holding = bool(np.linalg.norm(self.object_pos - grasp_point) < 0.03)
        elif not closing and self.closed:
            self.holding = False
            self.object_pos[2] = TABLE_Z  # dropped, back to the table
        self.closed = closing

        if self.holding:
            self.object_pos = self.eef_pos - np.array([0.0, 0.0, self.cfg.grasp_height])

        return self.observe()


def scene(seed: int) -> FakePlant:
    """A cube where the task's reset event puts one, at its measured rest height."""
    rng = np.random.default_rng(seed)
    return FakePlant(
        object_pos=[rng.uniform(0.45, 0.60), rng.uniform(-0.15, 0.15), TABLE_Z],
        object_yaw=rng.uniform(-1.0, 1.0),
    )


def run(plant: FakePlant, max_steps: int = 3000) -> LiftStateMachine:
    controller = LiftStateMachine()
    obs = plant.observe()
    for _ in range(max_steps):
        if controller.finished:
            break
        obs = plant.step(controller.step(obs))
    return controller


@pytest.mark.parametrize("seed", range(8))
def test_the_cube_ends_up_off_the_table(seed):
    plant = scene(seed)
    controller = run(plant)

    assert controller.state is State.DONE, controller.outcome().format()
    assert plant.holding, "the cube must still be in the gripper at the end"
    assert plant.object_pos[2] > LIFT_THRESHOLD, "the task's own lift threshold must be cleared"
    assert plant.object_pos[2] > TABLE_Z + 0.1, "and by a margin, not just clipped"


def test_the_states_run_in_the_intended_order():
    controller = run(scene(1))
    order = [t.frm for t in controller.transitions] + [controller.state]
    assert order == [State.HOVER, State.DESCEND, State.CLOSE, State.LIFT, State.HOLD, State.DONE]


def test_the_gripper_opens_and_closes_at_the_right_states():
    plant = scene(0)
    controller = LiftStateMachine()
    obs = plant.observe()
    seen: dict[State, set[float]] = {}

    for _ in range(3000):
        if controller.finished:
            break
        state = controller.state
        action = controller.step(obs)
        seen.setdefault(state, set()).add(float(action[6]))
        obs = plant.step(action)

    assert seen[State.HOVER] == {1.0}, "approach with the gripper open"
    assert seen[State.DESCEND] == {1.0}
    assert seen[State.CLOSE] == {-1.0}
    assert seen[State.LIFT] == {-1.0}
    assert seen[State.HOLD] == {-1.0}, "never let go of a cube being held up"


def test_the_gripper_stays_shut_after_finishing():
    # Unlike stacking, success here is holding the cube. A controller that opens
    # its gripper on the last step drops the thing it just picked up.
    controller = run(scene(2))
    action = controller.step(LiftObservation(np.zeros(3), 0.0, np.zeros(3), 0.0))
    assert action[6] < 0


def test_the_arm_holds_still_while_the_fingers_close():
    # Moving during the close drags the cube instead of gripping it.
    plant = scene(3)
    controller = LiftStateMachine()
    obs = plant.observe()

    for _ in range(3000):
        if controller.finished or controller.state is State.CLOSE:
            break
        obs = plant.step(controller.step(obs))

    assert controller.state is State.CLOSE
    action = controller.step(obs)
    assert action[:6] == pytest.approx(np.zeros(6))


def test_success_requires_holding_and_not_one_lucky_frame():
    cfg = LiftConfig()
    controller = run(scene(4))
    hold = next(t for t in controller.transitions if t.to is State.DONE)
    lift = next(t for t in controller.transitions if t.to is State.HOLD)
    assert hold.step - lift.step >= cfg.hold_steps


def test_a_stuck_arm_gives_up_instead_of_pushing_forever():
    plant = scene(5)
    plant.tracking = 0.0  # the plant ignores every command
    controller = run(plant, max_steps=LiftConfig().state_timeout * 3)

    assert controller.state is State.FAILED
    assert "timed out" in controller.transitions[-1].reason


def test_actions_stay_inside_the_configured_limits():
    cfg = LiftConfig()
    controller = LiftStateMachine(cfg)
    absurd = LiftObservation(
        eef_pos=np.array([10.0, -10.0, 10.0]),
        eef_yaw=0.0,
        object_pos=np.zeros(3),
        object_yaw=2.0,
    )
    action = controller.step(absurd)

    assert action.shape == (7,)
    assert np.all(np.abs(action[:3]) <= cfg.max_pos_step + 1e-9)
    assert abs(action[5]) <= cfg.max_yaw_step + 1e-9
    assert action[3] == 0.0 and action[4] == 0.0, "roll and pitch are left alone"


def test_the_outcome_records_the_peak_height():
    plant = scene(6)
    controller = run(plant)
    outcome = controller.outcome()

    assert outcome.succeeded
    assert outcome.peak_object_height > LIFT_THRESHOLD
    assert outcome.steps > 0
    assert "success" in outcome.format()


def test_every_state_has_a_handler():
    controller = LiftStateMachine()
    assert set(controller._handlers) == set(State)
