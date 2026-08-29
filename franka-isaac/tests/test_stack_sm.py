"""The scripted controller, run against a plant made of arithmetic.

A state machine is mostly a claim about ordering: that it grasps before it
lifts, that it opens the gripper before it retreats, that it gives up instead of
pushing forever. None of that needs contact dynamics to check, and checking it
here is the difference between finding a transposed axis in a unit test and
finding it in an hour of rented L4.

The plant below is deliberately generous -- perfect tracking with a first-order
lag, a cube that snaps to a closed gripper. It cannot tell us the controller
will succeed in Isaac Lab. It can tell us the controller is internally
consistent: that its targets are reachable by the actions it emits, that its
completion tests are satisfiable by its own targets, and that a stack built by
following it satisfies the environment's real success predicate.
"""

import math

import numpy as np
import pytest

from harness.stack_sm import (
    Observation,
    StackConfig,
    StackStateMachine,
    State,
    wrap_to_pi,
    yaw_error,
    yaw_from_quat,
)

# The environment's own numbers, from mdp.cubes_stacked and the reset event.
TABLE_Z = 0.0203
XY_THRESHOLD = 0.04
HEIGHT_THRESHOLD = 0.005
HEIGHT_DIFF = 0.0468


class FakePlant:
    """A point gripper, three cubes, and no physics worth the name."""

    def __init__(self, cube_pos, cube_yaw, eef_pos=(0.4, 0.0, 0.4), eef_yaw=0.0, tracking=0.6, grasp_radius=0.03):
        self.cube_pos = np.array(cube_pos, dtype=float)
        self.cube_yaw = np.array(cube_yaw, dtype=float)
        self.eef_pos = np.array(eef_pos, dtype=float)
        self.eef_yaw = float(eef_yaw)
        self.tracking = tracking
        self.grasp_radius = grasp_radius
        self.cfg = StackConfig()
        self.closed = False
        self.held: int | None = None

    def observe(self) -> Observation:
        return Observation(
            eef_pos=self.eef_pos.copy(),
            eef_yaw=self.eef_yaw,
            cube_pos=self.cube_pos.copy(),
            cube_yaw=self.cube_yaw.copy(),
        )

    def step(self, action) -> Observation:
        action = np.asarray(action, dtype=float)
        self.eef_pos += action[:3] * self.cfg.action_scale * self.tracking
        self.eef_yaw = wrap_to_pi(self.eef_yaw + action[5] * self.cfg.action_scale * self.tracking)

        closing = action[6] < 0
        if closing and not self.closed:
            self._attach()
        elif not closing and self.closed:
            self._release()
        self.closed = closing

        if self.held is not None:
            # A held cube rides at the offset the controller grasped it at.
            self.cube_pos[self.held] = self.eef_pos - np.array([0.0, 0.0, self.cfg.grasp_height])

        return self.observe()

    def _attach(self) -> None:
        grasp_point = self.eef_pos - np.array([0.0, 0.0, self.cfg.grasp_height])
        distances = np.linalg.norm(self.cube_pos - grasp_point, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < self.grasp_radius:
            self.held = nearest

    def _release(self) -> None:
        if self.held is None:
            return
        dropped = self.held
        self.held = None
        # Settle onto whatever is under it: the tallest cube it overlaps, or
        # the table.
        resting = TABLE_Z
        for other in range(len(self.cube_pos)):
            if other == dropped:
                continue
            if np.linalg.norm(self.cube_pos[other][:2] - self.cube_pos[dropped][:2]) < XY_THRESHOLD:
                resting = max(resting, self.cube_pos[other][2] + HEIGHT_DIFF)
        self.cube_pos[dropped][2] = resting


def cubes_stacked(cube_pos) -> bool:
    """The environment's success predicate, in NumPy.

    Mirrors mdp.cubes_stacked for the parallel-gripper case: cube 2 sits on
    cube 1, cube 3 sits on cube 2. The gripper-open half of the real predicate
    is not modelled here; the SETTLE state is what covers it, and
    test_final_release_leaves_the_gripper_open pins that separately.
    """
    for upper, lower in ((1, 0), (2, 1)):
        diff = cube_pos[lower] - cube_pos[upper]
        if np.linalg.norm(diff[:2]) >= XY_THRESHOLD:
            return False
        if abs(np.linalg.norm(diff[2:])) - HEIGHT_DIFF >= HEIGHT_THRESHOLD:
            return False
        if diff[2] >= 0.0:  # the lower cube must actually be lower
            return False
    return True


def run(plant: FakePlant, max_steps: int = 4000) -> StackStateMachine:
    sm = StackStateMachine()
    obs = plant.observe()
    for _ in range(max_steps):
        if sm.finished:
            break
        obs = plant.step(sm.step(obs))
    return sm


def scattered_scene(seed: int) -> FakePlant:
    """Three cubes on the table, laid out the way the reset event lays them.

    Matches randomize_cube_positions in stack_ik_rel_env_cfg.py: x in
    [0.4, 0.6], y in [-0.1, 0.1], yaw in [-1, 1], and no two cubes closer than
    10 cm.
    """
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < 3:
        candidate = np.array([rng.uniform(0.4, 0.6), rng.uniform(-0.10, 0.10), TABLE_Z])
        if all(np.linalg.norm(candidate[:2] - p[:2]) >= 0.10 for p in positions):
            positions.append(candidate)
    return FakePlant(np.array(positions), rng.uniform(-1.0, 1.0, size=3))


def test_yaw_error_is_zero_for_a_quarter_turn():
    # A square cube is indistinguishable every 90 degrees, so a quarter turn is
    # not an error to be corrected.
    for turns in (-2, -1, 0, 1, 2):
        assert yaw_error(turns * math.pi / 2, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_yaw_error_takes_the_short_way_round():
    assert yaw_error(math.radians(80), 0.0) == pytest.approx(math.radians(-10), abs=1e-9)
    assert yaw_error(math.radians(-80), 0.0) == pytest.approx(math.radians(10), abs=1e-9)
    for target in np.linspace(-math.pi, math.pi, 41):
        assert abs(yaw_error(target, 0.3)) <= math.pi / 4 + 1e-9


def test_yaw_from_quat_reads_wxyz():
    for angle in (0.0, 0.4, -1.2, 2.5):
        quat = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        assert yaw_from_quat(quat) == pytest.approx(wrap_to_pi(angle), abs=1e-9)


def test_actions_stay_inside_the_configured_limits():
    cfg = StackConfig()
    sm = StackStateMachine(cfg)
    # An absurd error must not produce an absurd command.
    obs = Observation(
        eef_pos=np.array([10.0, -10.0, 10.0]),
        eef_yaw=0.0,
        cube_pos=np.zeros((3, 3)),
        cube_yaw=np.array([3.0, 0.0, 0.0]),
    )
    action = sm.step(obs)
    assert action.shape == (7,)
    assert np.all(np.abs(action[:3]) <= cfg.max_pos_step + 1e-9)
    assert abs(action[5]) <= cfg.max_yaw_step + 1e-9
    assert action[3] == 0.0 and action[4] == 0.0  # roll and pitch are left alone


def test_gripper_is_open_while_approaching_and_closed_while_carrying():
    plant = scattered_scene(0)
    sm = StackStateMachine()
    obs = plant.observe()
    seen: dict[State, set[float]] = {}
    for _ in range(4000):
        if sm.finished:
            break
        state = sm.state
        action = sm.step(obs)
        seen.setdefault(state, set()).add(float(action[6]))
        obs = plant.step(action)

    # +1 opens, -1 closes; see the module docstring of harness.stack_sm.
    assert seen[State.HOVER_PICK] == {1.0}
    assert seen[State.DESCEND_PICK] == {1.0}
    assert seen[State.CLOSE] == {-1.0}
    assert seen[State.LIFT] == {-1.0}
    assert seen[State.DESCEND_PLACE] == {-1.0}
    assert seen[State.OPEN] == {1.0}
    assert seen[State.RETREAT] == {1.0}


def test_states_run_in_the_intended_order():
    sm = run(scattered_scene(1))
    assert sm.state is State.DONE

    order = [t.frm for t in sm.transitions] + [sm.state]
    per_stage = [
        State.HOVER_PICK,
        State.DESCEND_PICK,
        State.CLOSE,
        State.LIFT,
        State.HOVER_PLACE,
        State.DESCEND_PLACE,
        State.OPEN,
        State.RETREAT,
    ]
    assert order == per_stage * 2 + [State.SETTLE, State.DONE]


@pytest.mark.parametrize("seed", range(8))
def test_scripted_stack_satisfies_the_environments_success_test(seed):
    plant = scattered_scene(seed)
    sm = run(plant)
    assert sm.state is State.DONE, sm.transitions[-1] if sm.transitions else "no transitions"
    assert cubes_stacked(plant.cube_pos)


def test_final_release_leaves_the_gripper_open():
    # mdp.cubes_stacked also requires both fingers back at gripper_open_val, so
    # the machine must not finish mid-grasp.
    plant = scattered_scene(2)
    sm = StackStateMachine()
    obs = plant.observe()
    last = None
    for _ in range(4000):
        if sm.finished:
            break
        last = sm.step(obs)
        obs = plant.step(last)
    assert last is not None and last[6] > 0
    assert plant.held is None


def test_a_stuck_arm_fails_instead_of_pushing_forever():
    # A plant that ignores commands: every state must run out of patience, and
    # the episode must end rather than burn the instance's clock.
    plant = scattered_scene(3)
    plant.tracking = 0.0
    sm = run(plant, max_steps=StackConfig().state_timeout * 3)
    assert sm.state is State.FAILED
    assert sm.step_count <= StackConfig().state_timeout + 1
    assert "timed out" in sm.transitions[-1].reason


def test_a_finished_machine_holds_still():
    sm = run(scattered_scene(4))
    action = sm.step(Observation(np.zeros(3), 0.0, np.zeros((3, 3)), np.zeros(3)))
    assert np.all(action[:6] == 0.0)
    assert action[6] > 0
