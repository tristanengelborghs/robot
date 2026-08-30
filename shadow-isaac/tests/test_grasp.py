"""The scripted grasp, pinned against the asset's real joint limits.

LOWER/UPPER are the limit columns printed by `make bodies` on the box
(2026-08-30, Isaac Lab 3.0.0), in the simulator's own joint order. The point of
testing against them rather than invented numbers is the sign convention: the
Shadow Hand's ranges do not agree on which direction is "curl", and a grasp
written directly in action space gets the thumb backwards silently.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_D = pathlib.Path(__file__).resolve().parent.parent / "source/catching/catching/tasks/direct/catch"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"catch_{name}", _D / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gr = _load("grasp")
jt = _load("joints")

NAMES = [
    "robot0_WRJ1", "robot0_WRJ0", "robot0_FFJ3", "robot0_MFJ3", "robot0_RFJ3",
    "robot0_LFJ4", "robot0_THJ4", "robot0_FFJ2", "robot0_MFJ2", "robot0_RFJ2",
    "robot0_LFJ3", "robot0_THJ3", "robot0_FFJ1", "robot0_MFJ1", "robot0_RFJ1",
    "robot0_LFJ2", "robot0_THJ2", "robot0_FFJ0", "robot0_MFJ0", "robot0_RFJ0",
    "robot0_LFJ1", "robot0_THJ1", "robot0_LFJ0", "robot0_THJ0",
]
LOWER = np.array([-0.489, -0.698, -0.349, -0.349, -0.349, 0.0, -1.047, 0.0, 0.0, 0.0,
                  -0.349, 0.0, 0.0, 0.0, 0.0, 0.0, -0.209, 0.0, 0.0, 0.0, 0.0,
                  -0.524, 0.0, -1.571])
UPPER = np.array([0.140, 0.489, 0.349, 0.349, 0.349, 0.785, 1.047, 1.571, 1.571, 1.571,
                  0.349, 1.222, 1.571, 1.571, 1.571, 1.571, 0.209, 1.571, 1.571, 1.571,
                  1.571, 0.524, 1.571, 0.000])

ACT = jt.actuated_indices(NAMES)
A_NAMES = [NAMES[i] for i in ACT]
A_LOWER, A_UPPER = LOWER[ACT], UPPER[ACT]


def _act(pose):
    return gr.pose_to_action(pose, A_NAMES, A_LOWER, A_UPPER)


def test_every_actuated_joint_has_a_pose_entry():
    _act(gr.GRASP_POSE)
    _act(gr.OPEN_POSE)


def test_actions_stay_inside_the_action_space():
    for pose in (gr.GRASP_POSE, gr.OPEN_POSE):
        a = _act(pose)
        assert a.shape == (20,)
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_the_grasp_curls_the_fingers():
    a = _act(gr.GRASP_POSE)
    for j in ("robot0_FFJ1", "robot0_MFJ1", "robot0_RFJ1", "robot0_LFJ1"):
        assert a[A_NAMES.index(j)] > 0.5, f"{j} should be well flexed in a grasp"


def test_the_grasp_does_not_splay_the_fingers():
    """Spreading the fingers is how a ball falls through them."""
    a = _act(gr.GRASP_POSE)
    for j in ("robot0_FFJ3", "robot0_MFJ3", "robot0_RFJ3"):
        assert abs(a[A_NAMES.index(j)]) < 0.1


def test_the_thumbs_sign_is_not_inverted():
    """THJ0 runs [-1.571, 0]: flexion is the NEGATIVE end, so a grasp written
    in action space rather than radians sticks the thumb out."""
    i = A_NAMES.index("robot0_THJ0")
    assert A_LOWER[i] < 0 and A_UPPER[i] == 0.0, "the trap this test guards"
    grasp, open_ = _act(gr.GRASP_POSE)[i], _act(gr.OPEN_POSE)[i]
    assert grasp < open_, "the thumb must flex further when closing, not less"
    assert grasp < 0.0


def test_closing_moves_every_flexion_joint_toward_flexion():
    grasp, open_ = _act(gr.GRASP_POSE), _act(gr.OPEN_POSE)
    for j in ("robot0_FFJ1", "robot0_FFJ2", "robot0_MFJ1", "robot0_LFJ2"):
        i = A_NAMES.index(j)
        assert grasp[i] > open_[i]


def test_a_target_outside_a_joints_range_is_clipped_not_extrapolated():
    a = gr.pose_to_action({**gr.GRASP_POSE, "FFJ1": 99.0}, A_NAMES, A_LOWER, A_UPPER)
    assert a[A_NAMES.index("robot0_FFJ1")] == pytest.approx(1.0)


def test_mismatched_limit_arrays_are_refused():
    with pytest.raises(ValueError, match="must agree"):
        gr.pose_to_action(gr.GRASP_POSE, A_NAMES, A_LOWER[:5], A_UPPER)


# -- timing ----------------------------------------------------------------

def test_closure_ramps_rather_than_triggers():
    """Fingers move through a spring; a last-instant snap means the hand is
    still opening when the ball lands."""
    lead = 0.1
    assert gr.closure(0.20, lead) == 0.0
    assert gr.closure(0.05, lead) == pytest.approx(0.5)
    assert gr.closure(0.00, lead) == pytest.approx(1.0)
    assert gr.closure(-0.05, lead) == pytest.approx(1.0)


def test_no_incoming_ball_holds_the_hand_open():
    """None means 'not coming'. Read as zero it would mean 'arriving now' and
    the hand would clench at nothing."""
    assert gr.closure(None, 0.1) == 0.0


def test_action_interpolates_between_open_and_closed():
    open_a, closed_a = _act(gr.OPEN_POSE), _act(gr.GRASP_POSE)
    assert gr.action(None, open_a, closed_a, 0.1) == pytest.approx(open_a)
    assert gr.action(0.0, open_a, closed_a, 0.1) == pytest.approx(closed_a)
    mid = gr.action(0.05, open_a, closed_a, 0.1)
    assert mid == pytest.approx(0.5 * (open_a + closed_a))
