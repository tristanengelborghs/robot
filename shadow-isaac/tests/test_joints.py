"""Actuated-joint selection, pinned against the real asset.

JOINT_NAMES below is not invented: it is the output of `make bodies` against
Isaac Lab's Shadow Hand on the box, in the simulator's own order. Anything that
changes the asset changes this list, and this test is where that shows up.
"""

import importlib.util
import pathlib
import sys

import pytest

_PATH = (pathlib.Path(__file__).resolve().parent.parent
         / "source/catching/catching/tasks/direct/catch/joints.py")
_spec = importlib.util.spec_from_file_location("catch_joints", _PATH)
jt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = jt
_spec.loader.exec_module(jt)

# Read off the asset, 2026-08-30, Isaac Lab 3.0.0.
JOINT_NAMES = [
    "robot0_WRJ1", "robot0_WRJ0", "robot0_FFJ3", "robot0_MFJ3", "robot0_RFJ3",
    "robot0_LFJ4", "robot0_THJ4", "robot0_FFJ2", "robot0_MFJ2", "robot0_RFJ2",
    "robot0_LFJ3", "robot0_THJ3", "robot0_FFJ1", "robot0_MFJ1", "robot0_RFJ1",
    "robot0_LFJ2", "robot0_THJ2", "robot0_FFJ0", "robot0_MFJ0", "robot0_RFJ0",
    "robot0_LFJ1", "robot0_THJ1", "robot0_LFJ0", "robot0_THJ0",
]

# The four joints the simulator reports with stiffness, damping and effort 0.
COUPLED = [17, 18, 19, 22]


def test_the_asset_has_twenty_four_joints():
    assert len(JOINT_NAMES) == 24


def test_twenty_joints_are_actuated():
    assert jt.actuated_indices(JOINT_NAMES) == [
        i for i in range(24) if i not in COUPLED]


def test_the_tendon_coupled_joints_are_excluded():
    idx = jt.actuated_indices(JOINT_NAMES)
    for i in COUPLED:
        assert i not in idx, f"{JOINT_NAMES[i]} has effort limit 0 and cannot be commanded"


def test_the_naive_range_twenty_is_wrong_in_both_directions():
    """Why this module exists rather than `range(20)`."""
    idx = set(jt.actuated_indices(JOINT_NAMES))
    naive = set(range(20))
    commands_the_undrivable = naive - idx
    misses_the_drivable = idx - naive
    assert commands_the_undrivable == {17, 18, 19}, "FFJ0/MFJ0/RFJ0 cannot be driven"
    assert misses_the_drivable == {20, 21, 23}, "LFJ1, THJ1 and THJ0 would never move"


def test_the_thumbs_distal_joint_is_actuated():
    """THJ0 is independently actuated on a Shadow Hand; the fingers' are not."""
    assert not jt.is_coupled("robot0_THJ0")
    assert JOINT_NAMES.index("robot0_THJ0") in jt.actuated_indices(JOINT_NAMES)


def test_a_wrong_count_is_loud():
    with pytest.raises(RuntimeError, match="expected 20"):
        jt.actuated_indices(JOINT_NAMES[:10])
