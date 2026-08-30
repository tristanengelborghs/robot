"""Turning stick deflections into commands, checked without a controller.

Everything here is arithmetic, which is the point of keeping it out of the
hardware wrapper: axis signs, dead zones and button edges are exactly the things
that are wrong in a way you only notice when a robot moves the wrong way, and
none of them need a DualSense to check.
"""

import pytest

from harness.dualsense import (
    GamepadMapping,
    apply_dead_zone,
    command_from_state,
    toggle_on_press,
)

CENTRED = [0.0, 0.0, 0.0, 0.0]


def axes(left_x=0.0, left_y=0.0, right_x=0.0, right_y=0.0) -> list[float]:
    """Axis list in the default mapping's order, for readable tests."""
    return [left_x, left_y, right_x, right_y]


def test_a_centred_pad_commands_no_motion():
    command = command_from_state(CENTRED, gripper_closed=False)
    assert command.dpos == (0.0, 0.0, 0.0)
    assert command.drot == (0.0, 0.0, 0.0)


def test_a_resting_stick_does_not_drift():
    # Sticks rarely rest at exactly zero. Without a dead zone the arm wanders
    # whenever nobody is touching it, which reads as a control bug.
    command = command_from_state(axes(left_x=0.05, left_y=-0.04), gripper_closed=False)
    assert command.dpos == (0.0, 0.0, 0.0)


def test_dead_zone_rescales_so_response_starts_from_zero():
    # Clipping alone would jump straight to dead_zone worth of motion at the
    # threshold; rescaling keeps it continuous.
    assert apply_dead_zone(0.1, 0.1) == pytest.approx(0.0)
    assert apply_dead_zone(0.55, 0.1) == pytest.approx(0.5)
    assert apply_dead_zone(1.0, 0.1) == pytest.approx(1.0)
    assert apply_dead_zone(-1.0, 0.1) == pytest.approx(-1.0)
    assert apply_dead_zone(-0.55, 0.1) == pytest.approx(-0.5)


def test_dead_zone_of_one_is_rejected():
    with pytest.raises(ValueError, match="below 1.0"):
        apply_dead_zone(0.5, 1.0)


def test_pushing_the_left_stick_up_moves_the_arm_forward():
    # A stick pushed up reports negative, and forward is +x.
    command = command_from_state(axes(left_y=-1.0), gripper_closed=False)
    assert command.dpos[0] > 0
    assert command.dpos[1] == pytest.approx(0.0)
    assert command.dpos[2] == pytest.approx(0.0)


def test_pushing_the_left_stick_down_moves_the_arm_back():
    assert command_from_state(axes(left_y=1.0), gripper_closed=False).dpos[0] < 0


def test_the_left_stick_sideways_moves_the_arm_sideways():
    right = command_from_state(axes(left_x=1.0), gripper_closed=False)
    left = command_from_state(axes(left_x=-1.0), gripper_closed=False)
    assert right.dpos[1] < 0 < left.dpos[1]
    assert right.dpos[0] == pytest.approx(0.0)


def test_the_right_stick_controls_height_and_yaw_only():
    up = command_from_state(axes(right_y=-1.0), gripper_closed=False)
    assert up.dpos[2] > 0
    assert up.dpos[0] == pytest.approx(0.0) and up.dpos[1] == pytest.approx(0.0)
    assert up.drot == (0.0, 0.0, 0.0)

    turned = command_from_state(axes(right_x=1.0), gripper_closed=False)
    assert turned.drot[2] != 0.0
    assert turned.drot[0] == 0.0 and turned.drot[1] == 0.0, "roll and pitch are left alone"
    assert turned.dpos == (0.0, 0.0, 0.0)


def test_full_deflection_respects_the_configured_sensitivity():
    mapping = GamepadMapping(pos_sensitivity=0.02, rot_sensitivity=0.04, dead_zone=0.0)
    assert command_from_state(axes(left_y=-1.0), False, mapping).dpos[0] == pytest.approx(0.02)
    assert command_from_state(axes(right_x=-1.0), False, mapping).drot[2] == pytest.approx(0.04)


def test_the_gripper_state_is_carried_not_the_press():
    # A command carries the latched state, so a dropped packet cannot leave the
    # gripper in the wrong position.
    assert command_from_state(CENTRED, gripper_closed=True).close_gripper is True
    assert command_from_state(CENTRED, gripper_closed=False).close_gripper is False


def test_the_gripper_toggles_once_per_press_not_once_per_poll():
    state = False
    # A button held for several polls is one press to a person.
    state = toggle_on_press(pressed=True, was_pressed=False, state=state)
    assert state is True
    for _ in range(30):
        state = toggle_on_press(pressed=True, was_pressed=True, state=state)
    assert state is True, "holding the button must not flap the gripper"

    state = toggle_on_press(pressed=False, was_pressed=True, state=state)
    assert state is True, "releasing does nothing"
    state = toggle_on_press(pressed=True, was_pressed=False, state=state)
    assert state is False, "the next press closes the toggle"


def test_a_pad_with_too_few_axes_is_rejected_clearly():
    with pytest.raises(IndexError, match="needs at least 4 axes"):
        command_from_state([0.0, 0.0], gripper_closed=False)


def test_a_custom_axis_assignment_is_honoured():
    # Axis numbers vary by driver and connection type, so they are configurable.
    mapping = GamepadMapping(left_x=2, left_y=3, right_x=0, right_y=1, dead_zone=0.0)
    command = command_from_state([0.0, 0.0, 0.0, -1.0], gripper_closed=False, mapping=mapping)
    assert command.dpos[0] > 0, "axis 3 is the forward axis under this mapping"
