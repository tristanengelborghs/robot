"""Turning a gamepad's sticks into end-effector commands, on the laptop.

The arithmetic is separated from the hardware on purpose. Deciding what a stick
deflection means -- the dead zone, the sign of each axis, how a button press
becomes a latched gripper state -- is the part that is easy to get subtly wrong
and easy to test, so it lives in :func:`command_from_state` and is tested
without a controller. Reading a physical pad is a thin wrapper around pygame at
the bottom of the file.

The mapping mirrors Isaac Lab's own ``Se3Gamepad`` so that muscle memory carries
across if the browser path is ever fixed:

===========================  =========================
Left stick                   end-effector x and y
Right stick, up and down     end-effector z
Right stick, left and right  yaw
Cross (button 0)             toggle the gripper
===========================  =========================

Axis *numbers* are configurable because they are not a fact about DualSense, they
are a fact about whatever SDL/pygame reports on a given machine, and it varies
with connection type and driver. ``scripts/teleop_bridge.py --show-axes`` prints
what the pad is actually sending so a wrong guess takes seconds to correct
rather than being debugged through a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.teleop_link import Se3Command


@dataclass(frozen=True)
class GamepadMapping:
    """Which axis is which, and how hard each one pushes.

    Signs are part of the mapping rather than baked into the arithmetic. On a
    typical pad, pushing a stick *up* reports a *negative* value, and the robot
    frame wants forward to be positive, so most of these are inversions rather
    than an accident.
    """

    # Axis indices as pygame reports them.
    left_x: int = 0
    left_y: int = 1
    right_x: int = 2
    right_y: int = 3

    # Signs, applied after the dead zone.
    forward_sign: float = -1.0  # stick up (negative) means +x, forward
    lateral_sign: float = -1.0  # stick right (positive) means -y
    vertical_sign: float = -1.0  # stick up (negative) means +z
    yaw_sign: float = -1.0

    # Below this deflection a stick counts as centred. Sticks rarely rest at
    # exactly zero, and without this the arm drifts whenever nobody is touching
    # it -- which looks like a control bug rather than worn hardware.
    dead_zone: float = 0.08

    # Metres and radians per environment step at full deflection. Deliberately
    # small: this is a delta applied every control step at 20 Hz, so 0.02 is
    # already 40 cm per second.
    pos_sensitivity: float = 0.02
    rot_sensitivity: float = 0.04

    # The button that toggles the gripper. 0 is Cross on a standard mapping.
    gripper_button: int = 0


def apply_dead_zone(value: float, dead_zone: float) -> float:
    """Zero out small deflections, and rescale what is left.

    Rescaling matters: clipping alone would make the stick jump from nothing to
    ``dead_zone`` worth of motion as it crosses the threshold. This keeps the
    response continuous, starting from zero at the edge of the dead zone.
    """
    if dead_zone >= 1.0:
        raise ValueError(f"dead_zone must be below 1.0, got {dead_zone}")

    magnitude = abs(value)
    if magnitude <= dead_zone:
        return 0.0

    rescaled = (magnitude - dead_zone) / (1.0 - dead_zone)
    return rescaled if value > 0 else -rescaled


def command_from_state(
    axes: list[float],
    gripper_closed: bool,
    mapping: GamepadMapping | None = None,
) -> Se3Command:
    """Build one command from the current stick positions.

    Args:
        axes: Axis values in ``[-1, 1]``, as pygame reports them.
        gripper_closed: The *latched* gripper state, which the caller owns --
            see :func:`toggle_on_press`. A command carries the state rather than
            the press, so a dropped packet cannot leave the gripper wrong.
        mapping: Axis assignment and sensitivities.

    Raises:
        IndexError: if ``axes`` is too short for the mapping, naming what was
            expected. A pad reporting fewer axes than assumed would otherwise
            silently drive the wrong direction.
    """
    mapping = mapping or GamepadMapping()
    needed = max(mapping.left_x, mapping.left_y, mapping.right_x, mapping.right_y)
    if len(axes) <= needed:
        raise IndexError(f"the mapping needs at least {needed + 1} axes, the pad reported {len(axes)}")

    def deflection(index: int) -> float:
        return apply_dead_zone(axes[index], mapping.dead_zone)

    forward = deflection(mapping.left_y) * mapping.forward_sign
    lateral = deflection(mapping.left_x) * mapping.lateral_sign
    vertical = deflection(mapping.right_y) * mapping.vertical_sign
    yaw = deflection(mapping.right_x) * mapping.yaw_sign

    return Se3Command(
        dpos=(
            forward * mapping.pos_sensitivity,
            lateral * mapping.pos_sensitivity,
            vertical * mapping.pos_sensitivity,
        ),
        drot=(0.0, 0.0, yaw * mapping.rot_sensitivity),
        close_gripper=gripper_closed,
    )


def toggle_on_press(pressed: bool, was_pressed: bool, state: bool) -> bool:
    """Flip ``state`` on the rising edge of a button.

    Edge rather than level: a button held down for half a second at 60 Hz is one
    press to a person and thirty to a polling loop.
    """
    return not state if (pressed and not was_pressed) else state


class DualSenseReader:
    """Reads a physical pad through pygame. The only part that needs hardware.

    pygame is used rather than raw HID because a DualSense enumerates as an
    ordinary joystick on macOS over both USB and Bluetooth, and parsing HID
    reports by hand would add a second thing to get wrong.
    """

    def __init__(self, joystick_index: int = 0):
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("reading a gamepad needs pygame: run `make install`") from exc

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "no gamepad found. Pair the controller over Bluetooth or plug it in, "
                "and check it appears in macOS before trying again."
            )

        self._joystick = pygame.joystick.Joystick(joystick_index)
        self._joystick.init()

    @property
    def name(self) -> str:
        return self._joystick.get_name()

    @property
    def axis_count(self) -> int:
        return self._joystick.get_numaxes()

    def poll(self) -> tuple[list[float], list[bool]]:
        """Current axes and buttons. Pumping events is what refreshes them."""
        self._pygame.event.pump()
        axes = [self._joystick.get_axis(i) for i in range(self._joystick.get_numaxes())]
        buttons = [bool(self._joystick.get_button(i)) for i in range(self._joystick.get_numbuttons())]
        return axes, buttons

    def close(self) -> None:
        self._pygame.joystick.quit()
        self._pygame.quit()
