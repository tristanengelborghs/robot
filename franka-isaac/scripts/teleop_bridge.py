"""Drive the simulator from a controller plugged into this laptop.

The browser stream forwards keyboard and mouse to the remote Isaac Sim but not a
gamepad, so this takes the controller the other way round: read it here, turn
sticks into end-effector commands here, and send six numbers and a boolean over
an ssh forward to a device waiting inside the container.

    make teleop TELEOP_DEVICE=network     # on the box, in one shell
    make bridge                           # on the laptop, in another

Three modes, because a remote control link has three separate things that can be
wrong and it is worth being able to test them apart:

``--show-axes``
    Print what the pad reports and stop. Answers "is the controller working and
    which axis is which", with no network and no simulator involved.

``--demo``
    Send a slow scripted circle instead of reading a pad. Answers "does the link
    reach the arm", with no controller involved -- which is how the whole path
    was verified before a DualSense was ever plugged in.

default
    Read the pad and stream it.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time

from harness import remote
from harness.dualsense import DualSenseReader, GamepadMapping, command_from_state, toggle_on_press
from harness.teleop_link import DEFAULT_PORT, STOP, CommandSender, Se3Command

SEND_HZ = 60


def open_tunnel(port: int) -> subprocess.Popen:
    """Forward the laptop's port to the same port on the box.

    ``ControlPath=none`` for the reason documented in :func:`harness.remote.tunnel`:
    Brev's ssh config multiplexes, and against an existing master ``ssh -N``
    hands over the forward and exits, leaving a tunnel nothing can close.
    """
    command = [
        "ssh", "-N",
        "-o", "ControlPath=none",
        "-o", "ExitOnForwardFailure=yes",
        "-L", f"{port}:127.0.0.1:{port}",
        remote.INSTANCE,
    ]
    print(f"$ {' '.join(command)}", flush=True)
    tunnel = subprocess.Popen(command)
    time.sleep(2.0)  # give the forward a moment before anything tries to use it
    if tunnel.poll() is not None:
        raise RuntimeError("the ssh tunnel exited immediately; is the box running?")
    return tunnel


def show_axes(reader: DualSenseReader) -> int:
    """Print live axis values so a wrong axis assignment takes seconds to spot."""
    print(f"\n{reader.name}: {reader.axis_count} axes\n")
    print("Move one stick at a time and watch which number changes.")
    print("Ctrl-C to stop.\n")

    try:
        while True:
            axes, buttons = reader.poll()
            rendered = "  ".join(f"{i}:{value:+.2f}" for i, value in enumerate(axes))
            pressed = [i for i, down in enumerate(buttons) if down]
            print(f"\r{rendered}   buttons={pressed}    ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n")
        return 0


def demo_command(elapsed: float) -> Se3Command:
    """A slow circle in the horizontal plane, for testing the link alone."""
    return Se3Command(
        dpos=(0.004 * math.cos(elapsed), 0.004 * math.sin(elapsed), 0.0),
        drot=(0.0, 0.0, 0.0),
        close_gripper=False,
    )


def stream_demo(sender: CommandSender, seconds: float) -> int:
    """Send scripted motion, so the link can be tested with no controller."""
    print(f"sending a scripted circle for {seconds:.0f}s -- the arm should move", flush=True)
    started = time.monotonic()
    sent = 0

    while time.monotonic() - started < seconds:
        sender.send(demo_command(time.monotonic() - started))
        sent += 1
        time.sleep(1.0 / SEND_HZ)

    sender.send(STOP)
    print(f"sent {sent} commands, then a stop", flush=True)
    return 0


def stream_pad(sender: CommandSender, reader: DualSenseReader, mapping: GamepadMapping) -> int:
    """Read the controller and stream it until interrupted."""
    print(f"\nstreaming {reader.name} -- ctrl-C to stop\n", flush=True)
    print("  left stick: x/y     right stick: z and yaw     cross: gripper\n", flush=True)

    gripper_closed = False
    was_pressed = False
    sent = 0

    try:
        while True:
            axes, buttons = reader.poll()
            pressed = buttons[mapping.gripper_button] if len(buttons) > mapping.gripper_button else False
            gripper_closed = toggle_on_press(pressed, was_pressed, gripper_closed)
            was_pressed = pressed

            command = command_from_state(axes, gripper_closed, mapping)
            sender.send(command)
            sent += 1

            if sent % SEND_HZ == 0:
                grip = "closed" if gripper_closed else "open"
                moving = any(abs(v) > 1e-9 for v in command.dpos + command.drot)
                print(f"\r{sent} sent   gripper {grip}   {'moving ' if moving else 'still  '}", end="", flush=True)

            time.sleep(1.0 / SEND_HZ)
    except KeyboardInterrupt:
        # Always leave the arm stopped, never mid-motion.
        sender.send(STOP)
        print(f"\n\nstopped after {sent} commands; sent a final stop", flush=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--show-axes", action="store_true", help="print what the pad reports, then exit")
    parser.add_argument("--demo", action="store_true", help="send scripted motion instead of reading a pad")
    parser.add_argument("--demo-seconds", type=float, default=20.0)
    parser.add_argument("--no-tunnel", action="store_true", help="assume an ssh forward is already up")
    parser.add_argument("--joystick", type=int, default=0, help="which pad, if several are attached")
    parser.add_argument("--dead-zone", type=float, default=GamepadMapping.dead_zone)
    parser.add_argument("--pos-sensitivity", type=float, default=GamepadMapping.pos_sensitivity)
    parser.add_argument("--rot-sensitivity", type=float, default=GamepadMapping.rot_sensitivity)
    args = parser.parse_args()

    mapping = GamepadMapping(
        dead_zone=args.dead_zone,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )

    # Checking the controller needs no network at all, so it happens first and
    # on its own.
    if args.show_axes:
        return show_axes(DualSenseReader(args.joystick))

    reader = None if args.demo else DualSenseReader(args.joystick)

    tunnel = None if args.no_tunnel else open_tunnel(args.port)
    try:
        with CommandSender(port=args.port) as sender:
            if args.demo:
                return stream_demo(sender, args.demo_seconds)
            return stream_pad(sender, reader, mapping)
    except ConnectionRefusedError:
        print(
            f"\nnothing is listening on port {args.port} inside the container.\n"
            "Start the simulator side first:\n"
            "    make teleop TELEOP_DEVICE=network\n",
            file=sys.stderr,
        )
        return 1
    finally:
        if reader is not None:
            reader.close()
        if tunnel is not None:
            tunnel.terminate()


if __name__ == "__main__":
    sys.exit(main())
