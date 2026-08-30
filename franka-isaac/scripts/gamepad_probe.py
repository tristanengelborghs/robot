"""Report what the simulator actually sees from a gamepad.

A gamepad that does not work has three possible failure points and they look
identical from the browser: the controller is not exposed to the page at all,
the stream is not forwarding its input, or Isaac Lab's device is bound to a
gamepad slot that was empty when the device was built. Guessing between them
costs a session each time, because only a person with a controller can test.

So this asks the simulator directly. It launches a bare Kit app with the same
livestream flags a teleoperation session uses, then reports:

* which gamepad slots have a device in them, at startup and as they change
* every gamepad event that arrives, with its input name and value

Run it, connect the controller in the browser, and move the sticks. What the log
says narrows the fault to one place:

* no connection event, no input events -- the browser is not forwarding, or the
  page never saw the controller (press a button on it with the page focused)
* a connection event but no input events -- forwarding of axes is the problem
* input events here, but the arm does not move in a real session -- the fault is
  in Isaac Lab's device binding, not in the network path

Usage, inside the container::

    ./isaaclab.sh -p scripts/gamepad_probe.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Report gamepad activity as the simulator sees it.")
parser.add_argument("--seconds", type=int, default=300, help="How long to listen before exiting.")
parser.add_argument("--slots", type=int, default=4, help="How many gamepad slots to watch.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Same flags a teleoperation session uses: streaming, with a Kit visualizer so
# there is something to look at and something to focus.
args_cli.livestream = 2
args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs with a live simulator."""

import time  # noqa: E402

import carb  # noqa: E402
import omni.appwindow  # noqa: E402


def describe_slots(app_window, input_interface, slots: int) -> dict[int, str]:
    """Report what each gamepad slot holds, without hiding an unnamed device.

    ``get_gamepad`` may hand back a handle for a slot that holds nothing, and an
    empty name is not the same as an empty slot. Both are reported distinctly,
    because "no gamepad" and "a gamepad Kit cannot name" call for different
    fixes.
    """
    found = {}
    for slot in range(slots):
        gamepad = app_window.get_gamepad(slot)
        if gamepad is None:
            continue
        name = input_interface.get_gamepad_name(gamepad)
        found[slot] = name if name else "<handle exists, no name>"
    return found


def main() -> None:
    app_window = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()

    print("\n[probe] listening. The viewer will show an EMPTY BLACK viewport:")
    print("[probe] this app loads no scene, so there is no robot to see. That is")
    print("[probe] correct -- what matters is the log below, not the picture.")
    print("[probe] Open the viewer, click the viewport, then press a button on")
    print("[probe] the controller and move the sticks.\n", flush=True)

    seen_events = {"count": 0}
    subscriptions: dict[int, object] = {}

    def on_event(event, *_args, label: str) -> bool:
        seen_events["count"] += 1
        # Sticks stream continuously, so only report meaningful deflection and
        # keep the log readable.
        if abs(event.value) > 0.15:
            print(f"[probe] {label}: {event.input.name} = {event.value:+.3f}", flush=True)
        return True

    def subscribe(gamepad, label: str) -> None:
        key = id(gamepad)
        if gamepad is None or key in subscriptions:
            return
        subscriptions[key] = input_interface.subscribe_to_gamepad_events(
            gamepad, lambda event, *a, label=label: on_event(event, *a, label=label)
        )
        print(f"[probe] subscribed to {label}", flush=True)

    # The device-independent listener. This is the one that matters: it fires on
    # any gamepad connecting, whatever slot it lands in and whether or not Kit
    # can name it, and it is the only way to bind a controller that appears
    # *after* start-up -- which a browser-forwarded one always does.
    def on_connection(event) -> None:
        gamepad = getattr(event, "gamepad", None)
        name = input_interface.get_gamepad_name(gamepad) if gamepad is not None else "?"
        print(f"[probe] CONNECTION EVENT: {event!r} name={name!r}", flush=True)
        subscribe(gamepad, f"connected gamepad {name!r}")

    connection_sub = input_interface.subscribe_to_gamepad_connection_events(on_connection)
    print(f"[probe] watching for gamepad connections (subscription {connection_sub})", flush=True)

    at_startup = describe_slots(app_window, input_interface, args_cli.slots)
    if at_startup:
        for slot, name in at_startup.items():
            print(f"[probe] slot {slot} already holds: {name}", flush=True)
    else:
        print(f"[probe] all {args_cli.slots} slots report no gamepad handle at startup", flush=True)

    # Subscribe to every slot that has a handle, named or not. Isaac Lab's
    # Se3Gamepad subscribes to slot 0 alone, once, in its constructor; if that
    # is the bug then this probe has to avoid making the same assumption.
    for slot in range(args_cli.slots):
        subscribe(app_window.get_gamepad(slot), f"slot {slot}")
    print(f"[probe] {len(subscriptions)} slot subscription(s) at startup\n", flush=True)

    deadline = time.time() + args_cli.seconds
    known = dict(at_startup)
    last_poll = 0.0

    while simulation_app.is_running() and time.time() < deadline:
        simulation_app.update()

        now = time.time()
        if now - last_poll > 1.0:
            last_poll = now
            current = describe_slots(app_window, input_interface, args_cli.slots)
            for slot, name in current.items():
                if known.get(slot) != name:
                    print(f"[probe] slot {slot} now holds: {name}", flush=True)
                    known[slot] = name
                    subscribe(app_window.get_gamepad(slot), f"slot {slot}")
            for slot in list(known):
                if slot not in current:
                    print(f"[probe] slot {slot} emptied", flush=True)
                    known.pop(slot)

    input_interface.unsubscribe_to_gamepad_connection_events(connection_sub)

    print(f"\n[probe] finished. {seen_events['count']} gamepad events arrived.", flush=True)
    if not seen_events["count"]:
        print("[probe] none at all: the input never reached the simulator.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
