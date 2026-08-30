"""The laptop-to-box teleoperation link, exercised over a real loopback socket.

These run the actual server and client against each other on 127.0.0.1. There is
no simulator involved and nothing is mocked: the thing under test is a socket
protocol, and a mocked socket would only prove that the mock behaves.

The property worth the most here is the safety one. A control link that keeps
commanding motion after the far end has gone away turns a dropped wifi
connection into a robot driving itself into the table, so "stale means stop" is
tested directly rather than assumed.
"""

import time

import pytest

from harness.teleop_link import (
    STOP,
    CommandSender,
    NetworkTeleopDevice,
    Se3Command,
)


@pytest.fixture
def device():
    """A device on an ephemeral port, closed when the test finishes."""
    device = NetworkTeleopDevice(port=0, max_age=0.2)
    yield device
    device.close()


@pytest.fixture
def sender(device):
    with CommandSender(port=device.address[1]) as sender:
        yield sender


def wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll until true. The reader is a thread, so arrival is not instant."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_command_survives_the_round_trip(device, sender):
    sent = Se3Command(dpos=(0.01, -0.02, 0.03), drot=(0.0, 0.0, 0.4), close_gripper=True)
    sender.send(sent)

    assert wait_until(lambda: device.stats.received == 1)
    action = device.advance()

    assert action[:3] == pytest.approx([0.01, -0.02, 0.03])
    assert action[3:6] == pytest.approx([0.0, 0.0, 0.4])
    assert action[6] == -1.0  # closed; Isaac Lab's convention is negative


def test_gripper_convention_matches_isaac_lab():
    # +1 opens, -1 closes -- the same convention harness.stack_sm emits.
    assert Se3Command(close_gripper=False).as_action()[6] == 1.0
    assert Se3Command(close_gripper=True).as_action()[6] == -1.0


def test_a_stale_command_becomes_a_stop(device, sender):
    sender.send(Se3Command(dpos=(0.05, 0.05, 0.05)))
    assert wait_until(lambda: device.stats.received == 1)
    assert device.advance()[:3] == pytest.approx([0.05, 0.05, 0.05])

    # Nothing further arrives: the laptop has gone quiet.
    time.sleep(0.3)  # longer than max_age

    action = device.advance()
    assert action[:3] == pytest.approx([0.0, 0.0, 0.0]), "a stale link must stop the arm, not coast"
    assert action[3:6] == pytest.approx([0.0, 0.0, 0.0])
    assert device.stats.stale_reads >= 1


def test_a_disconnected_laptop_stops_the_arm(device):
    with CommandSender(port=device.address[1]) as sender:
        sender.send(Se3Command(dpos=(0.05, 0.0, 0.0)))
        assert wait_until(lambda: device.stats.received == 1)

    # The sender has closed. Whatever the arm was doing, it stops.
    time.sleep(0.3)
    assert device.advance() == pytest.approx(STOP.as_action())


def test_only_the_newest_command_is_acted_on(device, sender):
    # A control link, not a queue: a backlog would make the arm lag further
    # behind the longer it ran.
    for step in range(20):
        sender.send(Se3Command(dpos=(step / 1000, 0.0, 0.0)))

    assert wait_until(lambda: device.stats.received >= 1)
    time.sleep(0.05)
    assert device.advance()[0] == pytest.approx(0.019)


def test_a_malformed_newest_line_falls_back_to_the_last_good_one(device, sender):
    # The reader walks backwards from the newest line and stops at the first one
    # it can decode, so garbage at the head of the stream costs one command
    # rather than the whole batch.
    sender.send(Se3Command(dpos=(0.01, 0.0, 0.0)))
    sender._socket.sendall(b"this is not json\n")

    assert wait_until(lambda: device.stats.received == 1)
    assert device.stats.malformed == 1
    assert device.advance()[0] == pytest.approx(0.01)


def test_garbage_never_becomes_motion(device, sender):
    # Whether garbage is counted depends on how the stream happens to be split
    # into packets, which is not ours to control and not worth asserting. What
    # must hold either way: rubbish on the wire never moves the arm, and a real
    # command after it still gets through.
    sender._socket.sendall(b"rubbish\nalso rubbish\n")
    assert wait_until(lambda: device.stats.malformed > 0 or device.stats.received > 0)
    assert device.advance() == pytest.approx(STOP.as_action())

    sender.send(Se3Command(dpos=(0.02, 0.0, 0.0)))
    assert wait_until(lambda: device.stats.received == 1)
    assert device.advance()[0] == pytest.approx(0.02)


def test_decoding_rejects_a_line_that_is_not_a_command():
    with pytest.raises(ValueError, match="not a teleop command"):
        Se3Command.from_json('{"something": "else"}')
    with pytest.raises(ValueError, match="not a teleop command"):
        Se3Command.from_json("half a line")


def test_decoding_rejects_the_wrong_number_of_axes():
    with pytest.raises(ValueError, match="expected 3 position"):
        Se3Command.from_json('{"dpos": [1, 2], "drot": [0, 0, 0]}')


def test_json_round_trips_exactly():
    command = Se3Command(dpos=(0.125, -0.25, 0.5), drot=(-1.0, 0.0, 0.75), close_gripper=True)
    assert Se3Command.from_json(command.to_json()) == command


def test_reset_forgets_the_current_command(device, sender):
    sender.send(Se3Command(dpos=(0.05, 0.0, 0.0)))
    assert wait_until(lambda: device.stats.received == 1)

    device.reset()

    assert device.advance() == pytest.approx(STOP.as_action())


def test_the_device_accepts_the_callbacks_a_recorder_binds(device):
    # record_demos.py binds a reset key on whatever device it is given. This one
    # has no buttons, but it must not refuse.
    device.add_callback("R", lambda: None)


def test_a_device_with_no_client_reports_a_stop(device):
    assert device.advance() == pytest.approx(STOP.as_action())
    assert device.stats.received == 0


def test_stats_format_is_readable(device):
    assert "0 connection(s)" in device.stats.format()
    assert "never" in device.stats.format()


def test_an_injected_tensor_factory_is_used():
    """The box needs torch tensors; the laptop must not need torch to test this.

    Isaac Lab's recorder calls .repeat(num_envs, 1) on a device's output, so on
    the box advance() has to return a tensor. Injecting the conversion is what
    keeps this module importable, and this whole file runnable, on a machine
    with no simulator.
    """
    device = NetworkTeleopDevice(port=0, as_tensor=lambda values: ("tensor", tuple(values)))
    try:
        kind, values = device.advance()
        assert kind == "tensor"
        assert values == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    finally:
        device.close()
