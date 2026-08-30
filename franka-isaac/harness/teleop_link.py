"""A teleoperation link from the laptop to the simulator on the GPU box.

NVIDIA's browser stream forwards keyboard and mouse to the remote Isaac Sim but
not, as far as five separate experiments could establish, a gamepad: Chrome sees
a DualSense perfectly and Kit never receives one. Rather than keep guessing at a
closed pipe, this opens our own.

The shape is deliberately the one Component 6 needs anyway. Hand tracking will
have exactly the same problem -- a webcam on the laptop, a simulator on a rented
box -- and exactly the same answer: read the device where it is plugged in, turn
it into an end-effector command, and ship the command. What travels is six
numbers and a boolean, so the link is small, and the interesting parts stay on
whichever side can actually see the hardware.

::

    laptop                                    GPU box (inside the container)
    ------                                    -----------------------------
    DualSense / webcam / anything             NetworkTeleopDevice.advance()
      -> Se3Command                             ^
      -> CommandSender.send()  --- ssh -L --->  TCP server on 127.0.0.1

The transport is a TCP socket over an ssh forward, because ssh is the one thing
this instance definitely allows. Nothing is exposed to the network: the box side
listens on loopback, and ``ssh -L`` carries it.

Two properties matter more than throughput:

* **A stale command means stop.** If the laptop disconnects, the process dies,
  or the link stalls, the arm must not keep moving on the last thing it heard.
  :meth:`NetworkTeleopDevice.advance` returns zero motion once a command is
  older than ``max_age``, which is the difference between a dropped connection
  being a non-event and being a crash into the table.
* **The newest command wins.** This is a live control link, not a queue. A
  backlog of stale poses would make the arm lag further behind the longer it
  ran, so the reader keeps only the last line it saw.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field

# Loopback on the box; `ssh -L` is what connects the laptop to it.
DEFAULT_PORT = 5599
DEFAULT_HOST = "127.0.0.1"

# How old a command may be before it is treated as "no command". At a 60 Hz send
# rate this is many missed packets, so it does not trip on jitter, but it is
# still a fifth of a second of travel rather than an unbounded amount.
DEFAULT_MAX_AGE_S = 0.2


@dataclass(frozen=True)
class Se3Command:
    """One end-effector command: a pose delta and a gripper state.

    The three position and three rotation numbers are deltas in the environment
    frame, in the same units the task's action space expects. ``close_gripper``
    is a latched state rather than an edge, so a dropped packet cannot leave the
    gripper in the wrong position.
    """

    dpos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    close_gripper: bool = False

    def to_json(self) -> str:
        """Encode as one line. Newline-delimited JSON keeps the reader trivial."""
        return json.dumps(
            {"dpos": list(self.dpos), "drot": list(self.drot), "grip": bool(self.close_gripper)},
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> Se3Command:
        """Decode one line.

        Raises:
            ValueError: if the line is not a command. The caller is reading from
                a socket, where a truncated or foreign line is a normal thing to
                meet, and silently treating it as zeros would hide a real fault.
        """
        try:
            payload = json.loads(line)
            dpos = tuple(float(v) for v in payload["dpos"])
            drot = tuple(float(v) for v in payload["drot"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a teleop command: {line[:80]!r}") from exc

        if len(dpos) != 3 or len(drot) != 3:
            raise ValueError(f"expected 3 position and 3 rotation values, got {len(dpos)} and {len(drot)}")

        return cls(dpos=dpos, drot=drot, close_gripper=bool(payload.get("grip", False)))

    def as_action(self, gripper_open_value: float = 1.0) -> list[float]:
        """The 7 numbers the task's action space wants.

        The gripper convention is Isaac Lab's: positive opens, negative closes.
        """
        gripper = -gripper_open_value if self.close_gripper else gripper_open_value
        return [*self.dpos, *self.drot, gripper]


STOP = Se3Command()


@dataclass
class LinkStats:
    """What the link has done, for the diagnostics a remote link always needs."""

    received: int = 0
    malformed: int = 0
    connections: int = 0
    last_command_at: float = 0.0
    stale_reads: int = 0

    def format(self) -> str:
        age = "never" if not self.last_command_at else f"{time.monotonic() - self.last_command_at:.1f}s ago"
        return (
            f"{self.connections} connection(s), {self.received} commands, "
            f"{self.malformed} malformed, {self.stale_reads} stale reads, last {age}"
        )


class NetworkTeleopDevice:
    """The box side: a teleoperation device fed by the laptop over a socket.

    Deliberately duck-typed rather than a subclass of Isaac Lab's ``DeviceBase``.
    ``record_demos.py`` only ever calls ``advance``, ``reset`` and
    ``add_callback`` on a device, and not inheriting keeps this module free of
    any ``isaaclab`` import -- so the whole link, including this class, is
    testable on a laptop that cannot run the simulator at all.

    It starts its own listener thread on construction, which suits how Isaac Lab
    builds devices: once, at start-up, long before anyone connects.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_age: float = DEFAULT_MAX_AGE_S,
        gripper_open_value: float = 1.0,
        report_every: float = 5.0,
        as_tensor=None,
    ):
        self.host = host
        self.port = port
        self.max_age = max_age
        self.gripper_open_value = gripper_open_value
        # Isaac Lab's recorder calls .repeat(num_envs, 1) on whatever a device
        # returns, so on the box this has to be a torch tensor. The conversion
        # is injected rather than imported: keeping torch out of this module is
        # what lets the whole link be tested on a laptop that has no simulator
        # and no CUDA. The container-side patch supplies it.
        self._as_tensor = as_tensor

        self.stats = LinkStats()
        self.report_every = report_every
        self._reported_at = 0.0
        self._command = STOP
        self._command_at = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._callbacks: dict[str, object] = {}

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        self._server.settimeout(0.5)

        self._thread = threading.Thread(target=self._serve, name="teleop-link", daemon=True)
        self._thread.start()

    @property
    def address(self) -> tuple[str, int]:
        """Where this is actually listening, which matters when port is 0."""
        return self._server.getsockname()

    def advance(self):
        """Return the latest command, or a stop if it has gone stale.

        Returns a plain list of 7 numbers, or whatever ``as_tensor`` makes of
        them when one was supplied.
        """
        with self._lock:
            command, at = self._command, self._command_at

        if at and (time.monotonic() - at) > self.max_age:
            self.stats.stale_reads += 1
            command = STOP

        values = command.as_action(self.gripper_open_value)
        return self._as_tensor(values) if self._as_tensor is not None else values

    def reset(self) -> None:
        """Forget the current command. Called between episodes."""
        with self._lock:
            self._command = STOP
            self._command_at = 0.0

    def add_callback(self, key: str, func) -> None:
        """Accepted so this can stand in for a keyboard, and otherwise unused.

        Isaac Lab's recorder binds a reset key here. A laptop-side device sends
        pose commands and nothing else, so the callbacks are held but never
        fired; they are kept rather than rejected so the device is a drop-in.
        """
        self._callbacks[key] = func

    def close(self) -> None:
        self._stop.set()
        self._server.close()

    def _serve(self) -> None:
        """Accept one client at a time and read its commands until it leaves."""
        while not self._stop.is_set():
            try:
                connection, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return  # the socket was closed underneath us; time to stop

            self.stats.connections += 1
            print(f"[network] laptop connected ({self.stats.connections} so far)", flush=True)
            with connection:
                self._read_commands(connection)

    def _maybe_report(self) -> None:
        """Say something occasionally, so the link is visible from the log.

        A remote control link that works and one that is silently dead look
        identical from the box's console otherwise, and the console is all there
        is when the picture comes over a video stream.
        """
        if self.report_every <= 0:
            return
        now = time.monotonic()
        if self._reported_at and (now - self._reported_at) < self.report_every:
            return
        self._reported_at = now
        print(f"[network] {self.stats.format()}", flush=True)

    def _read_commands(self, connection: socket.socket) -> None:
        connection.settimeout(0.5)
        buffer = b""

        while not self._stop.is_set():
            try:
                chunk = connection.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                return
            if not chunk:
                print("[network] laptop disconnected; the arm stops", flush=True)
                return  # staleness takes over from here

            buffer += chunk
            *lines, buffer = buffer.split(b"\n")

            # Only the newest line is worth decoding. This is a control link,
            # not a queue: acting on a backlog would make the arm lag further
            # behind the longer the session ran.
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    command = Se3Command.from_json(line.decode("utf-8", "replace"))
                except ValueError:
                    self.stats.malformed += 1
                    continue
                with self._lock:
                    self._command = command
                    self._command_at = time.monotonic()
                self.stats.received += 1
                self.stats.last_command_at = self._command_at
                self._maybe_report()
                break


@dataclass
class CommandSender:
    """The laptop side: connects to the forwarded port and streams commands."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    _socket: socket.socket | None = field(default=None, repr=False)

    def connect(self, timeout: float = 5.0) -> None:
        self._socket = socket.create_connection((self.host, self.port), timeout=timeout)

    def send(self, command: Se3Command) -> None:
        """Send one command.

        Raises:
            RuntimeError: if called before :meth:`connect`.
        """
        if self._socket is None:
            raise RuntimeError("connect() before sending")
        self._socket.sendall((command.to_json() + "\n").encode("utf-8"))

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> CommandSender:
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
