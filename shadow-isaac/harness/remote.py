"""Command construction for the Isaac Lab GPU host.

Isaac Sim does not run on this laptop, so every simulator command is really
three nested: ssh to the Brev instance, ``docker exec`` into the container that
holds Isaac Lab, then the Isaac Lab invocation. Building those strings by hand
is where the mistakes live, so they are built here and tested without a GPU, a
network, or a running instance.

The invariants below were each paid for once on the sibling project
(``franka-isaac``) and are enforced rather than remembered:

* A headless run passes ``--viz none`` and NOT ``--headless``. Isaac Lab 3.0
  replaced the headless flag with a visualizer system: ``--headless`` is
  deprecated, and stock scripts default the visualizer to ``kit``, which cannot
  initialise without a display. ``--viz none`` is correct in both cases.
* Headless and livestream are mutually exclusive. ``--livestream 2`` feeds the
  browser viewer; a headless run opens no streaming ports, so asking for both
  means one of them is a lie -- and the failure looks like a broken network
  path rather than a flag mistake, because the stream connects and is black.
* Datasets and logs go OUTSIDE the project directory. ``make sync`` deletes the
  project directory on the box before copying a fresh one in, so a training run
  that writes into it is erased by the next sync.
"""

from __future__ import annotations

import shlex

INSTANCE = "isaac-launchable-e4cbd5"
CONTAINER = "vscode"
ISAACLAB = "/workspace/isaaclab"

# The container ships no system Python; this is the only interpreter on it.
ISAAC_SIM_PYTHON = "/isaac-sim/python.sh"
WORKDIR = "/workspace/robot/shadow-isaac"

TASK = "Catch-Shadow-Direct-v0"

# Outside WORKDIR on purpose: `make sync` rm -rf's WORKDIR on the box.
LOG_DIR = "/workspace/logs/shadow-isaac"

# The viewer is the Brev console URL with /viewer appended, taken from the
# console's Access tab. It goes through Brev's proxy, which already exposes
# port 80 and the WebRTC media port, so it needs no tunnel and no
# security-group change.
VIEWER_URL = "https://isaac-fzq49bb7n.brevlab.com/viewer"
VIEWER_PORT = 80
WEBRTC_PORT = 47998
LOCAL_VIEWER_PORT = 8090


def ssh(inner: str, *, forward_agent: bool = False) -> str:
    """Wrap ``inner`` so it runs on the GPU host."""
    flags = "-o BatchMode=yes -T"
    if forward_agent:
        flags += " -A"
    return f"ssh {flags} {INSTANCE} {shlex.quote(inner)}"


def in_container(inner: str, *, workdir: str = WORKDIR) -> str:
    """Wrap ``inner`` so it runs inside the Isaac Lab container on that host."""
    return f"docker exec {CONTAINER} bash -lc {shlex.quote(f'cd {workdir} && {inner}')}"


def isaaclab(
    script: str,
    *,
    task: str | None = TASK,
    num_envs: int | None = 4,
    headless: bool = True,
    livestream: bool = False,
    extra: tuple[str, ...] = (),
) -> str:
    """Build an ``isaaclab.sh`` invocation.

    ``num_envs=None`` and ``task=None`` omit those flags entirely, for scripts
    that do not accept them -- passing an argument argparse has never heard of
    is a hard error, not a warning.

    Raises:
        ValueError: if both ``headless`` and ``livestream`` are requested.
    """
    if headless and livestream:
        raise ValueError(
            "headless and livestream are mutually exclusive: a headless run "
            "opens no Kit streaming ports, so the browser viewer stays blank"
        )
    args = []
    if task is not None:
        args.append(f"--task {task}")
    if num_envs is not None:
        args.append(f"--num_envs {num_envs}")
    if headless:
        # --viz none, and NOT --headless. Isaac Lab 3.0 deprecated --headless
        # ("Omit '--viz' for default headless"), and passing both warns that
        # the deprecated flag takes precedence. --viz none is the form that
        # works in both cases: default-headless scripts are unaffected, and
        # scripts that do parser.set_defaults(visualizer=["kit"]) -- which
        # cannot initialise without a display -- are correctly overridden.
        args += ["--viz none"]
    if livestream:
        args += ["--livestream 2", "--viz kit"]
    args += list(extra)
    return f"{ISAACLAB}/isaaclab.sh -p {script} " + " ".join(args)


def install_extension() -> str:
    """Install this project's extension into the container's interpreter.

    Editable, so a `make sync` that copies new source in takes effect without
    reinstalling -- the package directory is replaced under the same path.
    """
    return ssh(in_container(f"{ISAAC_SIM_PYTHON} -m pip install -e source/catching"))


def smoke(num_envs: int = 4) -> str:
    """Random agent against our task, headless. The does-it-load check.

    Ours, not Isaac Lab's. Isaac Lab's scripts import `isaaclab_tasks` and
    nothing else, so an external extension's `gym.register` never runs and the
    task comes back NameNotFound however correctly it is installed.
    """
    return ssh(in_container(isaaclab("scripts/random_agent.py", num_envs=num_envs)))


def scripted(num_envs: int = 64, steps: int = 2000, lead_time: float = 0.08,
             livestream: bool = False, spawn_toward: float | None = None) -> str:
    """The scripted catcher: the gate before any GPU hours go into PPO.

    ``livestream=True`` is what makes the run watchable. Without it the run is
    headless with ``--viz none``, no Kit streaming ports listen at all, and the
    browser viewer correctly shows nothing -- which looks exactly like a broken
    viewer. Watch at `VIEWER_URL`; do NOT try to reach it through `tunnel()`.
    """
    return ssh(in_container(isaaclab(
        "scripts/scripted_catch.py", num_envs=num_envs,
        headless=not livestream, livestream=livestream,
        extra=(f"--steps {steps}", f"--lead-time {lead_time}")
        + ((f"--spawn-toward {spawn_toward}",) if spawn_toward is not None else ()))))


def train(num_envs: int = 4096, max_iterations: int = 1000) -> str:
    return ssh(in_container(isaaclab(
        f"{ISAACLAB}/scripts/reinforcement_learning/rl_games/train.py",
        num_envs=num_envs,
        extra=(f"--max_iterations {max_iterations}",),
    )))


def play(checkpoint: str, num_envs: int = 1, livestream: bool = True) -> str:
    """Watch a checkpoint in the browser viewer."""
    return ssh(in_container(isaaclab(
        f"{ISAACLAB}/scripts/reinforcement_learning/rl_games/play.py",
        num_envs=num_envs,
        headless=not livestream,
        livestream=livestream,
        extra=(f"--checkpoint {shlex.quote(checkpoint)}",),
    )))


def list_envs() -> str:
    """Registered task ids, to confirm ours registered and to find stock ones."""
    return ssh(in_container(isaaclab(
        f"{ISAACLAB}/scripts/environments/list_envs.py", task=None, num_envs=None,
        headless=False)))


def body_names() -> str:
    """Print the Shadow Hand's body and joint names from the loaded asset.

    Needed because the fingertip body pattern in `CatchEnvCfg` is a guess until
    it is read off the asset -- and a contact sensor whose regex matches nothing
    reports no contacts at all rather than failing, which would show up as a
    policy that can never catch.
    """
    return ssh(in_container(isaaclab("scripts/inspect_hand.py", task=None,
                                     num_envs=None)))


def orient_hand() -> str:
    """Measure the palm's facing and print the rotation that turns it up."""
    return ssh(in_container(isaaclab("scripts/orient_hand.py", task=None, num_envs=None)))


def find_palm_up() -> str:
    """Search all 24 axis-aligned orientations for palm-up, in one session."""
    return ssh(in_container(isaaclab("scripts/find_palm_up.py", task=None, num_envs=None)))


def inspect_pose() -> str:
    """Print the running hand's raw world geometry."""
    return ssh(in_container(isaaclab("scripts/inspect_pose.py", num_envs=None)))


def tunnel() -> str:
    """An ssh port-forward to the box. NOT a route to the 3D viewer.

    Kept for forwarding ordinary TCP services. The viewer does not work this
    way and cannot: the page loads over TCP and then the video never arrives,
    because Kit streams WebRTC over UDP and ssh forwards TCP only. It looks
    like a broken stream and is a broken idea. Use `VIEWER_URL`, which goes
    through Brev's proxy.

    ControlPath=none because Brev's generated ssh config enables multiplexing,
    so `ssh -N -L` against an existing master registers the forwards and exits
    immediately -- the tunnel is up, the command looks like it failed, and
    ctrl-C closes nothing.
    """
    return (f"ssh -o ControlPath=none -N "
            f"-L {LOCAL_VIEWER_PORT}:localhost:{VIEWER_PORT} "
            f"-L {WEBRTC_PORT}:localhost:{WEBRTC_PORT} {INSTANCE}")


def kill() -> str:
    """Stop a simulator left running in the container.

    `timeout N docker exec ...` kills only the local client; the simulator keeps
    running on the box and keeps billing.
    """
    return ssh(f"docker exec {CONTAINER} pkill -f 'isaac-sim|kit' || true")
