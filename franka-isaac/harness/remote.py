"""Command construction for the Isaac Lab GPU host.

Isaac Sim does not run on this laptop, so every simulator command is really
three commands nested: ssh to the Brev instance, ``docker exec`` into the
container that holds Isaac Lab, then the Isaac Lab invocation itself. Building
those strings by hand is where the mistakes live, so they are built here, and
tested without a GPU, a network, or a running instance.

Two invariants earned the hard way, both enforced below:

* A headless run must pass ``--viz none``. The scripts under
  ``scripts/environments/`` do ``parser.set_defaults(visualizer=["kit"])``, and
  Kit cannot initialise without a display -- ``--headless`` alone dies with
  "Explicitly requested visualizer(s) ['kit'] could not be configured".
* Headless and livestream are mutually exclusive. ``--livestream 2`` is what
  feeds the browser viewer; a headless run opens no streaming ports at all, so
  asking for both means one of them is a lie.
"""

from __future__ import annotations

import shlex

INSTANCE = "isaac-launchable-e4cbd5"
CONTAINER = "vscode"
ISAACLAB = "/workspace/isaaclab"

# The container ships no system Python; this is the only interpreter on it.
ISAAC_SIM_PYTHON = "/isaac-sim/python.sh"
WORKDIR = "/workspace/robot/franka-isaac"

# The stock Isaac Lab task this project starts from, before any custom assets.
BASE_TASK = "Isaac-Factory-PegInsert-Direct-v0"

# Component 2 validates the recording pipeline on a task that is already known
# to be teleoperable and mimic-ready, before any of it is pointed at insertion.
TELEOP_TASK = "Isaac-Stack-Cube-Franka-IK-Rel-v0"

# Where recorded datasets land inside the container. Deliberately *outside*
# WORKDIR: `make sync` rm -rf's the project directory on the box before copying
# a fresh one in, so a dataset recorded into it is deleted by the next sync --
# which is exactly how the first recorded set of demonstrations was lost.
DATASET_DIR = "/workspace/datasets"
DATASET_FILE = f"{DATASET_DIR}/stack_scripted.hdf5"
TELEOP_DATASET_FILE = f"{DATASET_DIR}/stack_teleop.hdf5"
LIFT_DATASET_FILE = f"{DATASET_DIR}/lift_scripted.hdf5"

# Pick-and-lift: the task the learning work starts from. See reference/LIFT.md.
LIFT_TASK = "Isaac-Lift-Cube-Franka-IK-Rel-v0"

# nginx on the box proxies /viewer/ to the web-viewer app on 5173, and Isaac
# Sim's WebRTC server signals on 8211. Only port 22 is open to the internet, so
# both reach the laptop through an ssh tunnel; see `make tunnel`.
VIEWER_PORT = 80
WEBRTC_PORT = 8211
LOCAL_VIEWER_PORT = 8090


def ssh(inner: str, *, forward_agent: bool = False) -> str:
    """Wrap ``inner`` so it runs on the GPU host.

    ``forward_agent`` is needed only for git operations against the private
    repo; it forwards this machine's ssh-agent rather than putting a key on the
    rented box.
    """
    flags = "-o BatchMode=yes -T"
    if forward_agent:
        flags += " -A"
    return f"ssh {flags} {INSTANCE} {shlex.quote(inner)}"


def in_container(inner: str, *, workdir: str = ISAACLAB) -> str:
    """Wrap ``inner`` so it runs inside the Isaac Lab container on that host."""
    return f"docker exec {CONTAINER} bash -lc {shlex.quote(f'cd {workdir} && {inner}')}"


def isaaclab(
    script: str,
    *,
    task: str = BASE_TASK,
    num_envs: int | None = 4,
    headless: bool = True,
    livestream: bool = False,
    extra: tuple[str, ...] = (),
    launcher: str = "./isaaclab.sh -p",
) -> str:
    """Build an ``isaaclab.sh`` invocation.

    ``num_envs=None`` omits the flag entirely, for scripts that do not take one
    -- passing an argument argparse has never heard of is a hard error, not a
    warning. ``launcher`` is absolute when the working directory is this project
    rather than the Isaac Lab tree.

    Raises:
        ValueError: if both ``headless`` and ``livestream`` are requested.
    """
    if headless and livestream:
        raise ValueError(
            "headless and livestream are mutually exclusive: a headless run "
            "opens no Kit streaming ports, so the browser viewer stays blank"
        )

    args = [f"--task {task}"]
    if num_envs is not None:
        args.append(f"--num_envs {num_envs}")
    if headless:
        # --viz none is not optional here; see module docstring.
        args += ["--headless", "--viz none"]
    if livestream:
        # --viz kit is not optional here; see module docstring. Livestreaming
        # implies headless, and headless with no visualizer renders nothing, so
        # the stream comes up black.
        args += ["--livestream 2", "--viz kit"]
    args += list(extra)

    return f"{launcher} {script} " + " ".join(args)


def smoke(num_envs: int = 4) -> str:
    """Full command for the run that verified the box: random agent, headless."""
    return ssh(in_container(isaaclab("scripts/environments/random_agent.py", num_envs=num_envs)))


def patch_container() -> str:
    """Reapply the container-side fixes in scripts/patch_container.py.

    Two things in Isaac Lab 3.0.0 stop a headless Franka run before it starts: a
    Franka USD path NVIDIA has moved, and a replay script that builds a keyboard
    device needing an app window. Both are in Isaac Lab's own source, so neither
    can be fixed from an environment config on our side -- Isaac Lab's scripts
    hit them too. The patch script explains each one.

    It runs on every ``make sync`` because the container's filesystem does not
    persist, and it is idempotent.

    The container has no system Python at all -- ``python3`` is not on the PATH,
    and running the patch with it exits 127 -- so this falls back to Isaac Sim's
    bundled interpreter, which is the only one there.
    """
    interpreter = f"$(command -v python3 || echo {ISAAC_SIM_PYTHON})"
    return ssh(in_container(f"{interpreter} {WORKDIR}/scripts/patch_container.py", workdir=WORKDIR))


def record_scripted(
    num_demos: int = 5,
    *,
    task: str = TELEOP_TASK,
    dataset_file: str = DATASET_FILE,
    extra: tuple[str, ...] = (),
) -> str:
    """Full command to record scripted demonstrations on the GPU host.

    Runs from ``WORKDIR`` rather than the Isaac Lab tree, because the script
    imports ``harness.stack_sm`` from this project. ``num_envs`` is fixed at 1:
    the recorder manager exports environment ``[0]``, and a second environment
    would be simulated for nothing.
    """
    inner = isaaclab(
        "scripts/record_scripted.py",
        task=task,
        num_envs=None,
        extra=(f"--num_demos {num_demos}", f"--dataset_file {dataset_file}", *extra),
        launcher=f"{ISAACLAB}/isaaclab.sh -p",
    )
    return ssh(in_container(inner, workdir=WORKDIR))


def replay(
    *,
    task: str = TELEOP_TASK,
    dataset_file: str = DATASET_FILE,
    extra: tuple[str, ...] = (),
) -> str:
    """Full command to replay a recorded dataset back through the simulator.

    This is the half of Component 2 that actually validates the recording: a
    dataset that cannot be replayed into the same trajectory is not a
    demonstration, it is a log.
    """
    inner = isaaclab(
        f"{ISAACLAB}/scripts/tools/replay_demos.py",
        task=task,
        num_envs=1,  # --validate_states is only meaningful with one environment
        extra=("--validate_states", "--validate_success_rate", f"--dataset_file {dataset_file}", *extra),
        launcher=f"{ISAACLAB}/isaaclab.sh -p",
    )
    return ssh(in_container(inner, workdir=WORKDIR))


def record_teleop(
    num_demos: int = 5,
    *,
    task: str = TELEOP_TASK,
    dataset_file: str = TELEOP_DATASET_FILE,
    device: str = "keyboard",
    extra: tuple[str, ...] = (),
) -> str:
    """Full command for a human-teleoperated recording session.

    This is Isaac Lab's own ``record_demos.py``, unmodified, driven by a person
    at a keyboard. It is the counterpart to :func:`record_scripted`: the same
    recorder, the same HDF5, a different source of actions.

    It cannot be headless. A keyboard device needs an application window to
    attach to, so the run streams instead -- ``--livestream 2`` -- and the
    window it attaches to is served to a browser. :func:`tunnel` is how that
    browser reaches it.

    ``--teleop_device`` is passed explicitly to force Isaac Lab's legacy device
    path. Left off, ``record_demos.py`` prefers the IsaacTeleop/CloudXR pipeline
    when the task configures one, which wants a VR headset rather than a
    keyboard.
    """
    inner = isaaclab(
        f"{ISAACLAB}/scripts/tools/record_demos.py",
        task=task,
        num_envs=None,  # record_demos.py fixes this at 1 and defines no flag
        headless=False,
        livestream=True,
        extra=(
            f"--teleop_device {device}",
            f"--num_demos {num_demos}",
            f"--dataset_file {dataset_file}",
            *extra,
        ),
        launcher=f"{ISAACLAB}/isaaclab.sh -p",
    )
    return ssh(in_container(inner, workdir=WORKDIR))


def tunnel() -> str:
    """Forward the browser viewer and its WebRTC signalling to this laptop.

    The instance's security group opens port 22 and nothing else, so the viewer
    is reached the same way everything else on the box is: through ssh. ``-N``
    opens the forwards without running a command, so the tunnel stays up until
    it is interrupted.

    ``ControlPath=none`` is what makes that last sentence true. Brev's generated
    ssh config turns on connection multiplexing, and against an existing master
    connection ``ssh -N`` hands over the forwards and *exits immediately*: the
    tunnel is up, the command looks like it failed, ctrl-C no longer closes
    anything, and the next attempt collides with the ports it left behind.
    Opening a dedicated connection costs one more TCP session and makes the
    tunnel behave the way the shell suggests it does.
    """
    forwards = f"-L {LOCAL_VIEWER_PORT}:localhost:{VIEWER_PORT} -L {WEBRTC_PORT}:localhost:{WEBRTC_PORT}"
    return f"ssh -N -o ControlPath=none -o ExitOnForwardFailure=yes {forwards} {INSTANCE}"


def viewer_url() -> str:
    """Where to point a browser once :func:`tunnel` is running."""
    return f"http://localhost:{LOCAL_VIEWER_PORT}/viewer/"


# Every script of ours, or Isaac Lab's, that starts a simulator in the
# container. `make kill` needs all of them: a teleoperation session left running
# because the cleanup only knew about random_agent.py bills exactly as much as
# one nobody tried to stop.
SIM_SCRIPTS = (
    "random_agent.py",
    "zero_agent.py",
    "record_scripted.py",
    "record_lift.py",
    "train_diffusion.py",
    "train_ppo.py",
    "eval_diffusion.py",
    "record_demos.py",
    "replay_demos.py",
    "gamepad_probe.py",
)


# Where long runs write their logs and checkpoints. Outside WORKDIR, because
# `make sync` deletes that directory before copying a fresh one in.
RUNS_DIR = "/workspace/runs"


def detached(inner: str, name: str) -> str:
    """Start a long run on the box that outlives this ssh connection.

    Recording a hundred demonstrations takes many minutes; training takes hours.
    Run through a plain ``ssh`` and the process dies when the laptop sleeps or
    the connection blips -- and the instance carries on billing regardless, which
    is the worst of both outcomes.

    ``setsid`` detaches it from the terminal, stdin is closed so nothing can
    block waiting for input, and everything goes to a log file that
    :func:`tail_log` can follow later.
    """
    log = f"{RUNS_DIR}/{name}.log"
    command = (
        f"mkdir -p {RUNS_DIR} && "
        f"setsid nohup {inner} > {log} 2>&1 < /dev/null & "
        f"sleep 1 && echo 'started {name}, logging to {log}'"
    )
    return ssh(in_container(command, workdir=WORKDIR))


def tail_log(name: str, lines: int = 40) -> str:
    """Show the end of a detached run's log."""
    return ssh(in_container(f"tail -n {lines} {RUNS_DIR}/{name}.log", workdir=WORKDIR))


def list_runs() -> str:
    """What logs exist on the box, and how recent they are."""
    return ssh(in_container(f"ls -lht {RUNS_DIR}/ 2>/dev/null || echo 'no runs yet'", workdir=WORKDIR))


def pull_runs(local_dir: str = "runs") -> str:
    """Copy logs and checkpoints back, because the container is not persistent."""
    return (
        f"ssh {INSTANCE} 'rm -rf ~/runs && docker cp {CONTAINER}:{RUNS_DIR} ~/runs' && "
        f"mkdir -p {local_dir} && rsync -az {INSTANCE}:~/runs/ {local_dir}/"
    )


def record_lift(num_demos: int = 100, dataset_file: str = LIFT_DATASET_FILE) -> str:
    """Command to record scripted pick-and-lift demonstrations, headless."""
    return isaaclab(
        "scripts/record_lift.py",
        task=LIFT_TASK,
        num_envs=None,  # record_lift.py fixes this at 1
        extra=(f"--num_demos {num_demos}", f"--dataset_file {dataset_file}"),
        launcher=f"{ISAACLAB}/isaaclab.sh -p",
    )


def train_diffusion(epochs: int = 300, dataset: str = LIFT_DATASET_FILE) -> str:
    """Train the diffusion policy on the box. Needs no simulator, only a GPU."""
    return (
        f"{ISAACLAB}/isaaclab.sh -p scripts/train_diffusion.py "
        f"--dataset {dataset} --epochs {epochs} --out {RUNS_DIR}/diffusion"
    )


def eval_diffusion(episodes: int = 50, num_envs: int = 25) -> str:
    """Roll the trained diffusion policy out and count successes."""
    return isaaclab(
        "scripts/eval_diffusion.py",
        task=LIFT_TASK,
        num_envs=None,  # the script has its own --num_envs
        extra=(
            f"--episodes {episodes}",
            f"--num_envs {num_envs}",
            f"--checkpoint {RUNS_DIR}/diffusion/policy.pt",
        ),
        launcher=f"{ISAACLAB}/isaaclab.sh -p",
    )


def train_ppo(iterations: int = 500, num_envs: int = 1024, pick_only: bool = False) -> str:
    """Train PPO on the lift task, optionally without the goal-tracking reward."""
    extra = [f"--iterations {iterations}", f"--num_envs {num_envs}", f"--out {RUNS_DIR}/ppo"]
    if pick_only:
        extra.append("--pick-only")
    return isaaclab(
        "scripts/train_ppo.py",
        task=LIFT_TASK,
        num_envs=None,  # the script has its own --num_envs
        extra=tuple(extra),
        launcher=f"{ISAACLAB}/isaaclab.sh -p",
    )


def kill_sim(script_name: str | None = None) -> str:
    """Kill simulator processes left running inside the container.

    With no argument this targets every known entry point, which is what
    ``make kill`` wants. Pass a name to kill just one, which is what a script
    cleaning up after its own run wants.

    This matters because a capped ``docker exec`` kills only the local client:
    the process inside the container keeps running, and the instance keeps
    billing, until someone notices.
    """
    names = (script_name,) if script_name else SIM_SCRIPTS
    # `|| true` per name: pkill exits non-zero when nothing matched, which is
    # the normal case for most of these and not a failure.
    inner = " ; ".join(f"pkill -f {shlex.quote(name)} || true" for name in names)
    return ssh(f"docker exec {CONTAINER} bash -lc {shlex.quote(inner)}")
