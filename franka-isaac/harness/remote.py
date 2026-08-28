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
WORKDIR = "/workspace/robot/franka-isaac"

# The stock Isaac Lab task this project starts from, before any custom assets.
BASE_TASK = "Isaac-Factory-PegInsert-Direct-v0"


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
    num_envs: int = 4,
    headless: bool = True,
    livestream: bool = False,
    extra: tuple[str, ...] = (),
) -> str:
    """Build an ``isaaclab.sh`` invocation.

    Raises:
        ValueError: if both ``headless`` and ``livestream`` are requested.
    """
    if headless and livestream:
        raise ValueError(
            "headless and livestream are mutually exclusive: a headless run "
            "opens no Kit streaming ports, so the browser viewer stays blank"
        )

    args = [f"--task {task}", f"--num_envs {num_envs}"]
    if headless:
        # --viz none is not optional here; see module docstring.
        args += ["--headless", "--viz none"]
    if livestream:
        args.append("--livestream 2")
    args += list(extra)

    return f"./isaaclab.sh -p {script} " + " ".join(args)


def smoke(num_envs: int = 4) -> str:
    """Full command for the run that verified the box: random agent, headless."""
    return ssh(in_container(isaaclab("scripts/environments/random_agent.py", num_envs=num_envs)))


def kill_sim(script_name: str = "random_agent.py") -> str:
    """Kill a simulator process left behind inside the container.

    ``timeout N docker exec ...`` kills only the local client; the process in
    the container keeps running and keeps billing. This is the cleanup.
    """
    return ssh(f"docker exec {CONTAINER} pkill -f {shlex.quote(script_name)}")
