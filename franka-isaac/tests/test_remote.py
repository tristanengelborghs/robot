"""The nesting is ssh -> docker exec -> isaaclab.sh, and each layer quotes the
next. These tests pin the two invariants that cost real GPU time to discover."""

import pytest

from harness import remote


def test_headless_run_always_disables_visualizers():
    # Without --viz none the demo scripts default to the Kit visualizer and
    # crash on a display-less host.
    cmd = remote.isaaclab("scripts/environments/random_agent.py")
    assert "--headless" in cmd
    assert "--viz none" in cmd


def test_livestream_run_is_not_headless():
    cmd = remote.isaaclab(
        "scripts/environments/random_agent.py", headless=False, livestream=True
    )
    assert "--livestream 2" in cmd
    assert "--headless" not in cmd
    assert "--viz none" not in cmd


def test_headless_and_livestream_together_is_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        remote.isaaclab("scripts/environments/random_agent.py", livestream=True)


def test_defaults_to_the_stock_factory_task():
    assert remote.BASE_TASK == "Isaac-Factory-PegInsert-Direct-v0"
    assert remote.BASE_TASK in remote.isaaclab("scripts/environments/random_agent.py")


def test_container_command_cds_into_isaaclab():
    cmd = remote.in_container("echo hi")
    assert cmd.startswith(f"docker exec {remote.CONTAINER} bash -lc ")
    assert remote.ISAACLAB in cmd


def test_agent_forwarding_is_opt_in():
    # The repo is private, so git operations need -A; simulator runs must not
    # forward the agent unnecessarily.
    assert " -A " in remote.ssh("git pull", forward_agent=True)
    assert " -A " not in remote.ssh("nvidia-smi")


def test_smoke_nests_all_three_layers():
    cmd = remote.smoke()
    assert cmd.startswith("ssh ")
    assert "docker exec" in cmd
    assert "isaaclab.sh" in cmd
