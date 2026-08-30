"""Remote command construction, tested with no GPU, network, or instance."""

import pytest

from harness import remote


def test_headless_runs_disable_the_visualizer_without_the_deprecated_flag():
    """--viz none, not --headless. Isaac Lab 3.0 deprecated --headless, and
    passing both warns that the deprecated one wins. --viz none is right in
    both cases: default-headless scripts are unaffected, and scripts that do
    set_defaults(visualizer=["kit"]) -- which cannot initialise without a
    display -- are correctly overridden."""
    cmd = remote.isaaclab("scripts/foo.py")
    assert "--viz none" in cmd
    assert "--headless" not in cmd


def test_livestream_asks_for_kit():
    """A livestream with no visualizer connects fine and carries a black
    picture, which looks exactly like a broken network path."""
    cmd = remote.isaaclab("scripts/foo.py", headless=False, livestream=True)
    assert "--livestream 2" in cmd and "--viz kit" in cmd


def test_headless_and_livestream_are_refused():
    with pytest.raises(ValueError, match="mutually exclusive"):
        remote.isaaclab("scripts/foo.py", headless=True, livestream=True)


def test_absent_flags_are_omitted_not_defaulted():
    """Passing an argument argparse has never heard of is a hard error."""
    cmd = remote.isaaclab("scripts/foo.py", task=None, num_envs=None)
    assert "--task" not in cmd and "--num_envs" not in cmd


def test_the_task_id_is_ours():
    assert "--task Catch-Shadow-Direct-v0" in remote.smoke()


def test_the_smoke_script_is_ours_not_isaac_labs():
    """Isaac Lab's scripts import isaaclab_tasks and nothing else, so an
    external extension's gym.register never runs: the task is NameNotFound
    however correctly the package is installed. Registration needs an import,
    and installing is not importing."""
    cmd = remote.smoke()
    assert "-p scripts/random_agent.py" in cmd
    assert "isaaclab/scripts/environments/random_agent.py" not in cmd


def test_logs_live_outside_the_synced_tree():
    """`make sync` rm -rf's WORKDIR on the box before copying a fresh tree in,
    so anything written inside it dies at the next sync."""
    assert not remote.LOG_DIR.startswith(remote.WORKDIR)


def test_the_tunnel_disables_ssh_multiplexing():
    """Brev's generated ssh config enables multiplexing, so `ssh -N -L` against
    an existing master registers the forwards and exits at once: the tunnel is
    up, the command looks like it failed, and ctrl-C closes nothing."""
    assert "-o ControlPath=none" in remote.tunnel()


def test_play_streams_rather_than_running_headless():
    cmd = remote.play("/workspace/logs/run/nn/last.pth")
    assert "--livestream 2" in cmd and "--viz kit" in cmd
    assert "--viz none" not in cmd


def test_commands_are_shell_quoted_for_nesting():
    """Three levels of nesting (ssh -> docker exec -> isaaclab.sh) means the
    inner command is quoted twice; unquoted, the first space ends it."""
    cmd = remote.smoke()
    assert cmd.startswith("ssh ") and "docker exec vscode bash -lc" in cmd


def test_the_scripted_gate_passes_its_knob_through():
    """--lead-time is the one real tuning knob on the scripted catcher, so a
    sweep must not silently run the default six times."""
    cmd = remote.scripted(num_envs=32, steps=500, lead_time=0.12)
    assert "scripts/scripted_catch.py" in cmd
    assert "--num_envs 32" in cmd and "--steps 500" in cmd and "--lead-time 0.12" in cmd


def test_a_watchable_run_streams_and_is_not_headless():
    """A headless run opens no Kit streaming ports, so the viewer shows
    nothing -- which is indistinguishable from a broken viewer."""
    cmd = remote.scripted(livestream=True)
    assert "--livestream 2" in cmd and "--viz kit" in cmd
    assert "--viz none" not in cmd


def test_the_default_scripted_run_is_still_headless():
    cmd = remote.scripted()
    assert "--viz none" in cmd and "--livestream" not in cmd


def test_the_tunnel_is_not_the_viewer_route():
    """WebRTC is UDP; ssh forwards TCP. The viewer goes through Brev's proxy."""
    assert "brevlab.com/viewer" in remote.VIEWER_URL
    assert "brevlab" not in remote.tunnel()
