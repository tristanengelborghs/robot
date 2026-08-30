"""The nesting is ssh -> docker exec -> isaaclab.sh, and each layer quotes the
next. These tests pin the two invariants that cost real GPU time to discover."""

from pathlib import Path

import pytest

from harness import remote


def test_headless_run_always_disables_visualizers():
    # Without --viz none the demo scripts default to the Kit visualizer and
    # crash on a display-less host.
    cmd = remote.isaaclab("scripts/environments/random_agent.py")
    assert "--headless" in cmd
    assert "--viz none" in cmd


def test_livestream_run_is_not_headless():
    cmd = remote.isaaclab("scripts/environments/random_agent.py", headless=False, livestream=True)
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


def test_record_runs_from_this_project_not_the_isaaclab_tree():
    # scripts/record_scripted.py imports harness.stack_sm, so it has to run with
    # this project as the working directory -- and therefore needs an absolute
    # path to isaaclab.sh.
    cmd = remote.record_scripted(num_demos=5)
    assert f"cd {remote.WORKDIR}" in cmd
    assert f"{remote.ISAACLAB}/isaaclab.sh -p scripts/record_scripted.py" in cmd
    assert "--num_demos 5" in cmd
    assert remote.TELEOP_TASK in cmd


def test_record_does_not_pass_num_envs():
    # record_scripted.py fixes num_envs at 1 and never defines the flag;
    # argparse rejects arguments it has never heard of.
    assert "--num_envs" not in remote.record_scripted()


def test_record_is_headless_with_visualizers_off():
    cmd = remote.record_scripted()
    assert "--headless" in cmd
    assert "--viz none" in cmd


def test_replay_validates_states_in_a_single_environment():
    # "Done when: a replayed episode reproduces the original trajectory" is
    # exactly what --validate_states checks, and it is only valid for one env.
    cmd = remote.replay()
    assert "replay_demos.py" in cmd
    assert "--validate_states" in cmd
    assert "--num_envs 1" in cmd


def test_record_and_replay_agree_on_the_dataset_path():
    assert remote.DATASET_FILE in remote.record_scripted()
    assert remote.DATASET_FILE in remote.replay()


def test_num_envs_can_be_omitted():
    assert "--num_envs" not in remote.isaaclab("scripts/x.py", num_envs=None)
    assert "--num_envs 8" in remote.isaaclab("scripts/x.py", num_envs=8)


def test_container_patch_runs_the_synced_patch_script():
    # Isaac Lab 3.0.0 ships a Franka USD path NVIDIA has since moved, and a
    # replay script that cannot start headless. Its own scripts hit both, so the
    # fixes go into the container's Isaac Lab rather than our environment config.
    cmd = remote.patch_container()
    assert f"{remote.WORKDIR}/scripts/patch_container.py" in cmd
    assert cmd.startswith("ssh ")


def test_container_patch_falls_back_to_isaac_sims_python():
    # The container has no system Python: `python3` is not on the PATH there and
    # running the patch with it exits 127.
    cmd = remote.patch_container()
    assert "command -v python3" in cmd
    assert remote.ISAAC_SIM_PYTHON in cmd


def test_datasets_are_recorded_outside_the_synced_directory():
    # `make sync` deletes WORKDIR on the box before copying a fresh copy in, so
    # anything recorded under it is destroyed by the next sync. This cost one
    # full set of recorded demonstrations.
    assert not remote.DATASET_FILE.startswith(remote.WORKDIR)
    assert remote.DATASET_FILE.startswith(remote.DATASET_DIR)


def test_teleop_streams_because_a_keyboard_needs_a_window():
    # A keyboard device attaches to an application window, so a teleoperated
    # recording cannot be headless -- it streams to a browser instead.
    cmd = remote.record_teleop()
    assert "record_demos.py" in cmd
    assert "--livestream 2" in cmd
    assert "--headless" not in cmd
    assert "--viz none" not in cmd


def test_teleop_forces_the_legacy_device_path():
    # Without an explicit --teleop_device, record_demos.py prefers the
    # IsaacTeleop/CloudXR pipeline when the task configures one, which wants a
    # VR headset rather than a keyboard.
    assert "--teleop_device keyboard" in remote.record_teleop()
    assert "--teleop_device spacemouse" in remote.record_teleop(device="spacemouse")


def test_teleop_records_to_its_own_dataset():
    # Hand-driven demos must not be written over the scripted ones, and both
    # live outside the directory `make sync` deletes.
    assert remote.TELEOP_DATASET_FILE != remote.DATASET_FILE
    assert remote.TELEOP_DATASET_FILE.startswith(remote.DATASET_DIR)
    assert remote.TELEOP_DATASET_FILE in remote.record_teleop()


def test_teleop_does_not_pass_num_envs():
    assert "--num_envs" not in remote.record_teleop()


def test_tunnel_forwards_the_viewer_and_its_signalling_port():
    # Only port 22 is open on the instance, so both the viewer page and the
    # WebRTC signalling reach the laptop through ssh.
    cmd = remote.tunnel()
    assert f"-L {remote.LOCAL_VIEWER_PORT}:localhost:{remote.VIEWER_PORT}" in cmd
    assert f"-L {remote.WEBRTC_PORT}:localhost:{remote.WEBRTC_PORT}" in cmd
    assert " -N " in cmd


def test_tunnel_opts_out_of_ssh_connection_multiplexing():
    # Brev's ssh config enables multiplexing, and against an existing master
    # `ssh -N` registers the forwards and exits at once: the tunnel is up, the
    # command looks like it failed, and ctrl-C closes nothing.
    cmd = remote.tunnel()
    assert "ControlPath=none" in cmd
    assert "ExitOnForwardFailure=yes" in cmd


def test_viewer_url_matches_the_tunnel():
    assert str(remote.LOCAL_VIEWER_PORT) in remote.viewer_url()
    assert remote.viewer_url().endswith("/viewer/")


def test_the_container_side_script_uses_the_same_flags():
    """The in-container script and harness.remote must not drift apart.

    scripts/teleop_in_container.sh exists because `make teleop` cannot run on
    the box -- it is the outside half of the pair. That means the teleop flags
    are written down twice, so this checks the second copy still matches the
    first. A silent divergence would record a subtly different dataset
    depending on which terminal it was started from.
    """
    script = (Path(__file__).resolve().parent.parent / "scripts/teleop_in_container.sh").read_text()
    built = remote.record_teleop()

    for flag in ("--livestream 2", "--viz kit", remote.TELEOP_TASK):
        assert flag in script, f"{flag} is in the built command but not in the script"
        assert flag in built

    # The script takes its device as a parameter, so only the flag is literal.
    assert "--teleop_device" in script
    assert "--teleop_device" in built

    # Both must reach the same script, and neither may be headless.
    assert "scripts/tools/record_demos.py" in script
    assert "--headless" not in script
    assert remote.TELEOP_DATASET_FILE in script


def test_the_two_sides_accept_the_same_teleop_devices():
    """The container script and the laptop CLI must offer the same devices.

    A device accepted by one and rejected by the other is a confusing failure:
    the same request works or does not depending on which terminal it is typed
    into.
    """
    script = (Path(__file__).resolve().parent.parent / "scripts/teleop_in_container.sh").read_text()
    cli = (Path(__file__).resolve().parent.parent / "scripts/run_remote.py").read_text()

    for device in ("keyboard", "gamepad", "spacemouse", "network"):
        assert device in script, f"{device} is offered by the CLI but not by the container script"
        assert device in cli, f"{device} is offered by the container script but not by the CLI"


def test_teleop_can_ask_for_the_network_device():
    # The laptop-side link, for when NVIDIA's stream will not carry a gamepad.
    assert "--teleop_device network" in remote.record_teleop(device="network")


def test_teleop_can_ask_for_a_gamepad():
    # NVIDIA's WebRTC client forwards gamepad input to the streamed app, with a
    # DualSense profile among others, so a controller on the laptop reaches the
    # simulator on the box.
    assert "--teleop_device gamepad" in remote.record_teleop(device="gamepad")


def test_kill_targets_every_simulator_entry_point_by_default():
    """`make kill` must not know about only one script.

    It used to target random_agent.py alone, left over from when that was the
    only thing that ran. A teleoperation session left running because the
    cleanup had never heard of it bills exactly as much as one nobody tried to
    stop.
    """
    cmd = remote.kill_sim()
    for script in ("record_demos.py", "record_scripted.py", "replay_demos.py", "random_agent.py"):
        assert script in cmd, f"{script} can start a simulator but kill_sim ignores it"


def test_kill_can_still_target_one_script():
    # Scripts cleaning up after their own run want to kill only their own.
    cmd = remote.kill_sim("replay_demos.py")
    assert "replay_demos.py" in cmd
    assert "record_demos.py" not in cmd


def test_kill_tolerates_nothing_matching():
    # pkill exits non-zero when it matches nothing, which is the normal case for
    # most of these names and must not read as a failure.
    assert "|| true" in remote.kill_sim()
