"""Reading Isaac Lab's HDF5 demo files, against a file we wrote ourselves.

The fixture below is built to match ``HDF5DatasetFileHandler.write_episode``
(vendored at reference/isaaclab_source/), attribute for attribute. That is the
point: if Isaac Lab changes the layout, these tests keep passing against the old
one, and the mismatch shows up the first time a real dataset is inspected. What
they do pin is our side -- that episodes are ordered numerically rather than
lexically, that a missing end-effector term is an error with a useful message
rather than a plot of the wrong quantity, and that the quaternion convention is
read from the file instead of assumed.
"""

import json

import h5py
import numpy as np
import pytest

from harness import dataset


def write_dataset(path, *, episodes=3, steps=12, format_version=1, ee_key="eef_pos", env_name="Isaac-Stack-Cube-v0"):
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = format_version
        data = handle.create_group("data")
        data.attrs["total"] = episodes
        data.attrs["env_args"] = json.dumps({"env_name": env_name, "type": 2})

        for index in range(episodes):
            length = steps + index  # episodes are not all the same length
            group = data.create_group(f"demo_{index}")
            group.attrs["num_samples"] = length
            group.attrs["success"] = True
            group.attrs["seed"] = 100 + index
            group.create_dataset("actions", data=np.zeros((length, 7), dtype=np.float32))
            obs = group.create_group("obs")
            ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
            obs.create_dataset(ee_key, data=np.stack([ramp, ramp * 2, ramp * 3], axis=1))
            obs.create_dataset("eef_quat", data=np.zeros((length, 4), dtype=np.float32))
            obs.create_dataset("gripper_pos", data=np.zeros((length, 2), dtype=np.float32))
            group.create_group("initial_state").create_dataset("dummy", data=np.zeros(3))
    return path


@pytest.fixture
def demo_file(tmp_path):
    return write_dataset(tmp_path / "demos.hdf5")


def test_summary_reports_shapes_and_metadata(demo_file):
    summary = dataset.summarize(demo_file)

    assert summary.env_name == "Isaac-Stack-Cube-v0"
    assert summary.format_version == 1
    assert len(summary.episodes) == 3

    first = summary.episodes[0]
    assert first.name == "demo_0"
    assert first.num_samples == 12
    assert first.action_shape == (12, 7)
    assert first.obs_shapes["eef_pos"] == (12, 3)
    assert first.obs_shapes["gripper_pos"] == (12, 2)
    assert first.success is True
    assert first.seed == 100


def test_episodes_are_ordered_numerically(tmp_path):
    # h5py hands back keys in lexical order, where demo_10 precedes demo_2.
    summary = dataset.summarize(write_dataset(tmp_path / "many.hdf5", episodes=12))
    assert [ep.name for ep in summary.episodes] == [f"demo_{i}" for i in range(12)]


def test_quaternion_convention_comes_from_the_file(tmp_path):
    modern = dataset.summarize(write_dataset(tmp_path / "v1.hdf5", format_version=1))
    legacy = dataset.summarize(write_dataset(tmp_path / "v0.hdf5", format_version=0))
    assert modern.quaternion_order == "xyzw"
    assert "wxyz" in legacy.quaternion_order


def test_end_effector_positions_come_back_as_a_trajectory(demo_file):
    name, values = dataset.ee_positions(demo_file, "demo_1")
    assert name == "eef_pos"
    assert values.shape == (13, 3)
    # The fixture ramps each axis at a different rate.
    assert values[-1] == pytest.approx([1.0, 2.0, 3.0])


def test_an_unfamiliar_task_fails_loudly(tmp_path):
    path = write_dataset(tmp_path / "other.hdf5", ee_key="tcp_position")
    with pytest.raises(KeyError) as excinfo:
        dataset.ee_positions(path, "demo_0")
    # The message has to name what was actually there, or the next step is
    # guessing at an unfamiliar dataset.
    assert "tcp_position" in str(excinfo.value)


def test_an_explicit_key_overrides_the_search(tmp_path):
    path = write_dataset(tmp_path / "other.hdf5", ee_key="tcp_position")
    name, values = dataset.ee_positions(path, "demo_0", key="tcp_position")
    assert name == "tcp_position"
    assert values.shape == (12, 3)


def test_summary_formats_without_crashing(demo_file):
    text = dataset.summarize(demo_file).format()
    assert "demo_0" in text
    assert "actions (12, 7)" in text
    assert "obs/eef_pos (12, 3)" in text
