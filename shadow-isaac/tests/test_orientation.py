"""The palm-up orientation, pinned to the search that found it.

`make palmup` writes all 24 axis-aligned orientations into a running simulator
and reads back where the palm points. These are its results (Isaac Lab 3.0.0,
2026-08-30, cupped reset pose). They are data, not derivation: two attempts to
compose the rotation analytically put the hand somewhere the arithmetic did not
predict, and the second of those produced a hand lying on its side while the
scripted catcher reported 100% caught.
"""

import pathlib

import numpy as np
import pytest

CFG = (pathlib.Path(__file__).resolve().parent.parent
       / "source/catching/catching/tasks/direct/catch/catch_env_cfg.py")

# rot (wxyz) -> (palm normal, finger direction), measured.
MEASURED = {
    (0.707107, 0.0, 0.0, -0.707107): ([0.0, -0.007, 1.0], [0.047, 0.999, 0.007]),
    (0.0, -0.707107, 0.707107, 0.0): ([0.0, 0.007, 1.0], [-0.047, -0.999, 0.007]),
    (-0.5, 0.5, -0.5, 0.5): ([-0.007, 0.0, 1.0], [0.999, -0.047, 0.007]),
    (-0.5, -0.5, 0.5, 0.5): ([0.007, 0.0, 1.0], [-0.999, 0.047, 0.007]),
    # The one that was in the config while the hand lay on its side.
    (-0.5, -0.5, 0.5, -0.5): ([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]),
}
CHOSEN = (0.707107, 0.0, 0.0, -0.707107)


def tilt(normal):
    return np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))


def test_the_chosen_orientation_is_palm_up():
    assert tilt(MEASURED[CHOSEN][0]) < 1.0


def test_the_chosen_orientation_leaves_the_fingers_horizontal():
    """Palm up with the fingers still vertical would be a fist, not a basket."""
    assert abs(MEASURED[CHOSEN][1][2]) < 0.05


def test_the_previous_value_was_a_hand_on_its_side():
    """Regression: this shipped, and the catch metric read 100% anyway."""
    assert tilt(MEASURED[(-0.5, -0.5, 0.5, -0.5)][0]) == pytest.approx(90.0, abs=1.0)


def test_the_config_uses_the_chosen_orientation():
    assert f"hand_rot: tuple = {CHOSEN}" in CFG.read_text()


def test_the_orientation_is_applied_at_reset_not_at_spawn():
    """`init_state.rot` is composed with the asset's own USD transform, so the
    quaternion that gives palm-up through `write_root_pose_to_sim` leaves the
    hand 90 deg off through init_state. `make palmup` validates the runtime
    path, so the env must use the runtime path -- setting it at spawn is how
    the searched value still produced a hand on its side."""
    env = (CFG.parent / "catch_env.py").read_text()
    assert "self.hand.write_root_pose_to_sim(root, env_ids=env_ids)" in env
    assert "init_state=SHADOW_HAND_CFG.init_state.replace(rot=" not in CFG.read_text()


def test_the_env_refuses_a_hand_on_its_side():
    """A green metric is not evidence the scene is right, so the env checks."""
    env = (CFG.parent / "catch_env.py").read_text()
    assert "_check_palm_is_up" in env
    assert "max_palm_tilt_deg" in CFG.read_text()
