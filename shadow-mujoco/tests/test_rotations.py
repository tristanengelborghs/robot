"""Regression suite for dexhand.rotations.

Rotation math fails silently: a wrong convention or a dropped sign still
produces unit quaternions and orthonormal matrices, so nothing crashes — the
reward is just wrong and the policy trains on the wrong objective. Each test
below is named for the algebraic property it pins and its docstring says what
breaks, quietly, without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from dexhand.rotations import (
    axis_angle_from_quat,
    mat_to_quat,
    quat_conj,
    quat_from_axis_angle,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_to_mat,
    random_quat,
    random_z_quat,
    rot_dist,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _batch(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.stack([random_quat(rng) for _ in range(n)])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def test_mul_conj_is_identity(rng):
    """q ⊗ q* must be the identity; breaks if mul and conj disagree on the
    wxyz layout, which every downstream relative-orientation computation
    (goal error, object delta) silently inherits."""
    q = _batch(rng, 64)
    prod = quat_mul(q, quat_conj(q))
    np.testing.assert_allclose(prod, np.broadcast_to(IDENTITY, prod.shape), atol=1e-12)


def test_mul_composes_like_matrices(rng):
    """quat_mul(a, b) must equal mat(a) @ mat(b); breaks if the Hamilton
    product is transposed, which flips every composed rotation's direction."""
    a, b = random_quat(rng), random_quat(rng)
    np.testing.assert_allclose(
        quat_to_mat(quat_mul(a, b)), quat_to_mat(a) @ quat_to_mat(b), atol=1e-12
    )


def test_rotate_agrees_with_matrix(rng):
    """quat_rotate must equal quat_to_mat(q) @ v; breaks if the cross-product
    expansion drops a term — the result stays unit-preserving-ish and
    plausible, so only this cross-check catches it."""
    q = _batch(rng, 32)
    v = rng.normal(size=(32, 3))
    expected = np.einsum("...ij,...j->...i", quat_to_mat(q), v)
    np.testing.assert_allclose(quat_rotate(q, v), expected, atol=1e-12)


def test_rotate_known_value():
    """90 degrees about z must send +x to +y; breaks on a handedness or
    convention flip that all the self-consistency tests above would miss
    because both sides of them would flip together."""
    q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    np.testing.assert_allclose(quat_rotate(q, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-12)


def test_axis_angle_round_trip(rng):
    """axis/angle -> quat -> axis/angle must return the input; breaks if the
    two functions disagree on half-angle handling, skewing every goal sampled
    or logged in axis-angle form."""
    axis = quat_rotate(_batch(rng, 16), np.array([1.0, 0.0, 0.0]))
    angle = rng.uniform(0.1, np.pi - 0.1, size=16)
    axis_out, angle_out = axis_angle_from_quat(quat_from_axis_angle(axis, angle))
    np.testing.assert_allclose(angle_out, angle, atol=1e-10)
    np.testing.assert_allclose(axis_out, axis, atol=1e-9)


@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-9, 1e-7])
def test_axis_angle_round_trip_near_zero(angle):
    """Tiny angles must survive the round trip without nan; breaks if the
    small-angle guard divides by sin(angle/2) ~ 0 — and near-zero deltas are
    the common case for a filtered pose stream, not the rare one."""
    axis = np.array([0.0, 1.0, 0.0])
    q = quat_from_axis_angle(axis, angle)
    assert np.all(np.isfinite(q))
    axis_out, angle_out = axis_angle_from_quat(q)
    assert np.all(np.isfinite(axis_out))
    np.testing.assert_allclose(angle_out, angle, atol=1e-10)


@pytest.mark.parametrize("angle", [np.pi - 1e-7, np.pi])
def test_axis_angle_round_trip_near_pi(angle):
    """Angles at and near pi must survive the round trip; breaks for
    arccos-based angle recovery, and pi flips are cube goal poses, not edge
    cases."""
    axis = np.array([1.0, 0.0, 0.0])
    q_in = quat_from_axis_angle(axis, angle)
    axis_out, angle_out = axis_angle_from_quat(q_in)
    np.testing.assert_allclose(angle_out, angle, atol=1e-9)
    q_out = quat_from_axis_angle(axis_out, angle_out)
    assert rot_dist(q_in, q_out) < 1e-8


def test_mat_quat_round_trip(rng):
    """quat -> mat -> quat must return the same rotation; breaks if either
    direction transposes, which self-cancels in mat-space tests but corrupts
    any quaternion read back from MuJoCo's xmat."""
    q = _batch(rng, 128)
    assert np.max(rot_dist(mat_to_quat(quat_to_mat(q)), q)) < 1e-10


@pytest.mark.parametrize("axis_idx", [0, 1, 2])
def test_mat_quat_round_trip_180_degrees(axis_idx):
    """180-degree flips about each axis must round-trip exactly; breaks for
    the naive trace formula (divides by w = 0 there), and these are precisely
    the goal orientations of a cube-flipping curriculum."""
    axis = np.zeros(3)
    axis[axis_idx] = 1.0
    q = quat_from_axis_angle(axis, np.pi)
    q_back = mat_to_quat(quat_to_mat(q))
    assert np.all(np.isfinite(q_back))
    assert rot_dist(q_back, q) < 1e-12


def test_rot_dist_of_identical_quats_is_zero(rng):
    """rot_dist(q, q) must be 0 despite |dot| rounding above 1; breaks
    without the clip before arccos, injecting nan into the reward."""
    q = _batch(rng, 64)
    np.testing.assert_allclose(rot_dist(q, q), 0.0, atol=1e-6)


def test_rot_dist_sign_invariance_regression(rng):
    """THE double-cover regression: rot_dist(q, -q) must be 0, not pi.
    Without abs() the reward punishes half of quaternion space for
    orientations identical to the goal, and the policy learns to avoid the
    'wrong' sign instead of reorienting the cube."""
    q = _batch(rng, 64)
    np.testing.assert_allclose(rot_dist(q, -q), 0.0, atol=1e-6)


def test_rot_dist_symmetry(rng):
    """d(a, b) must equal d(b, a); breaks if a future 'optimization'
    normalizes only one argument, making the reward depend on argument order."""
    a, b = _batch(rng, 64), _batch(rng, 64)
    np.testing.assert_allclose(rot_dist(a, b), rot_dist(b, a), atol=1e-12)


def test_rot_dist_triangle_inequality(rng):
    """d(a, c) <= d(a, b) + d(b, c) on random triples; breaks if the folded
    angle is mapped back incorrectly (e.g. missing the factor 2), in which
    case rot_dist is no longer a metric and distance-based curricula misorder
    goals."""
    a, b, c = _batch(rng, 256), _batch(rng, 256), _batch(rng, 256)
    assert np.all(rot_dist(a, c) <= rot_dist(a, b) + rot_dist(b, c) + 1e-9)


def test_rot_dist_max_is_pi(rng):
    """The distance must peak at exactly pi (a 180-degree flip) and never
    exceed it; breaks if the double cover is unfolded, which doubles the
    range to 2*pi and re-scales any reward normalized by the maximum."""
    flip = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi)
    np.testing.assert_allclose(rot_dist(IDENTITY, flip), np.pi, atol=1e-9)
    a, b = _batch(rng, 512), _batch(rng, 512)
    assert np.all(rot_dist(a, b) <= np.pi + 1e-12)


def test_random_quat_uniformity(rng):
    """Mean pairwise rot_dist of uniform SO(3) samples is pi/2 + 2/pi
    (~2.207); breaks if Shoemake's radicals are swapped or a component is
    dropped, which biases goal sampling toward a subset of orientations and
    silently makes success rate pose-dependent."""
    q = _batch(rng, 2000)
    d = 2.0 * np.arccos(np.clip(np.abs(q @ q.T), -1.0, 1.0))
    mean = d[~np.eye(len(q), dtype=bool)].mean()
    assert 2.1 < mean < 2.35
    # Marginal sanity: each component of a uniform quaternion has mean 0.
    assert np.all(np.abs(q.mean(axis=0)) < 0.05)


def test_random_z_quat_is_a_z_rotation(rng):
    """random_z_quat must leave world z fixed; breaks if the axis is wrong or
    unnormalized, turning the 'easy yaw-only curriculum' into out-of-plane
    goals the early policy cannot reach without regrasping."""
    for _ in range(50):
        q = random_z_quat(rng, max_angle=2.0)
        np.testing.assert_allclose(quat_rotate(q, np.array([0.0, 0.0, 1.0])), [0.0, 0.0, 1.0], atol=1e-12)


def test_random_z_quat_angle_bounded(rng):
    """Sampled yaw must stay within [-max_angle, max_angle]; breaks if the
    half-angle is applied twice (or not at all), silently doubling or halving
    the advertised curriculum difficulty."""
    max_angle = 0.5
    for _ in range(200):
        _, angle = axis_angle_from_quat(random_z_quat(rng, max_angle))
        assert angle <= max_angle + 1e-12


def test_batched_shapes(rng):
    """Every function must preserve arbitrary batch shapes; breaks if an op
    reduces over the wrong axis — which usually still broadcasts into a
    wrong-but-valid shape and corrupts values downstream instead of raising."""
    q = _batch(rng, 12).reshape(2, 3, 2, 4)
    v = rng.normal(size=(2, 3, 2, 3))
    assert quat_normalize(q).shape == (2, 3, 2, 4)
    assert quat_mul(q, q).shape == (2, 3, 2, 4)
    assert quat_conj(q).shape == (2, 3, 2, 4)
    assert quat_rotate(q, v).shape == (2, 3, 2, 3)
    assert rot_dist(q, q).shape == (2, 3, 2)
    assert quat_to_mat(q).shape == (2, 3, 2, 3, 3)
    assert mat_to_quat(quat_to_mat(q)).shape == (2, 3, 2, 4)
    axis, angle = axis_angle_from_quat(q)
    assert axis.shape == (2, 3, 2, 3) and angle.shape == (2, 3, 2)
    assert quat_from_axis_angle(axis, angle).shape == (2, 3, 2, 4)
    # Scalar angle against batched axes must broadcast, not raise.
    assert quat_from_axis_angle(axis, 0.3).shape == (2, 3, 2, 4)
