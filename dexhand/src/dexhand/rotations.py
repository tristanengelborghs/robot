"""Quaternion utilities in MuJoCo's wxyz convention.

Every rotation bug in a reorientation task is silent: the sim keeps stepping,
the reward keeps flowing, and the policy learns *something* — just not the
task. Two failure modes motivate owning this module instead of borrowing one:

Convention. MuJoCo stores quaternions wxyz; scipy stores xyzw. Feeding one to
the other is not an error, it is a different rotation, and for small angles a
nearly-plausible one. Everything here is wxyz so that arrays can flow between
`data.qpos`/`data.xquat` and reward code without a reordering step anyone can
forget.

Double cover. q and -q are the same physical rotation, and MuJoCo will hand
you either depending on integration history. A distance that is not
sign-invariant reports up to pi of error between identical orientations, so a
policy rewarded with it learns to steer around half of quaternion space —
visible only as a mysteriously bimodal success rate. `rot_dist` folds the
cover; the test suite pins that property as a regression.

All functions take and return float64 ndarrays, accept a single quaternion
(4,) or any batch (..., 4), and broadcast like numpy ufuncs. `rng` arguments
are `np.random.Generator`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_normalize",
    "quat_mul",
    "quat_conj",
    "quat_rotate",
    "rot_dist",
    "quat_from_axis_angle",
    "axis_angle_from_quat",
    "quat_to_mat",
    "mat_to_quat",
    "random_quat",
    "random_z_quat",
]

# Below this, a quaternion's vector part carries no usable direction and the
# rotation is treated as identity (axis chosen arbitrarily).
_EPS = 1e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm. A zero quaternion yields nan — loudly."""
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b: the rotation b followed by a, both wxyz."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate; equals the inverse for unit quaternions."""
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector(s) v by unit quaternion(s) q; broadcasts (...,4)x(...,3)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    xyz = q[..., 1:]
    # Expansion of q v q* — two cross products instead of two full quaternion
    # products, and no intermediate quaternion whose w must be discarded.
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def rot_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic distance on SO(3) in radians, in [0, pi].

    The abs() is the whole point: it folds the double cover so q and -q are at
    distance zero. Without it, this exact formula returns pi for identical
    rotations whenever the signs disagree — a reward-shaping landmine.
    """
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = np.abs(np.sum(a * b, axis=-1))
    # Clip before arccos: unit-norm rounding puts |dot| a few ulp above 1.0,
    # and arccos of that is nan, not a small angle.
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def quat_from_axis_angle(axis: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    """Quaternion for a rotation of `angle` radians about `axis` (any norm)."""
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    # A zero axis only makes sense with a zero angle; the guard keeps that case
    # an exact identity instead of a 0/0 nan.
    unit = axis / np.maximum(norm, _EPS)
    half = angle[..., None] / 2.0
    w = np.cos(half)
    xyz = unit * np.sin(half)
    batch = np.broadcast_shapes(w.shape[:-1], xyz.shape[:-1])
    return np.concatenate(
        [np.broadcast_to(w, batch + (1,)), np.broadcast_to(xyz, batch + (3,))], axis=-1
    )


def axis_angle_from_quat(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of quat_from_axis_angle: (unit axis (...,3), angle (...,) in [0, pi]).

    Uses atan2 rather than arccos(w): arccos loses half the float precision
    near angle 0, exactly where retargeting deltas live. For angles below
    machine noise the axis is arbitrary and reported as +x.
    """
    q = quat_normalize(q)
    # Canonical sign (w >= 0) so the angle lands in [0, pi] instead of the
    # equivalent 2*pi - angle.
    q = q * np.where(q[..., :1] < 0.0, -1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, q[..., 0])
    small = sin_half < _EPS
    safe = np.where(small[..., None], 1.0, sin_half[..., None])
    axis = np.where(small[..., None], np.array([1.0, 0.0, 0.0]), xyz / safe)
    return axis, angle


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) for unit quaternion(s) q."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    row0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1)
    row1 = np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1)
    row2 = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Quaternion (w >= 0) for rotation matrix/matrices (..., 3, 3).

    Shepperd's branch-on-largest-diagonal method. The textbook one-liner
    w = sqrt(1+trace)/2 divides by w, which vanishes at 180 degrees — and
    180-degree flips are not a corner case in cube reorientation, they are the
    goal poses. Each branch here divides by the component it just took a
    sqrt of, which the branch condition guarantees is the largest.
    """
    m = np.asarray(m, dtype=np.float64)
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    # 4w^2, 4x^2, 4y^2, 4z^2 (up to shared normalization). All four branches
    # are computed and the best selected per element, because a batched input
    # can need a different branch per matrix.
    t = np.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        axis=-1,
    )
    # maximum(_, _EPS): the *discarded* branches may have t <= 0 from rounding;
    # without the floor they produce nan that np.where still evaluates.
    s = np.sqrt(np.maximum(t, _EPS))
    d = 1.0 / s

    cand = np.empty(t.shape[:-1] + (4, 4), dtype=np.float64)
    cand[..., 0, 0] = s[..., 0]
    cand[..., 0, 1] = (m21 - m12) * d[..., 0]
    cand[..., 0, 2] = (m02 - m20) * d[..., 0]
    cand[..., 0, 3] = (m10 - m01) * d[..., 0]

    cand[..., 1, 0] = (m21 - m12) * d[..., 1]
    cand[..., 1, 1] = s[..., 1]
    cand[..., 1, 2] = (m01 + m10) * d[..., 1]
    cand[..., 1, 3] = (m02 + m20) * d[..., 1]

    cand[..., 2, 0] = (m02 - m20) * d[..., 2]
    cand[..., 2, 1] = (m01 + m10) * d[..., 2]
    cand[..., 2, 2] = s[..., 2]
    cand[..., 2, 3] = (m12 + m21) * d[..., 2]

    cand[..., 3, 0] = (m10 - m01) * d[..., 3]
    cand[..., 3, 1] = (m02 + m20) * d[..., 3]
    cand[..., 3, 2] = (m12 + m21) * d[..., 3]
    cand[..., 3, 3] = s[..., 3]

    best = np.argmax(t, axis=-1)
    q = np.take_along_axis(cand, best[..., None, None], axis=-2)[..., 0, :]
    q = quat_normalize(q)
    return q * np.where(q[..., :1] < 0.0, -1.0, 1.0)


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """One quaternion uniform on SO(3), by Shoemake's subgroup-algorithm method.

    Three uniforms map exactly to Haar measure. Uniformity matters because
    goal orientations are sampled with this: a biased sampler concentrates
    training goals and the policy's success rate quietly becomes
    pose-dependent.
    """
    u1, u2, u3 = rng.uniform(size=3)
    a, b = np.sqrt(1.0 - u1), np.sqrt(u1)
    return np.array(
        [
            b * np.cos(2.0 * np.pi * u3),
            a * np.sin(2.0 * np.pi * u2),
            a * np.cos(2.0 * np.pi * u2),
            b * np.sin(2.0 * np.pi * u3),
        ]
    )


def random_z_quat(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    """Rotation about world z, angle uniform in [-max_angle, max_angle].

    The curriculum's opening move: cube reorientation goals restricted to yaw
    are reachable without regrasping, so early training gets reward signal
    before full-SO(3) goals are on the menu.
    """
    angle = rng.uniform(-max_angle, max_angle)
    return quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), angle)
