"""Landmarks -> a 6-DoF pose, a scale, and a grip signal.

This is where 21 loose points become something a robot action can be computed from.
Three quantities come out, and each is deliberately built from the part of the hand
that can support it:

    orientation   from the five palm landmarks only, by Kabsch alignment to a
                  canonical template. Fingers are excluded because a fist and a
                  flat hand are the same pose and must produce the same frame.
    scale         from the wrist->knuckle distances. Divides out how large the
                  person's hand is and how near the camera it is, so a child and an
                  adult performing the same demo retarget to the same trajectory.
    aperture      from thumb-tip to index-tip, in units of hand scale. The one
                  quantity that is *supposed* to come from the fingers.

Everything is scale-free by the time it leaves this module except the wrist
translation, which is in whatever metric units the detector reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from robobench.hand.landmarks import (
    CANONICAL_SCALE,
    INDEX_TIP,
    MCP_IDS,
    NUM_LANDMARKS,
    PALM_IDS,
    THUMB_TIP,
    WRIST,
    canonical_palm,
)

__all__ = ["HandPose", "hand_scale", "palm_frame", "grip_aperture", "finger_curls", "estimate_pose"]

# Thumb-to-index distance in hand-scale units at a hard pinch and at a wide spread.
# Measured across adult hands; `calibrate_aperture` replaces them per-operator when
# you have a session to calibrate on.
APERTURE_PINCHED = 0.20
APERTURE_OPEN = 1.00


@dataclass
class HandPose:
    """A hand reduced to what retargeting needs."""

    position: np.ndarray  # (3,) wrist origin, metres, detector frame
    rotation: np.ndarray  # (3, 3) hand frame -> detector frame, orthonormal, det=+1
    aperture: float  # 0 = pinched shut, 1 = wide open
    scale: float  # metres per canonical hand
    curls: np.ndarray  # (5,) per-finger curl, 0 = straight, 1 = fully folded
    frame_confidence: float  # 0..1; collapses when the palm is edge-on to the camera
    handedness: str = "right"

    def as_vector(self) -> np.ndarray:
        """(7,) = wrist xyz | rotation as a rotation vector | hand scale.

        This is what gets written into the demo file's proprioception slot. It is
        *not* robot joint angles, and `robobench.hand.record` labels the file so the
        two can never be silently mixed.
        """
        return np.concatenate([self.position, rotation_vector(self.rotation), [self.scale]]).astype(np.float32)


def hand_scale(keypoints: np.ndarray) -> float:
    """Metres per canonical hand: median wrist->MCP distance over the four knuckles.

    Median rather than mean because a single badly-placed knuckle is a routine
    detector failure and would otherwise rescale the whole hand — which, since scale
    divides into the aperture, reads downstream as a phantom grasp.
    """
    kp = _check(keypoints)
    d = np.linalg.norm(kp[list(MCP_IDS)] - kp[WRIST], axis=-1)
    return float(np.median(d))


def palm_frame(keypoints: np.ndarray, handedness: str = "right") -> Tuple[np.ndarray, float]:
    """Least-squares hand orientation from the palm. Returns (R, confidence).

    R takes a vector in the canonical hand frame to the detector frame, so its
    columns are the hand's x/y/z axes as the camera sees them.

    Kabsch rather than the usual cross-product recipe (x = wrist->index, z = x cross
    wrist->pinky, y = z cross x). That recipe is exact, which is the problem: it puts
    the entire frame on three landmarks and hands any error on the pinky knuckle
    straight to the palm normal. Kabsch spreads the fit over all five palm points, so
    one bad landmark moves the frame by a fraction of its own error.

    The template is planar, so the fit determines the two in-plane axes and completes
    the third by the determinant correction. That is well conditioned right up until
    the palm turns edge-on to the camera and the in-plane axes project onto a line.

    Confidence is the residual of the fit — the RMS distance between the observed
    palm and the best-fitting rigid template, in units of hand scale — mapped to
    [0, 1]. Note what it is *not*: the obvious cheap proxy, the ratio of the fit's
    second to first singular value, is not monotone in foreshortening. A palm tilted
    60 degrees away from the camera projects to a patch that is more nearly square
    than a head-on palm is, so that ratio *improves* as the pose gets worse. The
    residual has no such failure — a rigid palm at any orientation fits exactly, and
    anything that cannot be explained by a rigid palm shows up as error. Below ~0.5
    the roll about the palm normal is guesswork; the tracker gates on this rather
    than emitting a confident wrong orientation.
    """
    kp = _check(keypoints)
    observed = kp[list(PALM_IDS)]
    scale = hand_scale(kp)
    template = canonical_palm(handedness) * (scale / CANONICAL_SCALE)

    a = template - template.mean(axis=0)
    b = observed - observed.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)

    # Reflections fit point sets just as well as rotations and are not poses.
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T

    if scale <= 1e-9:
        return r, 0.0
    rms = float(np.sqrt((np.linalg.norm(b - a @ r.T, axis=-1) ** 2).mean())) / scale
    return r, float(np.clip(1.0 - rms / _RESIDUAL_AT_ZERO_CONFIDENCE, 0.0, 1.0))


#: Kabsch residual, in units of hand scale, at which the palm frame is worthless.
#: A residual this large means the five palm points cannot be explained by a rigid
#: palm at any orientation — foreshortened to a line, or a landmark badly misplaced.
_RESIDUAL_AT_ZERO_CONFIDENCE = 0.25


def grip_aperture(
    keypoints: np.ndarray,
    scale: float | None = None,
    pinched: float = APERTURE_PINCHED,
    open_: float = APERTURE_OPEN,
) -> float:
    """Thumb-to-index distance in hand-scale units, mapped to [0, 1].

    Normalizing by hand scale is what makes this transfer between people and across
    distance from the camera. In raw metres the same gesture reads as a different
    aperture from a metre away than from half a metre, and the policy inherits a
    gripper command that tracks how far the demonstrator stood from the lens.
    """
    kp = _check(keypoints)
    s = hand_scale(kp) if scale is None else scale
    if s <= 1e-9:
        return 1.0
    ratio = float(np.linalg.norm(kp[THUMB_TIP] - kp[INDEX_TIP]) / s)
    return float(np.clip((ratio - pinched) / max(open_ - pinched, 1e-6), 0.0, 1.0))


def calibrate_aperture(apertures_ratio: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0):
    """Per-operator pinch/open bounds from a recorded session.

    Percentiles, not min/max: the extremes of a real session are detector failures,
    and calibrating on them compresses every genuine grasp into the middle of the
    range where the gripper never fully commits.
    """
    r = np.asarray(apertures_ratio, dtype=np.float64)
    return float(np.percentile(r, lo_pct)), float(np.percentile(r, hi_pct))


def finger_curls(keypoints: np.ndarray) -> np.ndarray:
    """(5,) curl per finger: 1 - (tip-to-MCP distance / summed bone length).

    Straight finger -> ~0, folded finger -> ~1. Not used by the default retargeting
    (a parallel-jaw gripper has one degree of freedom) but written into the demo so a
    multi-finger or dexterous-hand retargeting has something to consume.
    """
    from robobench.hand.landmarks import FINGERS

    kp = _check(keypoints)
    out = []
    for chain in FINGERS.values():
        pts = kp[list(chain)]
        chord = np.linalg.norm(pts[-1] - pts[0])
        arc = np.linalg.norm(np.diff(pts, axis=0), axis=-1).sum()
        out.append(1.0 - chord / arc if arc > 1e-9 else 0.0)
    return np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)


def estimate_pose(
    keypoints: np.ndarray,
    handedness: str = "right",
    pinched: float = APERTURE_PINCHED,
    open_: float = APERTURE_OPEN,
) -> HandPose:
    """Full pose from one frame of landmarks."""
    kp = _check(keypoints)
    scale = hand_scale(kp)
    rot, conf = palm_frame(kp, handedness)
    return HandPose(
        position=kp[WRIST].astype(np.float64),
        rotation=rot,
        aperture=grip_aperture(kp, scale, pinched, open_),
        scale=scale,
        curls=finger_curls(kp),
        frame_confidence=conf,
        handedness=handedness,
    )


# -- small rotation helpers -------------------------------------------------
# scipy.spatial.transform would do all of this, and is already a dependency, but
# these three are short enough that keeping the hand package importable without
# scipy is worth more than the reuse. `retarget` does use scipy, for Slerp.


def rotation_vector(r: np.ndarray) -> np.ndarray:
    """SO(3) matrix -> axis-angle vector (magnitude = angle in radians)."""
    r = np.asarray(r, dtype=np.float64)
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-8:
        return np.zeros(3)
    if angle > np.pi - 1e-6:
        # Near pi the skew part vanishes; recover the axis from R + I, whose
        # columns are all parallel to it, taking the best-conditioned one.
        a = (r + np.eye(3)) / 2.0
        axis = a[:, int(np.argmax(np.diag(a)))]
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.zeros(3)
        return axis / n * angle
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return axis / (2.0 * np.sin(angle)) * angle


def rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> SO(3) matrix (Rodrigues)."""
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)


def orthonormalize(r: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix, for cleaning up accumulated numerical drift."""
    u, _, vt = np.linalg.svd(np.asarray(r, dtype=np.float64))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def _check(keypoints: np.ndarray) -> np.ndarray:
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected keypoints of shape (21, 3), got {kp.shape}")
    return kp
