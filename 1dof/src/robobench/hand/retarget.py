"""Hand trajectory -> the 7-D action LIBERO's controller actually accepts.

The target is robosuite's OSC_POSE, which LIBERO drives at 20 Hz:

    a[0:3]  end-effector translation delta, normalized; +-1 maps to +-0.05 m
    a[3:6]  end-effector rotation delta as an axis-angle vector; +-1 maps to +-0.5 rad
    a[6]    gripper; -1 opens, +1 closes

Four things have to be right, and each is a place where a plausible-looking demo
turns out to be silently wrong:

Deltas, not poses.  The human's hand and the robot's gripper do not share an origin,
    a workspace, or a scale. Absolute positions would need a calibrated
    human-to-robot transform that is wrong the moment either the camera or the robot
    moves. Differences cancel the origin entirely — only the *shape* of the motion
    has to transfer, which is the part that is actually reproducible.

Frames.  Camera axes are +x right, +y down, +z into the scene. Robot base axes are
    +x forward, +y left, +z up. Rotations transform by conjugation, not by
    multiplication: R_robot = C R_cam C^T. Getting this wrong is the classic silent
    failure — the positions look right, the rotations are a mirrored mess, and
    nothing in the pipeline complains.

Rate.  Video arrives at 30 or 60 fps; the controller runs at 20 Hz. Retargeting
    frame-to-frame and letting the simulator consume it at its own rate stretches
    every demo by the ratio of the two, so a policy trained on it moves at the wrong
    speed. Resampling onto the control clock first is not optional. Rotations are
    resampled by Slerp — componentwise interpolation of rotation vectors is not a
    rotation, and near a half turn it is not even close.

Deadband.  A hand at rest still jitters. Without a floor, a quarter of a demo is
    action noise with no visual cause, and behaviour cloning fits it faithfully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from robobench.hand.pose import orthonormalize, rotation_matrix, rotation_vector
from robobench.hand.tracker import TrackFrame

__all__ = ["RetargetConfig", "RetargetedEpisode", "retarget", "integrate", "CAMERA_TO_ROBOT"]

#: Camera (x right, y down, z into scene) -> robot base (x forward, y left, z up).
#: Applied as v_robot = C @ v_camera. Check it against the three basis vectors: the
#: camera's forward becomes the robot's forward, the camera's right becomes -y
#: (right is negative y when y points left), and the camera's down becomes -z.
CAMERA_TO_ROBOT = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# robosuite's OSC_POSE defaults, which LIBERO does not override. If you change the
# controller config, change these to match or every action is scaled wrongly.
OSC_MAX_POS = 0.05  # metres per control step at |a| = 1
OSC_MAX_ROT = 0.50  # radians per control step at |a| = 1


@dataclass
class RetargetConfig:
    """Everything that turns a hand path into robot actions.

    position_gain   metres of robot motion per metre of hand motion. 1.0 is a
                    literal mapping. Below 1 buys precision at the cost of reach and
                    is the usual choice when the demonstrator's workspace is larger
                    than the robot's.
    control_hz      the simulator's control rate. 20 for LIBERO.
    deadband_m      per-step translation below which the action is zeroed. The
                    default is ~2 mm/step, about the residual jitter of a filtered
                    detector at arm's length, and well under a deliberate motion at
                    20 Hz (a slow 5 cm/s reach still moves 2.5 mm per step).
    gripper_mode    'hysteresis' emits a decisive -1/+1 with a Schmitt trigger.
                    'continuous' passes the aperture through linearly. Hysteresis by
                    default because a parallel-jaw gripper is a binary device and a
                    demo that hovers near zero teaches the policy to hover too —
                    it approaches the object with the fingers half closed and
                    knocks it over.
    """

    position_gain: float = 1.0
    control_hz: float = 20.0
    camera_to_robot: np.ndarray = field(default_factory=lambda: CAMERA_TO_ROBOT.copy())
    max_pos: float = OSC_MAX_POS
    max_rot: float = OSC_MAX_ROT
    deadband_m: float = 0.002
    deadband_rad: float = 0.01
    gripper_mode: str = "hysteresis"
    close_below: float = 0.35
    open_above: float = 0.55
    min_confidence: float = 0.5
    drop_coasted: bool = False

    def __post_init__(self) -> None:
        self.camera_to_robot = np.asarray(self.camera_to_robot, dtype=np.float64).reshape(3, 3)
        if abs(np.linalg.det(self.camera_to_robot) - 1.0) > 1e-6:
            raise ValueError(
                "camera_to_robot must be a rotation (det = +1); got det = "
                f"{np.linalg.det(self.camera_to_robot):.4f}. A determinant of -1 is a "
                "reflection, which mirrors every rotation in the demo while leaving "
                "the positions looking correct."
            )
        if self.gripper_mode not in ("hysteresis", "continuous"):
            raise ValueError(f"gripper_mode must be 'hysteresis' or 'continuous', got {self.gripper_mode!r}")
        if self.close_below >= self.open_above:
            raise ValueError("close_below must be < open_above, or the trigger has no hysteresis band")


@dataclass
class RetargetedEpisode:
    """One demo, on the control clock, in the robot's frame."""

    actions: np.ndarray  # (T, 7) in [-1, 1]
    positions: np.ndarray  # (T, 3) ee path, metres, robot frame
    rotations: np.ndarray  # (T, 3, 3)
    apertures: np.ndarray  # (T,) in [0, 1]
    times: np.ndarray  # (T,) seconds
    source_index: np.ndarray  # (T,) nearest source TrackFrame, for image lookup
    clipped: np.ndarray  # (T,) bool: the step exceeded the controller's per-step limit
    coasted: np.ndarray  # (T,) bool: interpolated through an occlusion
    max_gap_s: float = 0.0  # longest stretch with no usable source frame
    track_id: int = -1
    handedness: str = "right"

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def summary(self) -> dict:
        return {
            "steps": len(self),
            "seconds": float(self.times[-1] - self.times[0]) if len(self) else 0.0,
            "clipped_frac": float(self.clipped.mean()) if len(self) else 0.0,
            "coasted_frac": float(self.coasted.mean()) if len(self) else 0.0,
            "path_length_m": float(np.linalg.norm(np.diff(self.positions, axis=0), axis=-1).sum()),
            "grasps": int(np.sum(np.diff((self.apertures < 0.5).astype(int)) == 1)),
            "max_gap_s": float(self.max_gap_s),
            "handedness": self.handedness,
        }


def retarget(frames: Sequence[TrackFrame], cfg: Optional[RetargetConfig] = None) -> RetargetedEpisode:
    """Track frames (video clock, camera frame) -> actions (control clock, robot frame)."""
    cfg = cfg or RetargetConfig()
    frames = [f for f in frames if f.pose.frame_confidence >= cfg.min_confidence]
    if cfg.drop_coasted:
        frames = [f for f in frames if not f.coasted]
    if len(frames) < 2:
        return _empty(cfg)

    times = np.array([f.t for f in frames], dtype=np.float64)
    keep = np.concatenate([[True], np.diff(times) > 1e-9])  # Slerp rejects duplicates
    frames = [f for f, k in zip(frames, keep) if k]
    times = times[keep]
    if len(frames) < 2:
        return _empty(cfg)

    c = cfg.camera_to_robot
    positions = np.array([c @ f.pose.position for f in frames])
    rotations = np.array([c @ f.pose.rotation @ c.T for f in frames])
    apertures = np.array([f.pose.aperture for f in frames])
    coasted = np.array([f.coasted for f in frames])

    # Confidence gating happens here, after segmentation, so it can open gaps that
    # segmentation never saw — a palm turning edge-on mid-reach drops a stretch of
    # frames, and the resampling below would interpolate straight across it. That
    # interpolation is an invention: it is a straight line through the part of the
    # motion nobody observed. Measure the gap and let the writer decide.
    max_gap_s = float(np.max(np.diff(times))) if len(times) > 1 else 0.0

    grid = np.arange(times[0], times[-1] + 1e-9, 1.0 / cfg.control_hz)
    if len(grid) < 2:
        return _empty(cfg)

    pos = np.stack([np.interp(grid, times, positions[:, k]) for k in range(3)], axis=-1)
    rot = _slerp(times, rotations, grid)
    ap = np.interp(grid, times, apertures)
    src = np.abs(grid[:, None] - times[None, :]).argmin(axis=1)
    coast_g = coasted[src]

    n = len(grid)
    actions = np.zeros((n, 7), dtype=np.float64)
    clipped = np.zeros(n, dtype=bool)

    # Action at step k is the motion from k to k+1; the last step has no successor
    # and holds position rather than inventing one.
    #
    # Both the deadband and the controller's per-step limit are applied through a
    # carry, not by discarding. Discarding is the obvious implementation and it is
    # wrong in both directions:
    #
    #   A deadband that throws sub-threshold motion away deletes slow motion
    #   outright. A careful 3 cm/s approach to an object is 1.5 mm per control step,
    #   under any threshold set high enough to suppress jitter — so the approach
    #   phase of every careful demo becomes a sequence of zeros, and the policy
    #   learns to stop short. Carried instead, jitter still cancels (it is zero-mean,
    #   so the carry never accumulates past the threshold) while genuine slow motion
    #   sums up and is emitted a step or two later. Total displacement is preserved.
    #
    #   Clipping that throws the excess away makes a demo drift. Every step where the
    #   hand outran the controller loses the difference permanently, so the actions
    #   fall further behind the images with each one and the end of a fast demo is
    #   labelled with a robot pose that never gets near the object. Carried, the
    #   robot arrives a step late instead of not at all.
    #
    # The carry is capped, so a demo recorded far too fast degrades to a lag rather
    # than to a robot still travelling after the video has ended. When that happens
    # `clipped` is what tells you: re-record slower, do not train on it.
    max_carry_m = 3.0 * cfg.max_pos
    max_carry_rad = 3.0 * cfg.max_rot
    carry_p = np.zeros(3)
    carry_r = np.eye(3)

    for k in range(n - 1):
        pending = (pos[k + 1] - pos[k]) * cfg.position_gain + carry_p
        step = np.clip(pending, -cfg.max_pos, cfg.max_pos)
        if np.linalg.norm(step) < cfg.deadband_m:
            step = np.zeros(3)
        was_clipped = bool(np.any(np.abs(pending) > cfg.max_pos))
        carry_p = np.clip(pending - step, -max_carry_m, max_carry_m)
        actions[k, 0:3] = step / cfg.max_pos

        pending_r = orthonormalize((rot[k + 1] @ rot[k].T) @ carry_r)
        v = rotation_vector(pending_r)
        norm = float(np.linalg.norm(v))
        if norm > cfg.max_rot:
            was_clipped = True
            v_step = v * (cfg.max_rot / norm)
        elif norm < cfg.deadband_rad:
            v_step = np.zeros(3)
        else:
            v_step = v
        residual = rotation_vector(orthonormalize(pending_r @ rotation_matrix(v_step).T))
        res_norm = float(np.linalg.norm(residual))
        if res_norm > max_carry_rad:
            residual = residual * (max_carry_rad / res_norm)
        carry_r = rotation_matrix(residual)
        actions[k, 3:6] = v_step / cfg.max_rot
        clipped[k] = was_clipped

    actions[:, 6] = _gripper(ap, cfg)

    return RetargetedEpisode(
        actions=actions.astype(np.float32),
        positions=pos,
        rotations=rot,
        apertures=ap,
        times=grid,
        source_index=src,
        clipped=clipped,
        coasted=coast_g,
        max_gap_s=max_gap_s,
        track_id=frames[0].track_id,
        handedness=frames[0].handedness,
    )


def integrate(
    actions: np.ndarray,
    p0: np.ndarray,
    r0: np.ndarray,
    cfg: Optional[RetargetConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay actions forward from an initial pose. The inverse of `retarget`.

    Two uses. It is how the round-trip test checks that retargeting is lossless
    below the clipping limits — a check worth having, because every individual step
    of the transformation looks reasonable in isolation and the composition is where
    a transposed frame or a left-multiplied rotation hides. And it is how you preview
    where a demo would actually send the end-effector before spending simulator time
    on it.
    """
    cfg = cfg or RetargetConfig()
    a = np.asarray(actions, dtype=np.float64)
    pos = np.zeros((len(a) + 1, 3))
    rot = np.zeros((len(a) + 1, 3, 3))
    pos[0], rot[0] = np.asarray(p0, dtype=np.float64), np.asarray(r0, dtype=np.float64)
    for k in range(len(a)):
        pos[k + 1] = pos[k] + a[k, 0:3] * cfg.max_pos / max(cfg.position_gain, 1e-9)
        rot[k + 1] = orthonormalize(rotation_matrix(a[k, 3:6] * cfg.max_rot) @ rot[k])
    return pos, rot


def _gripper(aperture: np.ndarray, cfg: RetargetConfig) -> np.ndarray:
    if cfg.gripper_mode == "continuous":
        return np.clip(1.0 - 2.0 * aperture, -1.0, 1.0)

    # Schmitt trigger: commit to closing below `close_below`, and do not reopen until
    # the aperture clears `open_above`. A single threshold chatters at exactly the
    # moment the fingers touch the object, which is the moment that matters.
    out = np.empty_like(aperture)
    closed = bool(aperture[0] < cfg.close_below)
    for i, a in enumerate(aperture):
        if closed and a > cfg.open_above:
            closed = False
        elif not closed and a < cfg.close_below:
            closed = True
        out[i] = 1.0 if closed else -1.0
    return out


def _slerp(times: np.ndarray, rotations: np.ndarray, grid: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation, Slerp

        rots = Rotation.from_matrix(rotations)
        return Slerp(times, rots)(np.clip(grid, times[0], times[-1])).as_matrix()
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        out = np.zeros((len(grid), 3, 3))
        for i, g in enumerate(np.clip(grid, times[0], times[-1])):
            j = int(np.clip(np.searchsorted(times, g) - 1, 0, len(times) - 2))
            u = (g - times[j]) / max(times[j + 1] - times[j], 1e-12)
            delta = rotations[j + 1] @ rotations[j].T
            out[i] = orthonormalize(rotation_matrix(rotation_vector(delta) * u) @ rotations[j])
        return out


def _empty(cfg: RetargetConfig) -> RetargetedEpisode:
    return RetargetedEpisode(
        actions=np.zeros((0, 7), dtype=np.float32),
        positions=np.zeros((0, 3)),
        rotations=np.zeros((0, 3, 3)),
        apertures=np.zeros(0),
        times=np.zeros(0),
        source_index=np.zeros(0, dtype=int),
        clipped=np.zeros(0, dtype=bool),
        coasted=np.zeros(0, dtype=bool),
    )
