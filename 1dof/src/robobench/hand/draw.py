"""Overlay rendering: what the tracker actually sees, drawn on the frame.

Kept separate from both scripts because the live preview and the recorder need the
same picture. Uses OpenCV when it is available (anti-aliased lines and text, which
matters at 30 fps) and falls back to plain numpy, so the drawing works in the same
no-dependency environment as the rest of the package.

Colour carries the confidence gate rather than decorating it: green means the frame
would be used, amber means the palm pose is too foreshortened to trust, grey means
the position is extrapolated through an occlusion and is not an observation.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from robobench.hand.landmarks import BONES, MCP_IDS, TIP_IDS, WRIST

__all__ = ["draw_landmarks", "draw_hud", "GOOD", "LOW_CONFIDENCE", "COASTED"]

GOOD: Tuple[int, int, int] = (70, 220, 120)
LOW_CONFIDENCE: Tuple[int, int, int] = (250, 180, 60)
COASTED: Tuple[int, int, int] = (150, 150, 160)


def _cv2():
    try:
        import cv2

        return cv2
    except ImportError:  # pragma: no cover - optional
        return None


def track_colour(tf, min_confidence: float) -> Tuple[int, int, int]:
    if tf.coasted:
        return COASTED
    if tf.pose.frame_confidence < min_confidence:
        return LOW_CONFIDENCE
    return GOOD


def draw_landmarks(
    frame: np.ndarray,
    track_frames: Iterable,
    min_confidence: float = 0.5,
    thickness: int = 2,
) -> np.ndarray:
    """Skeleton, knuckles and fingertips for every tracked hand. Returns a new RGB frame."""
    img = np.ascontiguousarray(frame.copy())
    cv2 = _cv2()
    h, w = img.shape[:2]

    for tf in track_frames:
        colour = track_colour(tf, min_confidence)
        px = np.asarray(tf.keypoints_px, dtype=float)

        for a, b in BONES:
            _line(img, px[a], px[b], colour, thickness, cv2)
        for i in range(px.shape[0]):
            r = 4 if i in TIP_IDS else (4 if i in MCP_IDS else 3)
            _dot(img, px[i], colour, r, cv2)
        # The wrist is the origin of the retargeted trajectory — worth seeing plainly.
        _dot(img, px[WRIST], colour, 7, cv2, filled=False)

        if cv2 is not None and 0 <= px[WRIST, 0] < w and 0 <= px[WRIST, 1] < h:
            label = f"#{tf.track_id} {tf.handedness}"
            cv2.putText(img, label, (int(px[WRIST, 0]) + 10, int(px[WRIST, 1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return img


def draw_hud(
    frame: np.ndarray,
    track_frames: Sequence,
    fps: Optional[float] = None,
    min_confidence: float = 0.5,
) -> np.ndarray:
    """Live readouts: rate, and per-hand grip aperture, pose confidence and depth.

    The aperture bar is the one to watch. It is what becomes the gripper command, so
    if it does not swing decisively as you open and close your hand, nothing further
    down the pipeline can recover the grasp.
    """
    cv2 = _cv2()
    if cv2 is None:  # pragma: no cover - text needs opencv
        return frame

    img = np.ascontiguousarray(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 24

    header = f"{fps:4.1f} fps" if fps else ""
    header += f"   {len(track_frames)} hand(s)" if track_frames else "   no hands"
    cv2.putText(img, header, (12, y), font, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    y += 26

    for tf in track_frames:
        colour = track_colour(tf, min_confidence)
        pose = tf.pose
        state = "coasted" if tf.coasted else ("low-conf" if pose.frame_confidence < min_confidence else "ok")
        cv2.putText(
            img,
            f"#{tf.track_id} {tf.handedness:5s} conf {pose.frame_confidence:.2f}  "
            f"z {pose.position[2]:.2f}m  {state}",
            (12, y), font, 0.5, colour, 1, cv2.LINE_AA,
        )
        y += 20

        x0, bar_w, bar_h = 12, 160, 10
        cv2.rectangle(img, (x0, y - 8), (x0 + bar_w, y - 8 + bar_h), (90, 90, 90), 1)
        fill = int(bar_w * float(np.clip(pose.aperture, 0.0, 1.0)))
        if fill > 0:
            cv2.rectangle(img, (x0, y - 8), (x0 + fill, y - 8 + bar_h), colour, -1)
        cv2.putText(img, f"grip {pose.aperture:.2f}", (x0 + bar_w + 10, y + 1),
                    font, 0.45, colour, 1, cv2.LINE_AA)
        y += 26

    return img


# -- primitives -------------------------------------------------------------


def _line(img, p0, p1, colour, thickness, cv2) -> None:
    h, w = img.shape[:2]
    if not (_on(p0, w, h) or _on(p1, w, h)):
        return
    if cv2 is not None:
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), colour, thickness, cv2.LINE_AA)
        return
    n = max(int(np.hypot(p1[0] - p0[0], p1[1] - p0[1])), 1)
    for u in np.linspace(0.0, 1.0, n):
        x, y = int(round(p0[0] + u * (p1[0] - p0[0]))), int(round(p0[1] + u * (p1[1] - p0[1])))
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = colour


def _dot(img, p, colour, radius, cv2, filled: bool = True) -> None:
    h, w = img.shape[:2]
    if not _on(p, w, h):
        return
    if cv2 is not None:
        cv2.circle(img, (int(p[0]), int(p[1])), radius, colour, -1 if filled else 2, cv2.LINE_AA)
        return
    x, y = int(round(p[0])), int(round(p[1]))
    lo, hi = max(y - radius, 0), min(y + radius + 1, h)
    lo_x, hi_x = max(x - radius, 0), min(x + radius + 1, w)
    img[lo:hi, lo_x:hi_x] = colour


def _on(p, w: int, h: int) -> bool:
    return bool(np.isfinite(p).all() and -w < p[0] < 2 * w and -h < p[1] < 2 * h)
