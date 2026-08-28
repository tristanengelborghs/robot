"""Detector backends: one frame of pixels -> zero or more hands.

The seam is `HandDetector`. Everything downstream — filtering, tracking,
retargeting, demo writing — consumes `Detection` objects and never imports a
detector, so swapping MediaPipe for a newer model, a stereo rig, or a glove is a
one-file change that leaves the rest of the pipeline (and its tests) untouched.

Three backends ship:

    MediaPipeDetector   the real one. Optional dependency: requirements-hand.txt.
    SyntheticDetector   a scripted reach-grasp-lift-place with configurable noise
                        and dropouts. No camera, no model download, no network —
                        which is what lets the tests here run in the same
                        "no simulator, no GPU, no downloads" budget as the rest of
                        the suite, and lets you exercise the whole pipeline end to
                        end before you own a webcam.
    ReplayDetector      landmarks previously dumped to .npz. Re-running tracking and
                        retargeting on a recorded session is seconds instead of
                        minutes, and it is reproducible, which matters when you are
                        tuning filter constants against a fixed clip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

import numpy as np

from robobench.hand.landmarks import (
    INDEX_TIP,
    MCP_IDS,
    NUM_LANDMARKS,
    PALM_IDS,
    THUMB_TIP,
    WRIST,
    canonical_palm,
)

__all__ = ["Detection", "HandDetector", "MediaPipeDetector", "SyntheticDetector", "ReplayDetector"]


@dataclass
class Detection:
    """One hand in one frame.

    keypoints    (21, 3) metric, metres, in the camera frame: +x right, +y down,
                 +z away from the camera. Origin at the camera.
    keypoints_px (21, 2) pixel coordinates, for the overlay and the wrist-camera crop.
    handedness   'left' | 'right', from the detector's own classifier.
    score        detector confidence in [0, 1].
    """

    keypoints: np.ndarray
    keypoints_px: np.ndarray
    handedness: str
    score: float = 1.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
        self.keypoints_px = np.asarray(self.keypoints_px, dtype=np.float64).reshape(NUM_LANDMARKS, 2)
        if self.handedness not in ("left", "right"):
            raise ValueError(f"handedness must be 'left' or 'right', got {self.handedness!r}")


class HandDetector(Protocol):
    """Implement this to plug in your own detector."""

    def detect(self, frame: np.ndarray, t: float) -> List[Detection]:
        """frame: (H, W, 3) uint8 RGB. t: seconds. Returns one Detection per hand."""

    def close(self) -> None: ...


#: The bundled hand model, and where `make hand-model` puts it. MediaPipe's Tasks
#: API does not ship weights inside the wheel the way the old solutions API did.
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_HAND_MODEL = "cache/hand_landmarker.task"


def lift_to_camera_frame(
    local_xyz: np.ndarray,
    px: np.ndarray,
    focal_px: float,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Wrist-centred metric landmarks + pixel landmarks -> metric camera coordinates.

    A hand detector gives shape without position: the world landmarks are metres but
    are centred on the wrist, so they describe how the hand is held and say nothing
    about where it is. Position has to come from apparent size, the way any monocular
    system recovers it:

        Z = focal * span_metres / span_pixels

    measuring the same wrist-to-knuckle span in both sets, then X and Y from the
    pinhole model. The span is a palm quantity on purpose. Using a fingertip distance
    would make the hand appear to lunge toward the camera every time it closed,
    because the measured span would shrink for a reason that has nothing to do with
    depth — and the retargeted demo would contain a forward jab at each grasp.

    Returns None when the hand is too small on screen to measure, rather than
    dividing by a near-zero span and emitting a hand several metres away.
    """
    local = np.asarray(local_xyz, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
    px = np.asarray(px, dtype=np.float64).reshape(NUM_LANDMARKS, 2)
    local = local - local[WRIST]

    span_m = float(np.median(np.linalg.norm(local[list(MCP_IDS)], axis=-1)))
    span_px = float(np.median(np.linalg.norm(px[list(MCP_IDS)] - px[WRIST], axis=-1)))
    if span_px < 1.0 or span_m < 1e-4:
        return None

    z = focal_px * span_m / span_px
    cx, cy = width / 2.0, height / 2.0
    wrist_cam = np.array([(px[WRIST, 0] - cx) * z / focal_px, (px[WRIST, 1] - cy) * z / focal_px, z])
    return local + wrist_cam


class MediaPipeDetector:
    """MediaPipe HandLandmarker (Tasks API), lifted to metric camera coordinates.

    Targets mediapipe >= 1.0. The 1.0 release removed `mp.solutions` entirely — the
    legacy API that bundled its own weights — in favour of `mediapipe.tasks`, which
    takes an explicit model file. That is why there is a model path here at all and
    why `make hand-model` exists.

    MediaPipe emits two landmark sets and neither is what retargeting needs alone:
    image landmarks are normalized pixels with a z that is only relative depth in
    x-units, and world landmarks are metric but wrist-centred. `lift_to_camera_frame`
    combines them; see its docstring for how position is recovered.

    `focal_px` is the one number that sets the metric scale of the whole recording.
    The default is derived from an assumed 60-degree horizontal field of view, typical
    for a laptop webcam and wrong for anything wide-angle. It only scales the overall
    trajectory, and `RetargetConfig.position_gain` absorbs exactly that, so a mis-set
    focal length is recoverable — but calibrating the camera once is the honest fix.
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        focal_px: Optional[float] = None,
        hfov_degrees: float = 60.0,
        model_path: Optional[str] = None,
    ):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
        except ImportError as e:  # pragma: no cover - optional dependency
            raise SystemExit(
                "MediaPipe is not installed, or is too old. Tracking real video needs "
                "mediapipe >= 1.0; the synthetic backend and the tests need neither.\n"
                "  pip install -r requirements-hand.txt\n"
                "If you have mediapipe 0.10.x, upgrade: 1.0 removed the `mp.solutions` "
                "API this used to call.\n"
                f"(original error: {e})"
            )

        path = Path(model_path or DEFAULT_HAND_MODEL)
        if not path.is_file():
            raise SystemExit(
                f"hand landmark model not found at {path}.\n"
                "MediaPipe's Tasks API does not bundle weights in the wheel, so the "
                "model is a separate 7 MB download:\n"
                "  make hand-model\n"
                "or point at a copy you already have:\n"
                "  python scripts/track_hands.py ... --hand-model /path/to/hand_landmarker.task"
            )

        self._mp = mp
        self._landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        self.focal_px = focal_px
        self.hfov_degrees = hfov_degrees
        self._last_ms = -1

    def _focal(self, width: int) -> float:
        if self.focal_px is not None:
            return float(self.focal_px)
        return 0.5 * width / np.tan(np.deg2rad(self.hfov_degrees) / 2.0)

    def detect(self, frame: np.ndarray, t: float) -> List[Detection]:
        h, w = frame.shape[:2]
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)
        )
        # VIDEO mode requires strictly increasing millisecond timestamps and raises
        # otherwise. A 60 fps capture rounds to 16 ms steps so this never bites in
        # practice, but a duplicated frame time would abort a whole recording.
        ms = max(int(round(t * 1000.0)), self._last_ms + 1)
        self._last_ms = ms

        result = self._landmarker.detect_for_video(image, ms)
        if not result.hand_landmarks:
            return []

        f = self._focal(w)
        out: List[Detection] = []
        for lm, world, handed in zip(
            result.hand_landmarks, result.hand_world_landmarks, result.handedness
        ):
            px = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
            local = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float64)
            keypoints = lift_to_camera_frame(local, px, f, w, h)
            if keypoints is None:
                continue
            top = handed[0]
            out.append(
                Detection(
                    keypoints=keypoints,
                    keypoints_px=px,
                    # The label describes the hand as it appears in the image. A webcam
                    # feed is mirrored, so it is flipped relative to the person unless
                    # the caller un-mirrors the frame first (scripts/track_hands.py
                    # does). Recorded third-person video needs no correction.
                    handedness=str(top.category_name).lower(),
                    score=float(top.score),
                    meta={"depth_m": float(keypoints[WRIST, 2])},
                )
            )
        return out

    def close(self) -> None:
        self._landmarker.close()


class SyntheticDetector:
    """A scripted pick-and-place, with the failure modes of a real detector.

    The trajectory is deterministic given the seed, so a filter change shows up as a
    number rather than an impression. `noise_m` is per-landmark Gaussian jitter
    (2-4 mm is realistic for MediaPipe world landmarks at arm's length), `dropout`
    is the per-frame probability of losing the hand entirely, and `occlusion` inserts
    contiguous blackouts — the case that matters, because independent per-frame
    dropouts are easy and a two-thirds-of-a-second occlusion is not.
    """

    def __init__(
        self,
        duration: float = 6.0,
        fps: float = 30.0,
        noise_m: float = 0.003,
        dropout: float = 0.0,
        occlusions: Sequence[tuple] = (),
        handedness: str = "right",
        seed: int = 0,
        image_size: tuple = (480, 640),
        focal_px: float = 554.0,
        num_hands: int = 1,
        period: Optional[float] = None,
    ):
        self.duration, self.fps = duration, fps
        # Without a period the whole recording is one attempt. With one, the script
        # repeats every `period` seconds, so a long recording split on gaps yields
        # several *complete* pick-and-places rather than several slices of one — the
        # difference between smoke data with a learnable signal in it and smoke data
        # where three demos out of four never grasp anything.
        self.period = period
        self.noise_m, self.dropout = noise_m, dropout
        self.occlusions = tuple(occlusions)
        self.handedness = handedness
        self.rng = np.random.default_rng(seed)
        self.h, self.w = image_size
        self.focal_px = focal_px
        self.num_hands = num_hands

    # -- the scripted motion ------------------------------------------------

    def phase(self, t: float) -> float:
        span = self.period if self.period else self.duration
        u = (t % span) if self.period else t
        return float(np.clip(u / max(span, 1e-6), 0.0, 1.0))

    def wrist_path(self, t: float, hand_index: int = 0) -> np.ndarray:
        """Reach out and down, grasp, lift, traverse, place. Metres, camera frame."""
        p = self.phase(t)
        start = np.array([-0.15 + 0.30 * hand_index, -0.05, 0.55])
        grasp = np.array([0.00 + 0.30 * hand_index, 0.10, 0.62])
        lift = np.array([0.00 + 0.30 * hand_index, -0.05, 0.60])
        place = np.array([0.18 + 0.30 * hand_index, 0.06, 0.58])
        if p < 0.35:
            return _ease(start, grasp, p / 0.35)
        if p < 0.45:
            return grasp
        if p < 0.60:
            return _ease(grasp, lift, (p - 0.45) / 0.15)
        if p < 0.85:
            return _ease(lift, place, (p - 0.60) / 0.25)
        return place

    def aperture(self, t: float) -> float:
        """Open, close on the object at 40% through, hold, release at 88%."""
        p = self.phase(t)
        if p < 0.35:
            return 1.0
        if p < 0.45:
            return float(np.clip(1.0 - (p - 0.35) / 0.08, 0.0, 1.0))
        if p < 0.88:
            return 0.0
        return float(np.clip((p - 0.88) / 0.06, 0.0, 1.0))

    def wrist_rotation(self, t: float) -> np.ndarray:
        from robobench.hand.pose import rotation_matrix

        p = self.phase(t)
        return rotation_matrix(np.array([0.0, -0.9 * p, 0.35 * np.sin(2 * np.pi * p)]))

    # -- assembly -----------------------------------------------------------

    def _hand(self, t: float, hand_index: int) -> np.ndarray:
        kp = np.zeros((NUM_LANDMARKS, 3))
        handed = self.handedness if hand_index == 0 else _other(self.handedness)
        palm = canonical_palm(handed)
        for i, pid in enumerate(PALM_IDS):
            kp[pid] = palm[i]

        # Fingers: straight along +x from each knuckle, curling as the grip closes.
        from robobench.hand.landmarks import FINGERS

        a = self.aperture(t)
        for name, chain in FINGERS.items():
            if name == "thumb":
                continue
            base = kp[chain[0]].copy()
            direction = np.array([1.0, 0.0, 0.0])
            for j in range(1, len(chain)):
                bend = (1.0 - a) * 0.55 * j
                kp[chain[j]] = base + 0.025 * j * (
                    np.cos(bend) * direction + np.sin(bend) * np.array([0.0, 0.0, -1.0])
                )
        # Thumb closes onto the index tip, which is what makes aperture meaningful.
        far = np.array([0.115, 0.115 if handed == "right" else -0.115, 0.0])
        kp[THUMB_TIP] = kp[INDEX_TIP] + a * (far - kp[INDEX_TIP])
        for k, pid in enumerate((1, 2, 3)):
            kp[pid] = kp[WRIST] + (k + 1) / 4.0 * (kp[THUMB_TIP] - kp[WRIST])

        return kp @ self.wrist_rotation(t).T + self.wrist_path(t, hand_index)

    def _occluded(self, t: float) -> bool:
        return any(lo <= t < hi for lo, hi in self.occlusions)

    def detect(self, frame: np.ndarray, t: float) -> List[Detection]:
        if self._occluded(t) or self.rng.random() < self.dropout:
            return []
        out = []
        for i in range(self.num_hands):
            kp = self._hand(t, i) + self.rng.normal(0.0, self.noise_m, (NUM_LANDMARKS, 3))
            z = np.maximum(kp[:, 2], 1e-3)
            px = np.stack(
                [kp[:, 0] * self.focal_px / z + self.w / 2.0, kp[:, 1] * self.focal_px / z + self.h / 2.0],
                axis=-1,
            )
            handed = self.handedness if i == 0 else _other(self.handedness)
            out.append(Detection(keypoints=kp, keypoints_px=px, handedness=handed, score=0.95))
        return out

    def frames(self):
        """(frame, t) pairs. The image is a low-contrast render of the landmarks —
        enough for the wrist-camera crop and the overlay to have something to bite
        on, not a substitute for real video."""
        n = int(round(self.duration * self.fps))
        for i in range(n):
            t = i / self.fps
            img = np.full((self.h, self.w, 3), 24, dtype=np.uint8)
            img[:, :, 2] = 40
            for det in self.detect(img, t):
                for (x, y) in det.keypoints_px:
                    xi, yi = int(round(x)), int(round(y))
                    if 2 <= xi < self.w - 2 and 2 <= yi < self.h - 2:
                        img[yi - 2 : yi + 3, xi - 2 : xi + 3] = 220
            yield img, t

    def close(self) -> None:
        pass


class ReplayDetector:
    """Replays landmarks written by `scripts/track_hands.py --dump-landmarks`."""

    def __init__(self, path: str | Path):
        blob = np.load(Path(path), allow_pickle=True)
        self.times = blob["times"].astype(np.float64)
        self.keypoints = blob["keypoints"].astype(np.float64)  # (N, hands, 21, 3)
        self.keypoints_px = blob["keypoints_px"].astype(np.float64)
        self.valid = blob["valid"].astype(bool)  # (N, hands)
        self.handedness = [str(h) for h in blob["handedness"]]
        self._i = 0

    def detect(self, frame: np.ndarray, t: float) -> List[Detection]:
        i = int(np.argmin(np.abs(self.times - t)))
        return [
            Detection(
                keypoints=self.keypoints[i, j],
                keypoints_px=self.keypoints_px[i, j],
                handedness=self.handedness[j],
            )
            for j in range(self.keypoints.shape[1])
            if self.valid[i, j]
        ]

    def close(self) -> None:
        pass


def _ease(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """Smoothstep between two points — C1 continuous, so no velocity discontinuity
    at the waypoints for the filter to ring on."""
    u = float(np.clip(u, 0.0, 1.0))
    return a + (b - a) * (u * u * (3.0 - 2.0 * u))


def _other(handedness: str) -> str:
    return "left" if handedness == "right" else "right"
