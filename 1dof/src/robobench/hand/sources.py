"""Frame sources: video files and cameras, as (frame, timestamp) streams.

In the package rather than in the CLI because timestamps are not glue. Everything
downstream — the filter's adaptive cutoff, the tracker's association gate, the
resampling onto the control clock — is a function of *seconds*, so a source that
reports the wrong time reports the wrong velocity, and the demo trains at the wrong
speed with nothing anywhere to say so. That deserves tests, and tests need an import.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

__all__ = ["probe_fps", "video_frames", "camera_frames", "camera_error"]


def probe_fps(path: str | Path) -> float:
    """Frame rate of a video file. Raises rather than guessing.

    This function exists because guessing is worse than failing here. An earlier
    version asked imageio for the `pyav` plugin specifically, fell back to 30.0 when
    that plugin was absent — which it is by default, since the packaged decoder is
    imageio-ffmpeg — and so read *every* video as 30 fps. A 60 fps phone clip then
    produced timestamps twice as long as the real ones, halving every velocity in the
    demo. Nothing downstream can detect that: the trajectory is smooth, the actions
    are in range, the loss falls, and the policy has learned to move at half speed.

    So: let imageio choose its own plugin, and if the rate still cannot be read, say
    so and ask for `--fps` instead of substituting a plausible number.
    """
    path = Path(path)
    try:
        import imageio.v3 as iio
    except ImportError as e:  # pragma: no cover - imageio is a core dependency
        raise SystemExit(f"imageio is required to read video files ({e})")

    fps: Optional[float] = None
    try:
        meta = iio.immeta(path)
        value = meta.get("fps") or meta.get("frame_rate")
        if value:
            fps = float(value)
    except Exception:
        fps = None

    if not fps or not np.isfinite(fps) or fps <= 0:
        raise SystemExit(
            f"could not read the frame rate of {path}.\n"
            "Timestamps drive every velocity in the demo, so guessing a rate here would\n"
            "silently rescale the whole recording rather than fail. Pass the real one:\n"
            "  python scripts/track_hands.py --video <file> --fps 30 ...\n"
            "  ffprobe -v0 -select_streams v:0 -show_entries stream=r_frame_rate <file>\n"
            "Installing a decoder may also fix it:  pip install imageio-ffmpeg"
        )
    return fps


def video_frames(
    path: str | Path,
    max_seconds: Optional[float] = None,
    fps: Optional[float] = None,
) -> Iterator[Tuple[np.ndarray, float]]:
    """Decode a video file into (RGB frame, seconds) pairs."""
    import imageio.v3 as iio

    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"no such video file: {path}\n"
            "Record one first — QuickTime (File > New Movie Recording) or any phone will do —\n"
            "or try the whole pipeline with no video at all:\n"
            '  python scripts/track_hands.py --synthetic --instruction "pick up the block"'
        )

    rate = float(fps) if fps else probe_fps(path)
    for i, frame in enumerate(iio.imiter(path)):
        t = i / rate
        if max_seconds is not None and t > max_seconds:
            return
        frame = np.asarray(frame)
        if frame.ndim == 2:  # greyscale source
            frame = np.stack([frame] * 3, axis=-1)
        yield np.ascontiguousarray(frame[..., :3]), t


def camera_frames(
    index: int,
    max_seconds: float,
    mirror: bool = True,
) -> Iterator[Tuple[np.ndarray, float]]:
    """Webcam capture as (RGB frame, seconds) pairs.

    Timestamps come from the clock, not from a frame counter. A webcam drops frames
    under load, and a counter-derived timestamp turns a dropped frame into a
    permanent rescaling of every velocity that follows it.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - optional dependency
        raise SystemExit(
            f"opencv is required for webcam capture ({e}).\n  pip install -r requirements-hand.txt"
        )

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(camera_error(index))

    start = time.time()
    try:
        while True:
            ok, bgr = cap.read()
            t = time.time() - start
            if not ok or t > max_seconds:
                return
            rgb = bgr[..., ::-1]
            # A webcam feed is mirrored, so the detector's handedness label is flipped
            # relative to the person. Un-mirroring the frame rather than correcting the
            # label keeps the image and the label consistent, including in the overlay.
            yield np.ascontiguousarray(rgb[:, ::-1] if mirror else rgb), t
    finally:
        cap.release()


def camera_error(index: int) -> str:
    """Why the camera would not open, with the fix rather than the symptom.

    On macOS this is almost always TCC, not the camera: the process inherits camera
    permission from the terminal application it was launched from, and a plain CLI
    python has no bundle identity to raise a permission dialog with. OpenCV prints
    "not authorized to capture video (status 0), requesting..." and then fails,
    which reads like broken hardware and is a checkbox.
    """
    lines = [f"could not open camera {index}."]
    if sys.platform == "darwin":
        lines += [
            "",
            "On macOS this is usually camera permission, not the camera. The permission",
            "belongs to the terminal application you launched this from, not to python:",
            "",
            "  1. System Settings -> Privacy & Security -> Camera",
            "  2. Enable the terminal app you are running in (Terminal, iTerm, VS Code, ...)",
            "  3. Quit that app completely and reopen it — the permission is read at launch,",
            "     so a running terminal keeps the old answer until it restarts",
            "",
            "If it is already enabled, another application may hold the camera; close",
            "Zoom, Photo Booth, FaceTime and browser tabs, then retry.",
        ]
    else:
        lines += [
            "",
            "Check that the device exists and nothing else holds it:",
            "  ls /dev/video*        # a different index may be the right one",
            "  fuser -v /dev/video0  # what is using it",
        ]
    lines += [
        "",
        "No camera needed to record a demo — capture a clip any way you like and use:",
        '  python scripts/track_hands.py --video clip.mp4 --instruction "..."',
        "",
        "Or exercise the whole pipeline with no camera and no model at all:",
        '  python scripts/track_hands.py --synthetic --instruction "pick up the block"',
    ]
    return "\n".join(lines)
