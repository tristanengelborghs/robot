#!/usr/bin/env python3
"""Live hand tracking preview. Nothing is recorded, nothing is written.

    python scripts/watch_hands.py                    # webcam 0
    python scripts/watch_hands.py --camera 1
    python scripts/watch_hands.py --synthetic        # no camera, scripted motion

A window opens showing your hand with the tracked skeleton drawn over it. Press q or
Esc to quit.

What the colours mean — they are the confidence gate, not decoration:

    green   the pose is good; a frame like this is usable
    amber   the palm is too foreshortened to trust the orientation. Turn your palm
            more toward the camera.
    grey    the hand is not currently visible and the position shown is extrapolated
            through the gap, not observed.

The grip bar is worth watching more than the skeleton. It is thumb-to-index distance
normalized by your hand size, and it is what becomes the gripper command — if it does
not swing decisively between open and pinched, no amount of downstream tuning will
recover the grasp.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robobench.hand.detect import MediaPipeDetector, SyntheticDetector  # noqa: E402
from robobench.hand.draw import draw_hud, draw_landmarks  # noqa: E402
from robobench.hand.sources import camera_frames  # noqa: E402
from robobench.hand.tracker import HandTracker  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--synthetic", action="store_true", help="scripted motion; no camera needed")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until q)")
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--min-confidence", type=float, default=0.5, help="amber below this")
    ap.add_argument("--no-mirror", action="store_true", help="feed is not mirrored")
    ap.add_argument("--hand-model", default=None, help="path to hand_landmarker.task")
    ap.add_argument("--focal-px", type=float, default=None, help="camera focal length, if calibrated")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale factor")
    ap.add_argument("--save-frame", default=None, help="write the first tracked frame here and exit")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError as e:
        raise SystemExit(
            f"opencv is required to show a preview window ({e}).\n"
            "  pip install -r requirements-hand.txt"
        )

    if args.synthetic:
        detector = SyntheticDetector(duration=1e6, period=6.0, noise_m=0.003)
        source = detector.frames()
    else:
        detector = MediaPipeDetector(
            max_hands=args.max_hands, focal_px=args.focal_px, model_path=args.hand_model
        )
        source = camera_frames(args.camera, args.seconds or float("inf"), mirror=not args.no_mirror)

    tracker = HandTracker(max_hands=args.max_hands)
    window = "robobench - hand tracking (q to quit)"
    smoothed_fps, last = None, time.time()
    saved = False

    print("tracking. press q or Esc in the window to quit.")
    try:
        for frame, t in source:
            track_frames = tracker.update(detector.detect(frame, t), t)

            now = time.time()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            smoothed_fps = inst if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * inst

            vis = draw_landmarks(frame, track_frames, args.min_confidence)
            vis = draw_hud(vis, track_frames, smoothed_fps, args.min_confidence)

            if args.save_frame and track_frames and not saved:
                import imageio.v3 as iio

                Path(args.save_frame).parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(args.save_frame, vis)
                print(f"wrote {args.save_frame}")
                saved = True
                if args.synthetic:
                    break

            if args.scale != 1.0:
                vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window, vis[..., ::-1])  # the window wants BGR

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            if args.seconds is not None and t > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
