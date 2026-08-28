#!/usr/bin/env python3
"""Video (or webcam, or nothing at all) -> LIBERO-shaped demo files.

    # a recorded clip
    python scripts/track_hands.py --video demo.mp4 --instruction "pick up the mug"

    # live, one episode per press of the spacebar equivalent: it splits on gaps
    python scripts/track_hands.py --camera 0 --instruction "pick up the mug" --seconds 60

    # no camera, no mediapipe, no downloads — exercises the whole path
    python scripts/track_hands.py --synthetic --instruction "pick up the block" --episodes 4

Then the same three commands as any other dataset:

    python scripts/cache_language.py --data-root data/hand --suites hand_demos
    python -m robobench.train --config configs/hand_bc.yaml
    python -m robobench.eval --ckpt runs/<run>/ckpt_last.pt

One long recording becomes several demos: the tracker splits wherever the hand
leaves frame for longer than --max-gap, which is what a demonstrator does between
attempts anyway. Check the printed per-episode summary before training on it — a
high clipped fraction means the demonstration outran the controller and the actions
no longer match the images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robobench.hand.detect import MediaPipeDetector, SyntheticDetector  # noqa: E402
from robobench.hand.draw import draw_landmarks  # noqa: E402
from robobench.hand.record import segment_episodes, write_demos  # noqa: E402
from robobench.hand.retarget import CAMERA_TO_ROBOT, RetargetConfig  # noqa: E402
from robobench.hand.sources import camera_frames, video_frames  # noqa: E402
from robobench.hand.tracker import HandTracker  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to a recorded clip")
    src.add_argument("--camera", type=int, help="webcam index")
    src.add_argument("--synthetic", action="store_true", help="scripted motion; no camera or model needed")

    ap.add_argument("--instruction", required=True, help="language instruction for this task")
    ap.add_argument("--out", default="data/hand/hand_demos", help="suite directory to write into")
    ap.add_argument("--seconds", type=float, default=60.0, help="capture/read limit")
    ap.add_argument("--fps", type=float, default=None,
                    help="override the video frame rate; only needed if it cannot be read")
    ap.add_argument("--episodes", type=int, default=4, help="synthetic only: how many to generate")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--no-mirror", action="store_true", help="webcam only: feed is not mirrored")

    ap.add_argument("--max-gap", type=float, default=0.4, help="seconds without a hand that ends an episode")
    ap.add_argument("--min-duration", type=float, default=1.0, help="drop episodes shorter than this")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="palm-frame confidence below which a frame is unusable for retargeting")
    ap.add_argument("--drop-coasted", action="store_true", help="exclude frames extrapolated through occlusion")

    ap.add_argument("--position-gain", type=float, default=1.0, help="robot metres per hand metre")
    ap.add_argument("--control-hz", type=float, default=20.0, help="simulator control rate (LIBERO: 20)")
    ap.add_argument("--gripper-mode", default="hysteresis", choices=("hysteresis", "continuous"))
    ap.add_argument("--focal-px", type=float, default=None, help="camera focal length; sets the metric scale")
    ap.add_argument("--hand-model", default=None,
                    help="path to hand_landmarker.task (default: cache/, via `make hand-model`)")
    ap.add_argument("--hand", default=None, choices=("left", "right"), help="keep only this hand")

    ap.add_argument("--overlay", default=None, help="write an annotated mp4 for eyeballing the tracking")
    ap.add_argument("--dump-landmarks", default=None, help="write raw landmarks .npz for offline re-runs")
    ap.add_argument("--manifest", default=None, help="write the per-episode manifest as json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = RetargetConfig(
        position_gain=args.position_gain,
        control_hz=args.control_hz,
        camera_to_robot=CAMERA_TO_ROBOT,
        gripper_mode=args.gripper_mode,
        min_confidence=args.min_confidence,
        drop_coasted=args.drop_coasted,
    )

    if args.synthetic:
        period = max(args.seconds / max(args.episodes, 1), 2.0)
        total = period * args.episodes
        detector = SyntheticDetector(
            duration=total,
            period=period,
            noise_m=0.003,
            occlusions=_spread_occlusions(args.episodes, total),
            seed=args.seed,
        )
        source = detector.frames()
        label = "synthetic"
    elif args.video:
        detector = MediaPipeDetector(focal_px=args.focal_px, model_path=args.hand_model)
        source = video_frames(args.video, args.seconds, fps=args.fps)
        label = args.video
    else:
        detector = MediaPipeDetector(focal_px=args.focal_px, model_path=args.hand_model)
        source = camera_frames(args.camera, args.seconds, mirror=not args.no_mirror)
        label = f"camera:{args.camera}"

    # The confidence gate is a retargeting concern, not a tracking one: a hand whose
    # palm has turned edge-on is still the same hand, and dropping it in the tracker
    # would break the track and split the demo in two.
    tracker = HandTracker()
    frames, images, overlay_frames = [], [], []
    raw = {"times": [], "keypoints": [], "keypoints_px": [], "valid": [], "handedness": []}

    n_read = 0
    for image, t in source:
        n_read += 1
        detections = detector.detect(image, t)
        if args.hand:
            detections = [d for d in detections if d.handedness == args.hand]
        tfs = tracker.update(detections, t)
        for tf in tfs:
            frames.append(tf)
            images.append(image)
        if args.overlay is not None:
            overlay_frames.append(draw_landmarks(image, tfs, args.min_confidence))
        if args.dump_landmarks is not None:
            _accumulate_raw(raw, detections, t)
    detector.close()

    print(f"read {n_read} frames from {label}; {len(frames)} tracked hand-frames")
    if not frames:
        raise SystemExit(
            "no hands were tracked.\n"
            "  - is the hand fully in frame, and lit well enough to see the knuckles?\n"
            "  - a track needs 3 consecutive detections before it is reported\n"
            "  - --overlay out.mp4 renders what the detector actually saw"
        )

    episodes = segment_episodes(frames, images, max_gap=args.max_gap, min_duration=args.min_duration)
    print(f"segmented into {len(episodes)} episode(s): " + ", ".join(f"{e.duration:.1f}s" for e in episodes))

    out_dir = Path(args.out)
    path = out_dir / (args.instruction.strip().replace(" ", "_").replace("/", "_") + "_demo.hdf5")
    manifest = write_demos(
        path, episodes, args.instruction, cfg=cfg, image_size=args.image_size, source=label
    )

    for i, s in enumerate(manifest["summaries"]):
        print(
            f"  demo_{i}: {s['steps']:4d} steps  {s['seconds']:5.1f}s  path {s['path_length_m']:.2f}m  "
            f"grasps {s['grasps']}  clipped {s['clipped_frac']:.0%}  coasted {s['coasted_frac']:.0%}"
        )

    if args.synthetic:
        # Same bargain make_synthetic_data.py strikes: a synthetic run must not need
        # a CLIP download to reach the training loop. Real instructions get real
        # embeddings from scripts/cache_language.py; a scripted one gets a fixed
        # random vector, which is enough to prove the plumbing and nothing more.
        _write_synthetic_language_cache(args.instruction, args.seed)

    if args.overlay:
        _write_video(args.overlay, overlay_frames)
    if args.dump_landmarks:
        _write_landmarks(args.dump_landmarks, raw)
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))

    if args.synthetic:
        print(
            f"\nnext:\n  python -m robobench.train --config configs/hand_bc.yaml "
            f"data.root={out_dir.parent} data.lang_cache={SYNTHETIC_LANG_CACHE}"
        )
    else:
        print(
            "\nnext:\n"
            f"  python scripts/cache_language.py --data-root {out_dir.parent} --suites {out_dir.name}"
            " --out cache/lang_hand.npz\n"
            f"  python -m robobench.train --config configs/hand_bc.yaml data.root={out_dir.parent}"
        )


SYNTHETIC_LANG_CACHE = "cache/lang_hand_synthetic.npz"


def _write_synthetic_language_cache(instruction: str, seed: int) -> None:
    path = Path(SYNTHETIC_LANG_CACHE)
    path.parent.mkdir(parents=True, exist_ok=True)
    vec = np.random.default_rng(seed).normal(0.0, 1.0, (1, 512)).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    np.savez(path, keys=np.array([instruction], dtype=object), vectors=vec)
    print(f"wrote {path}")


def _spread_occlusions(episodes: int, seconds: float) -> List[Tuple[float, float]]:
    """Synthetic only: a gap between each episode, so segmentation has something to cut on."""
    if episodes <= 1:
        return []
    span = seconds / episodes
    return [((i + 1) * span - 0.6, (i + 1) * span) for i in range(episodes - 1)]


def _accumulate_raw(raw: dict, detections, t: float) -> None:
    from robobench.hand.landmarks import NUM_LANDMARKS

    slots = 2
    kp = np.zeros((slots, NUM_LANDMARKS, 3))
    px = np.zeros((slots, NUM_LANDMARKS, 2))
    valid = np.zeros(slots, dtype=bool)
    handed = ["right", "left"]
    for j, det in enumerate(detections[:slots]):
        kp[j], px[j], valid[j], handed[j] = det.keypoints, det.keypoints_px, True, det.handedness
    raw["times"].append(t)
    raw["keypoints"].append(kp)
    raw["keypoints_px"].append(px)
    raw["valid"].append(valid)
    raw["handedness"] = handed


def _write_landmarks(path: str, raw: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=np.array(raw["times"]),
        keypoints=np.array(raw["keypoints"]),
        keypoints_px=np.array(raw["keypoints_px"]),
        valid=np.array(raw["valid"]),
        handedness=np.array(raw["handedness"], dtype=object),
    )
    print(f"wrote {path}")


def _write_video(path: str, frames: List[np.ndarray]) -> None:
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, np.stack(frames), fps=30)
        print(f"wrote {path}")
    except Exception as e:  # pragma: no cover - depends on ffmpeg being present
        print(f"could not write overlay video ({e}); install imageio-ffmpeg")


if __name__ == "__main__":
    main()
