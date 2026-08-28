"""Tracked hands -> LIBERO-shaped HDF5 that `LiberoChunkDataset` reads unchanged.

The point of writing this exact layout is that nothing downstream has to know where
a demo came from. Training, ablations, normalization and the statistics all work on
hand demos the day they are written, and a hand-demo run is comparable to a
simulator-demo run because it goes through identical code.

Two conventions are load-bearing, and both are the silent kind:

Images are stored upside down. LIBERO's images come out of MuJoCo's offscreen
buffer bottom-up, and `LiberoChunkDataset` flips them back on the way in. A demo
written right-way-up therefore trains upside down relative to every other demo in
the dataset. Storing them flipped is not a bug being preserved for its own sake; it
is what makes one dataset out of two sources.

Proprioception is not joint angles. A human hand has no robot arm behind it, so the
proprio slot carries wrist pose and hand scale instead. It occupies the same nine
dimensions and trains without complaint, which is precisely the danger — so every
file written here is stamped `proprio_source = "hand_pose"`, and the dataset refuses
to build a mixed dataset out of files that disagree about what those nine numbers
mean. Mixing them is not a warning-level event: it is a policy conditioned on a
vector that means two different things depending on which file the sample came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np

from robobench.hand.landmarks import MCP_IDS, MIDDLE_MCP, WRIST
from robobench.hand.retarget import RetargetConfig, RetargetedEpisode, retarget
from robobench.hand.tracker import TrackFrame

__all__ = ["Episode", "segment_episodes", "wrist_view", "write_demos", "FORMAT_VERSION"]

FORMAT_VERSION = 1
PROPRIO_SOURCE = "hand_pose"

#: A demo with more clipped steps than this outran the controller badly enough that
#: the actions no longer describe the motion in the images. Re-record it slower.
CLIPPED_FRACTION_LIMIT = 0.15

#: The longest stretch, in seconds, that retargeting may interpolate across. Frames
#: gated out for low pose confidence leave holes that segmentation never saw, and a
#: hole wider than this is filled with a straight line through motion nobody
#: observed. Five control steps.
INTERPOLATION_GAP_LIMIT = 0.25


@dataclass
class Episode:
    """A contiguous stretch of one tracked hand, with the frames it came from."""

    frames: List[TrackFrame]
    images: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        return self.frames[-1].t - self.frames[0].t if self.frames else 0.0


def segment_episodes(
    frames: Sequence[TrackFrame],
    images: Sequence[np.ndarray],
    max_gap: float = 0.4,
    min_duration: float = 1.0,
    track_id: Optional[int] = None,
) -> List[Episode]:
    """Cut a continuous recording into demos wherever the hand goes away.

    A gap is where the demonstrator reset the scene, took their hand out to move the
    object back, or simply lost tracking. Retargeting straight across one produces a
    single enormous action at the seam, and since the seam is exactly where the scene
    changed, it is the one frame where the policy would most confidently learn the
    wrong thing.

    A track ending and another beginning is also a cut, even with no time gap: two
    ids mean the tracker was not able to establish that it was the same hand, and
    stitching them together would override that judgement with an assumption.
    """
    if not frames:
        return []
    if track_id is not None:
        keep = [i for i, f in enumerate(frames) if f.track_id == track_id]
        frames = [frames[i] for i in keep]
        images = [images[i] for i in keep]
    if not frames:
        return []

    episodes: List[Episode] = []
    cur_f, cur_i = [frames[0]], [images[0]]
    for prev, f, img in zip(frames, frames[1:], images[1:]):
        if (f.t - prev.t) > max_gap or f.track_id != prev.track_id:
            episodes.append(Episode(cur_f, cur_i))
            cur_f, cur_i = [], []
        cur_f.append(f)
        cur_i.append(img)
    episodes.append(Episode(cur_f, cur_i))
    return [e for e in episodes if e.duration >= min_duration]


def wrist_view(
    frame: np.ndarray,
    keypoints_px: np.ndarray,
    out_size: int = 128,
    zoom: float = 3.2,
) -> np.ndarray:
    """Synthesize the eye-in-hand camera the robot has and the demonstrator does not.

    A crop that follows the wrist, rotated so the hand points a fixed way in the
    image and scaled by the hand's apparent size so it fills a constant fraction of
    the frame. That is what a hand-mounted camera sees: a view where the gripper is
    always in the same place and the world moves around it.

    Both corrections matter. Without the rotation the "wrist" view spins whenever the
    demonstrator turns their hand, which is the one motion a real wrist camera cannot
    see. Without the scaling the object grows and shrinks with distance from the
    third-person camera — an artifact of the recording rig that has nothing to do
    with the manipulation and that a policy will happily key on.

    It is an approximation of a real wrist camera, not a substitute for one: it has
    the third-person camera's viewpoint and occlusions. Its value is that the wrist
    stream carries the hand-centred information a wrist camera carries, so an
    architecture that uses one has something to use.
    """
    h, w = frame.shape[:2]
    center = np.asarray(keypoints_px[WRIST], dtype=np.float64)
    span = float(np.median(np.linalg.norm(keypoints_px[list(MCP_IDS)] - keypoints_px[WRIST], axis=-1)))
    half = max(zoom * span / 2.0, 8.0)

    direction = np.asarray(keypoints_px[MIDDLE_MCP], dtype=np.float64) - center
    theta = float(np.arctan2(direction[1], direction[0])) if np.linalg.norm(direction) > 1e-6 else 0.0
    cos, sin = np.cos(theta), np.sin(theta)

    axis = np.linspace(-half, half, out_size)
    u, v = np.meshgrid(axis, axis)
    xs = np.clip(np.round(center[0] + u * cos - v * sin).astype(np.int64), 0, w - 1)
    ys = np.clip(np.round(center[1] + u * sin + v * cos).astype(np.int64), 0, h - 1)
    return np.ascontiguousarray(frame[ys, xs])


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == size and w == size:
        return np.ascontiguousarray(img)
    ys = (np.arange(size) * h / size).astype(np.int64).clip(0, h - 1)
    xs = (np.arange(size) * w / size).astype(np.int64).clip(0, w - 1)
    return np.ascontiguousarray(img[ys][:, xs])


def build_observations(
    episode: Episode,
    retargeted: RetargetedEpisode,
    image_size: int = 128,
    wrist_zoom: float = 3.2,
) -> Dict[str, np.ndarray]:
    """Assemble the per-timestep arrays on the *control* clock.

    `retargeted.source_index` is what keeps images and actions aligned: the actions
    live on the 20 Hz control grid, the images arrived at whatever the camera ran at,
    and each control step takes the frame nearest to it in time. Zipping the two
    sequences by position instead would drift by the ratio of the two rates, which on
    a 30 fps recording is a third of a second by the end of a ten-second demo.
    """
    n = len(retargeted)
    agent = np.zeros((n, image_size, image_size, 3), dtype=np.uint8)
    wrist = np.zeros((n, image_size, image_size, 3), dtype=np.uint8)
    proprio = np.zeros((n, 7), dtype=np.float32)
    gripper = np.zeros((n, 2), dtype=np.float32)
    curls = np.zeros((n, 5), dtype=np.float32)
    confidence = np.zeros((n,), dtype=np.float32)

    for k in range(n):
        i = int(retargeted.source_index[k])
        frame, tf = episode.images[i], episode.frames[i]
        # Stored bottom-up to match LIBERO; the dataset flips both back together.
        agent[k] = _resize(frame, image_size)[::-1]
        wrist[k] = wrist_view(frame, tf.keypoints_px, image_size, wrist_zoom)[::-1]
        proprio[k] = tf.pose.as_vector()
        a = float(retargeted.apertures[k])
        # LIBERO's gripper_states is a two-finger qpos pair, equal and opposite.
        gripper[k] = (a * 0.04, -a * 0.04)
        curls[k] = tf.pose.curls
        confidence[k] = tf.pose.frame_confidence

    return {
        "agentview_rgb": agent,
        "eye_in_hand_rgb": wrist,
        "joint_states": proprio,
        "gripper_states": gripper,
        "finger_curls": curls,
        "frame_confidence": confidence,
    }


def write_demos(
    path: str | Path,
    episodes: Sequence[Episode],
    instruction: str,
    cfg: Optional[RetargetConfig] = None,
    image_size: int = 128,
    wrist_zoom: float = 3.2,
    source: str = "",
    verbose: bool = True,
) -> Dict[str, object]:
    """Write one task file. Returns a manifest of what went in and what was rejected."""
    cfg = cfg or RetargetConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kept: List[tuple] = []
    rejected: List[Dict[str, object]] = []
    for idx, ep in enumerate(episodes):
        rt = retarget(ep.frames, cfg)
        if len(rt) < 2:
            rejected.append({"episode": idx, "reason": "too few usable frames after confidence gating"})
            continue
        if rt.max_gap_s > INTERPOLATION_GAP_LIMIT:
            rejected.append({
                "episode": idx,
                "reason": f"{rt.max_gap_s:.2f}s with no usable pose — too long to interpolate across",
            })
            continue
        clipped = float(rt.clipped.mean())
        if clipped > CLIPPED_FRACTION_LIMIT:
            # Not a warning. Above this the actions and the images describe different
            # motions, and a demo like that teaches a policy to under-reach.
            rejected.append({"episode": idx, "reason": f"{clipped:.0%} of steps exceeded the controller limit"})
            continue
        kept.append((ep, rt))

    if not kept:
        raise ValueError(
            f"no usable episodes for {path.name}.\n"
            + "\n".join(f"  episode {r['episode']}: {r['reason']}" for r in rejected)
            + "\nSlow the demonstration down, or keep the palm facing the camera so the "
            "pose estimate stays confident."
        )

    total = 0
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, (ep, rt) in enumerate(kept):
            g = data.create_group(f"demo_{i}")
            n = len(rt)
            g.create_dataset("actions", data=np.clip(rt.actions, -1.0, 1.0).astype(np.float32))
            g.create_dataset("dones", data=np.concatenate([np.zeros(n - 1), [1]]).astype(np.int64))
            g.create_dataset("rewards", data=np.zeros(n, dtype=np.float32))
            g.create_dataset("ee_positions", data=rt.positions.astype(np.float32))
            g.create_dataset("coasted", data=rt.coasted.astype(np.int64))
            g.create_dataset("clipped", data=rt.clipped.astype(np.int64))
            obs = g.create_group("obs")
            for key, arr in build_observations(ep, rt, image_size, wrist_zoom).items():
                obs.create_dataset(key, data=arr, compression="gzip", compression_opts=1)
            g.attrs["summary"] = json.dumps(rt.summary())
            total += n

        data.attrs["problem_info"] = json.dumps(
            {"language_instruction": instruction, "problem_name": "hand_demo"}
        )
        data.attrs["total"] = total
        # The stamp the dataset checks. Without it, nine numbers that mean wrist pose
        # are indistinguishable from nine numbers that mean joint angles.
        data.attrs["proprio_source"] = PROPRIO_SOURCE
        data.attrs["proprio_layout"] = json.dumps(
            {"wrist_xyz": [0, 3], "wrist_rotvec": [3, 6], "hand_scale": [6, 7], "gripper": [7, 9]}
        )
        data.attrs["robobench_hand_version"] = FORMAT_VERSION
        data.attrs["source"] = source
        data.attrs["retarget_config"] = json.dumps(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.__dict__.items()}
        )

    manifest = {
        "path": str(path),
        "instruction": instruction,
        "demos": len(kept),
        "timesteps": total,
        "rejected": rejected,
        "summaries": [rt.summary() for _, rt in kept],
    }
    if verbose:
        print(f"wrote {path}  ({len(kept)} demos, {total} steps)")
        for r in rejected:
            print(f"  rejected episode {r['episode']}: {r['reason']}")
    return manifest
