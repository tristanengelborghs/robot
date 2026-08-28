"""Per-frame detections -> continuous, identity-stable, filtered hand tracks.

A detector answers "what hands are in this frame". A demo needs "where did *this*
hand go", which is a different question and the one this module answers. Three
things stand between the two:

Identity.    With two hands in frame, the detector's output order is not stable and
             its left/right classifier flips under self-occlusion. If the identities
             swap for even one frame, the retargeted action for that frame is the
             vector between two different hands — a metre-scale jump in a trajectory
             whose real steps are millimetres. One such frame poisons a demo.

Gaps.        Hands pass behind objects and leave frame. Dropping those frames leaves
             holes that the action deltas then have to jump across; freezing the pose
             makes the hand teleport on re-acquisition. Coasting on the last velocity
             is right for a few frames and decays gracefully after that, and coasted
             frames are flagged so a demo writer can refuse to train on a long one.

Jitter.      Handled by the 1-Euro filters in `filters`, applied per track so a
             re-acquired hand starts a fresh filter instead of smoothing across the
             gap it was just absent for.

The matching is Hungarian on a gated cost, which is worth the scipy dependency over
greedy nearest-neighbour for exactly one case: two hands close together and moving
past each other. Greedy commits to whichever pair it examines first and can hand
both detections to the same track; the optimal assignment cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from robobench.hand.detect import Detection
from robobench.hand.filters import ConstantVelocity, OneEuroFilter
from robobench.hand.landmarks import WRIST
from robobench.hand.pose import HandPose, estimate_pose

__all__ = ["TrackFrame", "HandTracker", "Track"]

#: Cost added when a detection's handedness disagrees with a track's. Large enough
#: that a same-handed match is always preferred, finite so that a detector that
#: misclassifies a hand for a few frames does not spawn a phantom second track.
HANDEDNESS_PENALTY = 0.30


@dataclass
class TrackFrame:
    """One hand, one timestep, after filtering. The unit `retarget` consumes."""

    t: float
    track_id: int
    handedness: str
    pose: HandPose
    keypoints: np.ndarray  # (21, 3) filtered, metres, camera frame
    keypoints_px: np.ndarray  # (21, 2)
    score: float
    coasted: bool  # True = extrapolated through an occlusion, not observed
    age: int


@dataclass
class Track:
    """Mutable per-hand state. Not part of the public surface."""

    id: int
    handedness: str
    filt: OneEuroFilter
    motion: ConstantVelocity
    last_t: float
    last_seen_t: float  # last frame with an actual detection, not an extrapolation
    keypoints: np.ndarray
    keypoints_px: np.ndarray
    score: float = 1.0
    dead: bool = False
    hits: int = 1
    misses: int = 0
    age: int = 0
    confirmed: bool = False
    handedness_votes: Dict[str, int] = field(default_factory=dict)

    @property
    def wrist(self) -> np.ndarray:
        return self.keypoints[WRIST]

    def vote_handedness(self, label: str) -> None:
        """Majority vote over the track's life, not the latest frame.

        The classifier flips under self-occlusion, and handedness selects the palm
        template — so a single flipped frame produces a confidently wrong 180-degree
        rotation, which is worse than a missing one because nothing downstream can
        detect it.
        """
        self.handedness_votes[label] = self.handedness_votes.get(label, 0) + 1
        self.handedness = max(self.handedness_votes.items(), key=lambda kv: kv[1])[0]


class HandTracker:
    """Feeds on `Detection` lists, emits `TrackFrame` lists.

        tracker = HandTracker()
        for frame, t in source:
            for tf in tracker.update(detector.detect(frame, t), t):
                ...

    gate_m          maximum wrist displacement, in metres, that can still be the
                    same hand. Scaled by the elapsed time, so it survives a variable
                    frame rate. 1.2 m/s is a brisk deliberate reach; above that it is
                    more likely two hands than one fast one.
    max_coast_s     how long, in seconds, a track may be extrapolated before it is
                    killed and a re-acquisition becomes a new hand. In seconds and
                    not frames: the same half-second occlusion must behave the same
                    way whether the camera runs at 30 or 60 fps, and a frame count
                    silently halves the tolerance when someone plugs in a faster
                    camera.
    min_hits        detections before a track is reported at all. Suppresses the
                    single-frame false positives that a detector produces on faces
                    and patterned backgrounds.

    Pose quality is deliberately not a tracker concern. The tracker answers "is this
    the same hand", and a hand whose palm has turned edge-on is still the same hand —
    dropping it here would break the track and split the demo. The confidence gate
    belongs to retargeting, which is where an unusable *pose* actually matters; see
    `RetargetConfig.min_confidence`.
    """

    def __init__(
        self,
        gate_speed: float = 1.2,
        gate_floor_m: float = 0.06,
        max_coast_s: float = 0.5,
        min_hits: int = 3,
        min_score: float = 0.5,
        min_cutoff: float = 1.0,
        beta: float = 0.7,
        max_hands: int = 2,
    ):
        self.gate_speed = gate_speed
        self.gate_floor_m = gate_floor_m
        self.max_coast_s = max_coast_s
        self.min_hits = min_hits
        self.min_score = min_score
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.max_hands = max_hands

        self._tracks: List[Track] = []
        self._next_id = 0
        self._last_t: Optional[float] = None

    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._last_t = None

    # -- association --------------------------------------------------------

    def _gate(self, dt: float) -> float:
        return max(self.gate_floor_m, self.gate_speed * max(dt, 1e-3))

    def _cost_matrix(self, detections: List[Detection], t: float, dt: float) -> np.ndarray:
        cost = np.zeros((len(self._tracks), len(detections)))
        for i, tr in enumerate(self._tracks):
            predicted = tr.motion.predict(t)
            anchor = tr.wrist if predicted is None else predicted[WRIST]
            for j, det in enumerate(detections):
                d = float(np.linalg.norm(det.keypoints[WRIST] - anchor))
                cost[i, j] = d + (0.0 if det.handedness == tr.handedness else HANDEDNESS_PENALTY)
        return cost

    def _match(self, detections: List[Detection], t: float, dt: float):
        """Optimal assignment under a distance gate. Returns (pairs, unmatched dets)."""
        if not self._tracks or not detections:
            return [], list(range(len(detections)))

        cost = self._cost_matrix(detections, t, dt)
        gate = self._gate(dt)

        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(cost)
        except ImportError:  # pragma: no cover - scipy is a declared dependency
            rows, cols = _greedy_assignment(cost)

        pairs, used = [], set()
        for i, j in zip(rows, cols):
            # The gate is on distance alone: a handedness disagreement should cost a
            # match, not forbid one, or a misclassified frame orphans the detection
            # and spawns a duplicate track.
            if cost[i, j] - (0.0 if detections[j].handedness == self._tracks[i].handedness else HANDEDNESS_PENALTY) <= gate:
                pairs.append((i, j))
                used.add(j)
        return pairs, [j for j in range(len(detections)) if j not in used]

    # -- update -------------------------------------------------------------

    def update(self, detections: Iterable[Detection], t: float) -> List[TrackFrame]:
        detections = [d for d in detections if d.score >= self.min_score]
        dt = 1.0 / 30.0 if self._last_t is None else max(float(t) - self._last_t, 1e-3)
        self._last_t = float(t)

        pairs, unmatched = self._match(detections, t, dt)
        matched_tracks = {i for i, _ in pairs}

        for i, j in pairs:
            tr, det = self._tracks[i], detections[j]
            tr.keypoints = tr.filt(det.keypoints, t)
            tr.keypoints_px = det.keypoints_px
            tr.motion.update(tr.keypoints, t)
            tr.vote_handedness(det.handedness)
            tr.score = det.score
            tr.hits += 1
            tr.misses = 0
            tr.age += 1
            tr.last_t = float(t)
            tr.last_seen_t = float(t)
            tr.confirmed = tr.confirmed or tr.hits >= self.min_hits

        for i, tr in enumerate(self._tracks):
            if i in matched_tracks:
                continue
            coasted = tr.motion.coast(t)
            if coasted is None:
                tr.dead = True
                continue
            tr.keypoints = coasted
            tr.misses += 1
            tr.age += 1
            tr.last_t = float(t)

        self._tracks = [
            tr for tr in self._tracks
            if not tr.dead and (float(t) - tr.last_seen_t) <= self.max_coast_s
        ]

        for j in unmatched:
            if len(self._tracks) >= self.max_hands:
                break
            self._spawn(detections[j], t)

        return [self._emit(tr, t) for tr in self._tracks if tr.confirmed]

    def _spawn(self, det: Detection, t: float) -> None:
        filt = OneEuroFilter(min_cutoff=self.min_cutoff, beta=self.beta)
        motion = ConstantVelocity()
        kp = filt(det.keypoints, t)
        motion.update(kp, t)
        tr = Track(
            id=self._next_id,
            handedness=det.handedness,
            filt=filt,
            motion=motion,
            last_t=float(t),
            last_seen_t=float(t),
            keypoints=kp,
            keypoints_px=det.keypoints_px,
            score=det.score,
            confirmed=self.min_hits <= 1,
        )
        tr.vote_handedness(det.handedness)
        self._tracks.append(tr)
        self._next_id += 1

    def _emit(self, tr: Track, t: float) -> TrackFrame:
        pose = estimate_pose(tr.keypoints, tr.handedness)
        return TrackFrame(
            t=float(t),
            track_id=tr.id,
            handedness=tr.handedness,
            pose=pose,
            keypoints=tr.keypoints.copy(),
            keypoints_px=tr.keypoints_px.copy(),
            score=tr.score,
            coasted=tr.misses > 0,
            age=tr.age,
        )


def _greedy_assignment(cost: np.ndarray):  # pragma: no cover - scipy fallback
    rows, cols = [], []
    taken_r, taken_c = set(), set()
    for i, j in sorted(
        ((i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])),
        key=lambda ij: cost[ij],
    ):
        if i not in taken_r and j not in taken_c:
            rows.append(i)
            cols.append(j)
            taken_r.add(i)
            taken_c.add(j)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)
