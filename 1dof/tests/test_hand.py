"""Hand tracking: no camera, no MediaPipe, no downloads. `pytest tests/ -q`.

Same budget as the rest of the suite, which is possible because `SyntheticDetector`
produces the detector's *output* rather than requiring its input. Every stage after
detection — filtering, association, pose, retargeting, demo writing — is exercised
against a known ground truth, and the last test runs the result through the real
dataset reader.

The cases here are the ones that fail silently. A pipeline that mirrors a rotation,
swaps two hands for a frame, deletes slow motion, or writes images upside down
produces demos that look fine, train without error, and yield a policy that does
not work — with nothing in the loss curve to say which of the four went wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robobench.hand.detect import Detection, SyntheticDetector  # noqa: E402
from robobench.hand.filters import ConstantVelocity, OneEuroFilter  # noqa: E402
from robobench.hand.landmarks import (  # noqa: E402
    INDEX_TIP,
    NUM_LANDMARKS,
    PALM_IDS,
    THUMB_TIP,
    WRIST,
    canonical_palm,
)
from robobench.hand.pose import (  # noqa: E402
    estimate_pose,
    grip_aperture,
    hand_scale,
    palm_frame,
    rotation_matrix,
    rotation_vector,
)
from robobench.hand.record import segment_episodes, wrist_view, write_demos  # noqa: E402
from robobench.hand.retarget import CAMERA_TO_ROBOT, RetargetConfig, integrate, retarget  # noqa: E402
from robobench.hand.tracker import HandTracker  # noqa: E402


# -- fixtures ---------------------------------------------------------------


def flat_hand(handedness: str = "right", spread: float = 1.0) -> np.ndarray:
    """A hand in the canonical frame: palm on the template, thumb `spread` from index."""
    kp = np.zeros((NUM_LANDMARKS, 3))
    palm = canonical_palm(handedness)
    for i, pid in enumerate(PALM_IDS):
        kp[pid] = palm[i]
    kp[INDEX_TIP] = np.array([0.165, 0.022 if handedness == "right" else -0.022, 0.0])
    far = np.array([0.115, 0.115 if handedness == "right" else -0.115, 0.0])
    kp[THUMB_TIP] = kp[INDEX_TIP] + spread * (far - kp[INDEX_TIP])
    return kp


def track_synthetic(detector: SyntheticDetector, **tracker_kwargs):
    tracker = HandTracker(**tracker_kwargs)
    frames, images = [], []
    for image, t in detector.frames():
        for tf in tracker.update(detector.detect(image, t), t):
            frames.append(tf)
            images.append(image)
    return frames, images


# -- filtering --------------------------------------------------------------


def test_filter_suppresses_jitter_on_a_still_hand():
    """Detector noise becomes action noise of twice the amplitude once differenced."""
    rng = np.random.default_rng(0)
    t = np.arange(150) / 30.0
    noisy = rng.normal(0.0, 0.003, (150, NUM_LANDMARKS, 3))
    filt = OneEuroFilter()
    out = np.array([filt(noisy[i], t[i]) for i in range(len(t))])
    assert out[10:].std() < noisy[10:].std() / 2.0


def test_filter_adapts_to_speed():
    """beta > 0 must buy back the lag a plain low-pass costs on fast motion."""
    t = np.arange(120) / 30.0
    ramp = np.stack([t * 0.5] * 3, axis=-1)
    errs = {}
    for beta in (0.0, 0.7):
        filt = OneEuroFilter(beta=beta)
        out = np.array([filt(ramp[i], t[i]) for i in range(len(t))])
        errs[beta] = np.abs(out[30:] - ramp[30:]).mean()
    assert errs[0.7] < errs[0.0]


def test_filter_survives_a_variable_frame_rate():
    """A webcam drops frames; a filter that assumes a fixed dt changes its own
    bandwidth when that happens."""
    rng = np.random.default_rng(1)
    times = np.cumsum(rng.uniform(0.02, 0.08, 200))
    filt = OneEuroFilter()
    out = np.array([filt(np.full(3, 0.5) + rng.normal(0, 0.002, 3), t) for t in times])
    assert np.isfinite(out).all()
    assert np.abs(out[20:] - 0.5).max() < 0.01


def test_filter_ignores_a_repeated_timestamp():
    filt = OneEuroFilter()
    filt(np.zeros(3), 0.0)
    a = filt(np.ones(3), 0.1)
    b = filt(np.full(3, 5.0), 0.1)  # same timestamp: dt = 0 would divide by zero
    assert np.allclose(a, b) and np.isfinite(b).all()


def test_constant_velocity_coasting_decays():
    cv = ConstantVelocity(damping=0.8)
    for i in range(5):
        cv.update(np.array([[0.1 * i, 0.0, 0.0]]), i / 30.0)
    steps = [cv.coast((5 + k) / 30.0) for k in range(6)]
    advances = [np.linalg.norm(steps[k + 1] - steps[k]) for k in range(len(steps) - 1)]
    assert all(advances[k + 1] < advances[k] for k in range(len(advances) - 1))


# -- pose -------------------------------------------------------------------


@pytest.mark.parametrize("handedness", ["left", "right"])
def test_palm_frame_recovers_a_known_rotation(handedness):
    r_true = rotation_matrix(np.array([0.3, -0.6, 0.9]))
    kp = flat_hand(handedness) @ r_true.T + np.array([0.4, -0.1, 0.7])
    r_est, conf = palm_frame(kp, handedness)
    assert np.abs(r_est - r_true).max() < 1e-6
    assert conf > 0.95


def test_palm_frame_is_a_proper_rotation():
    kp = flat_hand() @ rotation_matrix(np.array([1.0, 2.0, -0.5])).T
    r, _ = palm_frame(kp)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_wrong_handedness_template_is_confidently_wrong():
    """Why handedness is a majority vote over a track and not the latest frame.

    Fitting a left hand with the right-hand template returns full confidence and a
    rotation that is half a turn out. Nothing downstream can detect it, so the
    tracker must never let a single misclassified frame select the template.
    """
    kp = flat_hand("left")
    _, conf_right_template = palm_frame(kp, "right")
    r_wrong, _ = palm_frame(kp, "right")
    error_deg = np.degrees(np.linalg.norm(rotation_vector(r_wrong)))
    assert conf_right_template > 0.9, "the wrong template still fits — that is the trap"
    assert error_deg > 90.0


def test_palm_frame_confidence_falls_monotonically_with_foreshortening():
    """The cheap proxy (ratio of singular values) is not monotone here; the fit
    residual is. Guards against anyone swapping it back."""
    kp = flat_hand()
    confs = []
    for tilt in (0, 30, 45, 60, 75, 89):
        obs = kp @ rotation_matrix(np.array([0.0, np.deg2rad(tilt), 0.0])).T
        obs[:, 2] *= 0.02  # monocular collapse of depth
        confs.append(palm_frame(obs)[1])
    assert confs == sorted(confs, reverse=True)
    assert confs[0] > 0.95 and confs[-1] < 0.3


def test_hand_scale_is_invariant_to_pose_and_linear_in_size():
    kp = flat_hand()
    moved = kp @ rotation_matrix(np.array([0.5, 0.5, 0.5])).T + np.array([1.0, 2.0, 3.0])
    assert hand_scale(moved) == pytest.approx(hand_scale(kp), rel=1e-9)
    assert hand_scale(kp * 1.5) == pytest.approx(hand_scale(kp) * 1.5, rel=1e-9)


def test_hand_scale_resists_one_bad_landmark():
    """A median, not a mean: scale divides into the aperture, so a knuckle thrown
    10 cm away would otherwise read downstream as a phantom grasp."""
    kp = flat_hand()
    broken = kp.copy()
    broken[PALM_IDS[2]] += np.array([0.10, 0.0, 0.0])
    assert hand_scale(broken) == pytest.approx(hand_scale(kp), rel=0.05)


def test_aperture_is_monotone_and_spans_the_range():
    values = [grip_aperture(flat_hand(spread=s)) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0, abs=1e-6)
    assert values[-1] == pytest.approx(1.0, abs=1e-6)


def test_aperture_is_independent_of_hand_size():
    """The same gesture from a large and a small hand must read the same, or the
    gripper command tracks the demonstrator rather than the task."""
    small, large = flat_hand(spread=0.5) * 0.8, flat_hand(spread=0.5) * 1.25
    assert grip_aperture(small) == pytest.approx(grip_aperture(large), abs=1e-6)


def test_pose_rejects_malformed_keypoints():
    with pytest.raises(ValueError, match=r"shape \(21, 3\)"):
        estimate_pose(np.zeros((17, 3)))


# -- overlay ----------------------------------------------------------------


def _one_track_frame(coasted=False, confidence=1.0):
    from robobench.hand.pose import estimate_pose
    from robobench.hand.tracker import TrackFrame

    kp = flat_hand() + np.array([0.0, 0.0, 0.55])
    pose = estimate_pose(kp, "right")
    pose.frame_confidence = confidence
    px = np.stack([kp[:, 0] * 554.0 / kp[:, 2] + 320.0, kp[:, 1] * 554.0 / kp[:, 2] + 240.0], -1)
    return TrackFrame(t=0.0, track_id=0, handedness="right", pose=pose, keypoints=kp,
                      keypoints_px=px, score=0.9, coasted=coasted, age=1)


def test_overlay_colour_reports_the_gate_not_decoration():
    """Green/amber/grey is the confidence gate made visible: if the colour did not
    follow the gate, the preview would show a usable-looking hand for frames the
    pipeline is about to discard."""
    from robobench.hand.draw import COASTED, GOOD, LOW_CONFIDENCE, track_colour

    assert track_colour(_one_track_frame(), 0.5) == GOOD
    assert track_colour(_one_track_frame(confidence=0.2), 0.5) == LOW_CONFIDENCE
    assert track_colour(_one_track_frame(coasted=True), 0.5) == COASTED


def test_overlay_draws_without_mutating_the_source_frame():
    from robobench.hand.draw import draw_landmarks

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_landmarks(frame, [_one_track_frame()])
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert frame.max() == 0, "the overlay wrote into the caller's frame"
    assert out.max() > 0, "nothing was drawn"


def test_overlay_survives_landmarks_off_screen():
    """A hand leaving frame must not crash the preview mid-session."""
    from robobench.hand.draw import draw_hud, draw_landmarks

    tf = _one_track_frame()
    tf.keypoints_px = tf.keypoints_px + np.array([5000.0, -5000.0])
    out = draw_landmarks(np.zeros((240, 320, 3), np.uint8), [tf])
    assert out.shape == (240, 320, 3)
    assert draw_hud(out, [tf], 30.0).shape == (240, 320, 3)


# -- frame sources ----------------------------------------------------------


def _write_clip(path, n_frames, fps, size=(128, 160)):
    iio = pytest.importorskip("imageio.v3")
    pytest.importorskip("imageio_ffmpeg")
    rng = np.random.default_rng(0)
    frames = np.stack([rng.integers(0, 80, (*size, 3), dtype=np.uint8) for _ in range(n_frames)])
    iio.imwrite(str(path), frames, fps=fps)
    return path


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_video_frame_rate_is_read_not_assumed(tmp_path, fps):
    """Regression: an earlier version asked imageio for the `pyav` plugin, which is
    not installed by default, and fell back to 30.0 whenever that failed — so every
    video read as 30 fps. A 60 fps clip then got timestamps twice as long as the real
    ones, halving every velocity in the demo, with nothing downstream able to notice."""
    from robobench.hand.sources import probe_fps

    clip = _write_clip(tmp_path / f"c{fps}.mp4", 30, fps)
    assert probe_fps(clip) == pytest.approx(fps, abs=0.5)


@pytest.mark.parametrize("fps", [30, 60])
def test_video_timestamps_match_the_real_rate(tmp_path, fps):
    from robobench.hand.sources import video_frames

    clip = _write_clip(tmp_path / f"c{fps}.mp4", fps, fps)
    times = [t for _, t in video_frames(clip)]
    assert len(times) > 1
    assert (times[1] - times[0]) == pytest.approx(1.0 / fps, rel=1e-6)
    assert times[-1] == pytest.approx((len(times) - 1) / fps, rel=1e-6)


def test_explicit_fps_overrides_the_probe(tmp_path):
    from robobench.hand.sources import video_frames

    clip = _write_clip(tmp_path / "c.mp4", 20, 30)
    times = [t for _, t in video_frames(clip, fps=10.0)]
    assert (times[1] - times[0]) == pytest.approx(0.1)


def test_missing_video_says_so_plainly(tmp_path):
    from robobench.hand.sources import video_frames

    with pytest.raises(SystemExit, match="no such video file"):
        list(video_frames(tmp_path / "nope.mp4"))


def test_video_respects_the_seconds_limit(tmp_path):
    from robobench.hand.sources import video_frames

    clip = _write_clip(tmp_path / "c.mp4", 90, 30)
    assert len(list(video_frames(clip, max_seconds=1.0))) <= 31


def test_camera_error_names_the_actual_fix():
    from robobench.hand.sources import camera_error

    msg = camera_error(0)
    assert "--synthetic" in msg and "--video" in msg
    if sys.platform == "darwin":
        assert "Privacy & Security" in msg


# -- monocular lifting ------------------------------------------------------
# The MediaPipe class itself needs the optional dependency, but the geometry it
# relies on does not — and the geometry is the part that can be quietly wrong.


def _project(keypoints_cam, focal_px, width, height):
    z = np.maximum(keypoints_cam[:, 2], 1e-6)
    return np.stack([keypoints_cam[:, 0] * focal_px / z + width / 2.0,
                     keypoints_cam[:, 1] * focal_px / z + height / 2.0], axis=-1)


@pytest.mark.parametrize("depth", [0.35, 0.6, 1.2])
def test_lifting_recovers_a_known_depth(depth):
    """World landmarks are wrist-centred, so position has to come from apparent size."""
    from robobench.hand.detect import lift_to_camera_frame

    focal, w, h = 554.0, 640, 480
    truth = flat_hand() + np.array([0.08, -0.05, depth])
    px = _project(truth, focal, w, h)
    local = truth - truth[WRIST]

    lifted = lift_to_camera_frame(local, px, focal, w, h)
    assert lifted is not None
    assert np.abs(lifted[WRIST] - truth[WRIST]).max() < 1e-3


def test_lifted_depth_is_unchanged_by_closing_the_hand():
    """Why the depth measurement uses a palm span and not a fingertip distance: a
    fingertip span shrinks when the hand closes, for a reason that has nothing to do
    with distance, and the demo would contain a forward jab at every grasp."""
    from robobench.hand.detect import lift_to_camera_frame

    focal, w, h = 554.0, 640, 480
    depths = []
    for spread in (1.0, 0.5, 0.0):
        truth = flat_hand(spread=spread) + np.array([0.0, 0.0, 0.6])
        px = _project(truth, focal, w, h)
        lifted = lift_to_camera_frame(truth - truth[WRIST], px, focal, w, h)
        depths.append(lifted[WRIST, 2])
    assert max(depths) - min(depths) < 1e-6, f"grasping moved the hand in depth: {depths}"


def test_lifting_refuses_a_hand_too_small_to_measure():
    """Rather than dividing by a near-zero pixel span and reporting a hand metres away."""
    from robobench.hand.detect import lift_to_camera_frame

    tiny = np.tile(np.array([320.0, 240.0]), (NUM_LANDMARKS, 1))
    assert lift_to_camera_frame(flat_hand(), tiny, 554.0, 640, 480) is None


class _FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeCategory:
    def __init__(self, name, score):
        self.category_name, self.score = name, score


class _FakeResult:
    def __init__(self, hand_landmarks, hand_world_landmarks, handedness):
        self.hand_landmarks = hand_landmarks
        self.hand_world_landmarks = hand_world_landmarks
        self.handedness = handedness


class _FakeLandmarker:
    """Stands in for mediapipe's HandLandmarker, emitting its result shape."""

    def __init__(self, result):
        self.result = result
        self.timestamps = []

    def detect_for_video(self, image, ms):
        self.timestamps.append(ms)
        return self.result

    def close(self):
        pass


def _stub_detector(result, focal_px=554.0):
    """A MediaPipeDetector with the native landmarker replaced.

    Built with __new__ so no model file or mediapipe install is needed: the point is
    to exercise the adapter — the code that turns mediapipe's result objects into
    Detections — which is where an upstream API change lands and which nothing else
    in this suite would touch.
    """
    import types

    from robobench.hand.detect import MediaPipeDetector

    det = MediaPipeDetector.__new__(MediaPipeDetector)
    det._landmarker = _FakeLandmarker(result)
    det._last_ms = -1
    det.focal_px = focal_px
    det.hfov_degrees = 60.0
    det._mp = types.SimpleNamespace(
        Image=lambda image_format, data: data, ImageFormat=types.SimpleNamespace(SRGB=0)
    )
    return det


def _fake_hand_result(depth=0.6, focal_px=554.0, w=640, h=480, label="Right"):
    truth = flat_hand() + np.array([0.05, -0.02, depth])
    z = np.maximum(truth[:, 2], 1e-6)
    px = np.stack([truth[:, 0] * focal_px / z + w / 2.0, truth[:, 1] * focal_px / z + h / 2.0], -1)
    local = truth - truth[WRIST]
    return _FakeResult(
        hand_landmarks=[[_FakeLandmark(p[0] / w, p[1] / h, 0.0) for p in px]],
        hand_world_landmarks=[[_FakeLandmark(*q) for q in local]],
        handedness=[[_FakeCategory(label, 0.98)]],
    ), truth


def test_detector_adapter_unpacks_landmarks_correctly():
    """The MediaPipe classes need the optional dependency; the adapter around them
    does not, and the adapter is what an upstream API change breaks."""
    result, truth = _fake_hand_result()
    det = _stub_detector(result)

    out = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), 0.0)
    assert len(out) == 1
    d = out[0]
    assert d.handedness == "right"
    assert d.score == pytest.approx(0.98)
    assert d.keypoints.shape == (NUM_LANDMARKS, 3)
    assert d.keypoints_px.shape == (NUM_LANDMARKS, 2)
    assert np.abs(d.keypoints[WRIST] - truth[WRIST]).max() < 2e-3


def test_detector_adapter_lowercases_handedness():
    """MediaPipe says 'Right'; the palm templates are keyed on 'right'. A mismatch
    here selects the wrong template and yields a confident 180-degree error."""
    result, _ = _fake_hand_result(label="Left")
    assert _stub_detector(result).detect(np.zeros((480, 640, 3), np.uint8), 0.0)[0].handedness == "left"


def test_detector_timestamps_are_strictly_increasing():
    """VIDEO mode raises on a repeated millisecond timestamp, which would abort a
    whole recording rather than drop one frame."""
    result, _ = _fake_hand_result()
    det = _stub_detector(result)
    for t in (0.0, 0.0, 0.0, 0.0001):
        det.detect(np.zeros((480, 640, 3), np.uint8), t)
    stamps = det._landmarker.timestamps
    assert all(b > a for a, b in zip(stamps, stamps[1:])), stamps


def test_mediapipe_detector_reports_a_missing_model_clearly():
    """The Tasks API takes an explicit model file; the wheel stopped bundling weights."""
    from robobench.hand.detect import MediaPipeDetector

    pytest.importorskip("mediapipe")
    with pytest.raises(SystemExit, match="make hand-model"):
        MediaPipeDetector(model_path="/nonexistent/hand_landmarker.task")


# -- tracking ---------------------------------------------------------------


def test_track_identity_survives_an_occlusion():
    det = SyntheticDetector(duration=5.0, noise_m=0.003, occlusions=[(2.0, 2.4)], seed=2)
    frames, _ = track_synthetic(det, min_hits=1)
    assert {f.track_id for f in frames} == {0}
    assert any(f.coasted for f in frames), "the occlusion should have been coasted through"


def test_long_occlusion_ends_the_track():
    det = SyntheticDetector(duration=6.0, noise_m=0.003, occlusions=[(2.0, 3.5)], seed=2)
    frames, _ = track_synthetic(det, min_hits=1)
    assert len({f.track_id for f in frames}) == 2, "coasting for 1.5 s is an invention, not a track"


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_coasting_tolerance_is_in_seconds_not_frames(fps):
    """The same occlusion must behave the same way on a faster camera."""
    det = SyntheticDetector(duration=5.0, fps=fps, noise_m=0.003, occlusions=[(2.0, 2.4)], seed=2)
    frames, _ = track_synthetic(det, min_hits=1)
    assert {f.track_id for f in frames} == {0}


def test_two_hands_keep_distinct_stable_identities():
    """One swapped frame makes that frame's action the vector between two different
    hands — a metre-scale step in a millimetre-scale trajectory."""
    det = SyntheticDetector(duration=5.0, noise_m=0.003, num_hands=2, seed=3)
    frames, _ = track_synthetic(det, min_hits=1, max_hands=2)
    by_id = {}
    for f in frames:
        by_id.setdefault(f.track_id, set()).add(f.handedness)
    assert len(by_id) == 2
    assert all(len(labels) == 1 for labels in by_id.values())


def test_handedness_is_a_vote_not_the_latest_frame():
    tracker = HandTracker(min_hits=1)
    kp = flat_hand("right")
    px = np.tile(np.array([320.0, 240.0]), (NUM_LANDMARKS, 1))
    for i in range(10):
        # One frame in ten is misclassified, as a real classifier does under occlusion.
        label = "left" if i == 5 else "right"
        out = tracker.update([Detection(kp + np.array([0.0, 0.0, 0.5]), px, label)], i / 30.0)
        assert out[0].handedness == "right"


def test_spurious_single_frame_detections_are_suppressed():
    tracker = HandTracker(min_hits=3)
    px = np.tile(np.array([320.0, 240.0]), (NUM_LANDMARKS, 1))
    kp = flat_hand() + np.array([0.0, 0.0, 0.5])
    assert tracker.update([Detection(kp, px, "right")], 0.0) == []
    assert tracker.update([Detection(kp, px, "right")], 1 / 30) == []
    assert len(tracker.update([Detection(kp, px, "right")], 2 / 30)) == 1


# -- retargeting ------------------------------------------------------------


def test_retarget_round_trips_exactly_without_a_deadband():
    """Every step of the transform looks plausible alone; the composition is where a
    transposed frame or a right-multiplied rotation hides."""
    cfg = RetargetConfig(deadband_m=0.0, deadband_rad=0.0)
    frames, _ = track_synthetic(SyntheticDetector(duration=6.0, noise_m=0.0), min_hits=1)
    ep = retarget(frames, cfg)
    pos, rot = integrate(ep.actions[:-1], ep.positions[0], ep.rotations[0], cfg)
    assert np.abs(pos - ep.positions).max() < 1e-6
    worst = max(
        np.degrees(np.linalg.norm(rotation_vector(rot[i] @ ep.rotations[i].T))) for i in range(len(pos))
    )
    assert worst < 1e-3


def test_deadband_error_is_bounded_not_accumulating():
    cfg = RetargetConfig()
    frames, _ = track_synthetic(SyntheticDetector(duration=8.0, noise_m=0.0), min_hits=1)
    ep = retarget(frames, cfg)
    pos, _ = integrate(ep.actions[:-1], ep.positions[0], ep.rotations[0], cfg)
    assert np.abs(pos - ep.positions).max() < cfg.deadband_m * 1.5


@pytest.mark.parametrize("speed", [0.01, 0.03])
def test_slow_motion_survives_the_deadband(speed):
    """A discarding deadband deletes a careful approach outright — 3 cm/s at 20 Hz is
    1.5 mm per step, under any threshold high enough to suppress jitter."""

    class Slow(SyntheticDetector):
        def wrist_path(self, t, hand_index=0):
            return np.array([-0.10 + speed * t, 0.0, 0.55])

    frames, _ = track_synthetic(Slow(duration=6.0, noise_m=0.0), min_hits=1)
    ep = retarget(frames)
    commanded = np.linalg.norm(ep.actions[:, :3] * ep_max_pos(), axis=-1).sum()
    travelled = speed * (ep.times[-1] - ep.times[0])
    assert commanded > 0.9 * travelled


def ep_max_pos() -> float:
    return RetargetConfig().max_pos


def test_jitter_does_not_accumulate_through_the_carry():
    """The carry preserves real displacement, so it must not preserve noise too.

    The property that matters is that the error does not grow: detector noise is
    zero-mean, so the carry fills and empties around zero instead of integrating into
    a drift. Some phantom motion is unavoidable — that is the price of not deleting
    slow motion — and at 3 mm landmark noise it comes to a few percent of a real
    demo's path, spent going nowhere.
    """
    real, _ = track_synthetic(SyntheticDetector(duration=6.0, noise_m=0.003, seed=11), min_hits=1)
    real_path = retarget(real).summary()["path_length_m"]

    still = SyntheticDetector(duration=6.0, noise_m=0.003, seed=11)
    still.wrist_path = lambda t, hand_index=0: np.array([0.0, 0.0, 0.55])
    frames, _ = track_synthetic(still, min_hits=1)
    ep = retarget(frames)

    drift = np.linalg.norm(ep.positions[-1] - ep.positions[0])
    phantom = np.linalg.norm(ep.actions[:, :3] * ep_max_pos(), axis=-1).sum()
    assert drift < 0.01, "the carry is integrating noise into a drift"
    assert phantom < 0.20 * real_path


@pytest.mark.parametrize("noise_m", [0.003, 0.010, 0.020])
def test_actions_are_always_within_the_controller_range(noise_m):
    frames, _ = track_synthetic(SyntheticDetector(duration=4.0, noise_m=noise_m, seed=5), min_hits=1)
    ep = retarget(frames)
    assert len(ep) > 0
    assert np.isfinite(ep.actions).all()
    assert ep.actions.min() >= -1.0 and ep.actions.max() <= 1.0


def test_hopeless_tracking_produces_no_episode_rather_than_garbage():
    """5 cm of landmark noise is not a hand. The confidence gate must empty the
    episode, so `write_demos` rejects it, rather than emitting clipped nonsense that
    looks like a demo and trains like one."""
    frames, _ = track_synthetic(SyntheticDetector(duration=4.0, noise_m=0.05, seed=5), min_hits=1)
    assert frames, "the tracker should still follow something; it is the pose that is hopeless"
    assert len(retarget(frames)) == 0


def test_retarget_resamples_onto_the_control_clock():
    """A 30 fps recording retargeted frame-to-frame runs 1.5x fast in a 20 Hz sim."""
    for fps in (24.0, 30.0, 60.0):
        frames, _ = track_synthetic(SyntheticDetector(duration=5.0, fps=fps, noise_m=0.0), min_hits=1)
        ep = retarget(frames, RetargetConfig(control_hz=20.0))
        assert len(ep) == pytest.approx(100, abs=4), f"{fps} fps produced {len(ep)} control steps"


def test_camera_to_robot_maps_the_axes_it_claims_to():
    c = CAMERA_TO_ROBOT
    assert np.allclose(c @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0])  # into scene -> forward
    assert np.allclose(c @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0])  # right -> -left
    assert np.allclose(c @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0])  # down -> -up
    assert np.linalg.det(c) == pytest.approx(1.0)


def test_a_reflection_is_refused_as_a_frame():
    """det = -1 mirrors every rotation while leaving positions looking correct."""
    with pytest.raises(ValueError, match="det"):
        RetargetConfig(camera_to_robot=np.diag([1.0, 1.0, -1.0]))


@pytest.mark.parametrize("noise_sd", [0.03, 0.06])
def test_gripper_hysteresis_does_not_chatter(noise_sd):
    """An aperture hovering exactly on the decision boundary is the moment the
    fingers touch the object — the worst possible moment to open and close 100
    times. Measured against the single-threshold version it replaces."""
    from robobench.hand.retarget import _gripper

    rng = np.random.default_rng(0)
    aperture = np.clip(0.45 + rng.normal(0.0, noise_sd, 200), 0.0, 1.0)

    cmd = _gripper(aperture, RetargetConfig())
    single = np.where(aperture < 0.45, 1.0, -1.0)
    toggles = int(np.sum(np.diff(cmd) != 0))
    single_toggles = int(np.sum(np.diff(single) != 0))

    assert set(np.unique(cmd)) <= {-1.0, 1.0}
    assert toggles < single_toggles / 10.0, f"{toggles} vs {single_toggles} for one threshold"


def test_gripper_still_commits_on_a_real_grasp():
    aperture = np.concatenate([np.ones(20), np.linspace(1.0, 0.0, 10), np.zeros(30)])
    from robobench.hand.retarget import _gripper

    cmd = _gripper(aperture, RetargetConfig())
    assert cmd[0] == -1.0 and cmd[-1] == 1.0
    assert int(np.sum(np.diff(cmd) != 0)) == 1


# -- recording --------------------------------------------------------------


def test_episodes_split_on_gaps():
    det = SyntheticDetector(duration=9.0, noise_m=0.003, occlusions=[(3.0, 3.8), (6.0, 6.8)], seed=0)
    frames, images = track_synthetic(det, min_hits=1)
    episodes = segment_episodes(frames, images, max_gap=0.4, min_duration=0.5)
    assert len(episodes) == 3


def _scene_with_object(hand_angle: float, hand_span: float, size: int = 240):
    """An image where a bright object sits at a fixed offset in the HAND's frame.

    As the hand rotates or moves nearer the camera, the object moves with it in image
    space — which is exactly the situation a wrist camera is defined by, and what the
    crop has to undo.
    """
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    wrist = np.array([size / 2.0, size / 2.0])
    cos, sin = np.cos(hand_angle), np.sin(hand_angle)
    rot = np.array([[cos, -sin], [sin, cos]])

    # Object at (1.0, 0.4) hand-spans from the wrist, in the hand's own frame.
    obj = wrist + rot @ (np.array([1.0, 0.4]) * hand_span)
    r = max(int(0.25 * hand_span), 2)
    y, x = int(obj[1]), int(obj[0])
    frame[max(y - r, 0) : y + r, max(x - r, 0) : x + r] = 255

    px = np.zeros((NUM_LANDMARKS, 2))
    px[WRIST] = wrist
    from robobench.hand.landmarks import MCP_IDS, MIDDLE_MCP

    for k, mid in enumerate(MCP_IDS):
        offset = np.array([1.0, (k - 1.5) * 0.15]) * hand_span
        px[mid] = wrist + rot @ offset
    px[MIDDLE_MCP] = wrist + rot @ np.array([hand_span, 0.0])
    return frame, px


def test_wrist_view_cancels_hand_rotation():
    """Without the rotation, the synthesized wrist view spins whenever the
    demonstrator turns their hand — the one motion a real wrist camera cannot see."""
    views = [wrist_view(*_scene_with_object(a, 40.0), out_size=48) for a in (0.0, 1.0, 2.5)]
    ref = views[0].astype(np.int16)
    for v in views[1:]:
        overlap = np.mean((v.astype(np.int16) > 128) == (ref > 128))
        assert overlap > 0.9, f"the object moved in the wrist view: {overlap:.2f} agreement"


def test_wrist_view_cancels_distance_from_the_camera():
    """Otherwise the object grows and shrinks with the demonstrator's distance from
    the rig — an artifact of the recording setup that a policy will key on."""
    near = wrist_view(*_scene_with_object(0.0, 60.0), out_size=48)
    far = wrist_view(*_scene_with_object(0.0, 30.0), out_size=48)
    overlap = np.mean((near > 128) == (far > 128))
    assert overlap > 0.9, f"apparent object size tracked camera distance: {overlap:.2f}"


def test_wrist_view_output_contract():
    frame, px = _scene_with_object(0.5, 40.0)
    view = wrist_view(frame, px, out_size=32)
    assert view.shape == (32, 32, 3) and view.dtype == np.uint8


def test_written_demos_load_through_the_real_dataset(tmp_path):
    """The whole point: hand demos must be indistinguishable to everything downstream."""
    from robobench.data.libero_dataset import LiberoChunkDataset, collate

    det = SyntheticDetector(duration=8.0, noise_m=0.003, occlusions=[(4.0, 4.8)], seed=0)
    frames, images = track_synthetic(det, min_hits=1)
    episodes = segment_episodes(frames, images, max_gap=0.4, min_duration=0.5)

    suite = tmp_path / "hand_demos"
    instruction = "pick up the block"
    write_demos(suite / f"{instruction.replace(' ', '_')}_demo.hdf5", episodes, instruction,
                image_size=64, verbose=False)

    rng = np.random.default_rng(0)
    ds = LiberoChunkDataset(
        root=tmp_path, suites=["hand_demos"], chunk_size=4, image_size=64,
        lang_cache={instruction: rng.normal(0, 1, 512).astype(np.float32)}, augment=False,
    )
    meta = ds.describe()
    assert meta["action_dim"] == 7 and meta["proprio_dim"] == 9
    assert meta["proprio_source"] == "hand_pose"

    item = ds[len(ds) // 2]
    assert item["images"]["agentview"].shape == (3, 64, 64)
    assert item["images"]["wrist"].shape == (3, 64, 64)
    assert item["proprio"].shape == (9,)
    batch = collate([ds[i] for i in range(4)])
    assert batch["actions"].abs().max() <= 1.001


def test_mixing_proprio_sources_is_refused(tmp_path):
    """Nine floats meaning wrist pose and nine floats meaning joint angles train
    together without complaint. That is the whole problem."""
    import subprocess

    from robobench.data.libero_dataset import LiberoChunkDataset

    suite = tmp_path / "mixed"
    subprocess.run(
        [sys.executable, "scripts/make_synthetic_data.py", "--out", str(suite),
         "--tasks", "1", "--demos", "1", "--length", "12", "--image-size", "64"],
        check=True, capture_output=True,
    )
    det = SyntheticDetector(duration=4.0, noise_m=0.003, seed=0)
    frames, images = track_synthetic(det, min_hits=1)
    write_demos(suite / "pick_up_the_block_demo.hdf5",
                segment_episodes(frames, images, min_duration=0.5),
                "pick up the block", image_size=64, verbose=False)

    with pytest.raises(ValueError, match="proprioception means different things"):
        LiberoChunkDataset(root=tmp_path, suites=["mixed"], chunk_size=4, image_size=64,
                           require_language=False)


def _synthetic_track(n=80, fps=30.0, low_confidence=()):
    """Hand-built track frames with controlled confidence, so the gating path can be
    tested without relying on a detector to fail in exactly the right way."""
    from robobench.hand.pose import estimate_pose
    from robobench.hand.tracker import TrackFrame

    frames, images = [], []
    for i in range(n):
        t = i / fps
        kp = flat_hand() + np.array([0.002 * i, 0.0, 0.55])
        pose = estimate_pose(kp, "right")
        if any(lo <= t < hi for lo, hi in low_confidence):
            pose.frame_confidence = 0.1
        frames.append(
            TrackFrame(t=t, track_id=0, handedness="right", pose=pose, keypoints=kp,
                       keypoints_px=np.tile([160.0, 120.0], (NUM_LANDMARKS, 1)),
                       score=0.9, coasted=False, age=i)
        )
        images.append(np.zeros((64, 64, 3), dtype=np.uint8))
    return frames, images


def test_confidence_gaps_are_measured_not_silently_interpolated():
    """Confidence gating runs after segmentation, so it opens holes segmentation
    never saw. Resampling would draw a straight line through the part of the motion
    nobody observed."""
    clean, _ = _synthetic_track()
    gapped, _ = _synthetic_track(low_confidence=[(1.0, 1.6)])
    assert retarget(clean).max_gap_s < 0.05
    assert retarget(gapped).max_gap_s > 0.5


def test_a_long_interpolated_gap_is_refused(tmp_path):
    from robobench.hand.record import Episode

    frames, images = _synthetic_track(low_confidence=[(1.0, 1.6)])
    with pytest.raises(ValueError, match="too long to interpolate across"):
        write_demos(tmp_path / "x_demo.hdf5", [Episode(frames, images)], "x",
                    image_size=32, verbose=False)


def test_unusable_episodes_are_refused_not_written(tmp_path):
    det = SyntheticDetector(duration=3.0, noise_m=0.0, seed=0)
    frames, images = track_synthetic(det, min_hits=1)
    episodes = segment_episodes(frames, images, min_duration=0.5)
    # A control rate this low makes every step exceed the controller's limit.
    cfg = RetargetConfig(control_hz=1.0)
    with pytest.raises(ValueError, match="no usable episodes"):
        write_demos(tmp_path / "x_demo.hdf5", episodes, "x", cfg=cfg, image_size=32, verbose=False)
