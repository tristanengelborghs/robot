"""Human hand tracking -> robot manipulation demos.

Records a person doing a task on camera and writes it out as LIBERO-shaped demo
files, so the existing training loop, ablations and statistics work on human video
with no changes. The motivation is arithmetic: a teleoperated LIBERO demo costs
minutes of a robot and an operator, a video demo costs the length of the video, and
behaviour cloning is bounded by how many demos you have.

    python scripts/track_hands.py --video demo.mp4 --instruction "pick up the mug"
    python scripts/track_hands.py --synthetic --instruction "pick up the block"

The pipeline, and what each stage is actually for:

    detect      one frame -> hands. Swappable; MediaPipe by default.
    tracker     hands -> identity-stable, jitter-filtered, occlusion-coasted tracks.
    pose        landmarks -> 6-DoF wrist pose, hand scale, grip aperture.
    retarget    pose trajectory -> 7-D OSC actions on the controller's clock.
    record      tracks + video -> LIBERO HDF5.

What this gives you and what it does not: the shape of a manipulation trajectory and
its grasp timing transfer, because those are properties of the task. Contact forces,
the robot's own kinematics and anything requiring more than one gripper degree of
freedom do not, because a human hand has none of them in common with a parallel-jaw
gripper. Hand demos are worth the most as a pretraining or co-training corpus for
the vision-to-action mapping, evaluated on real rollouts — which the harness already
does, per-episode, with a significance test attached.
"""

from robobench.hand.detect import (  # noqa: F401
    Detection,
    HandDetector,
    MediaPipeDetector,
    ReplayDetector,
    SyntheticDetector,
)
from robobench.hand.filters import ConstantVelocity, OneEuroFilter  # noqa: F401
from robobench.hand.pose import HandPose, estimate_pose, grip_aperture, hand_scale, palm_frame  # noqa: F401
from robobench.hand.record import Episode, segment_episodes, write_demos, wrist_view  # noqa: F401
from robobench.hand.retarget import RetargetConfig, RetargetedEpisode, integrate, retarget  # noqa: F401
from robobench.hand.sources import camera_frames, probe_fps, video_frames  # noqa: F401
from robobench.hand.tracker import HandTracker, TrackFrame  # noqa: F401

__all__ = [
    "Detection", "HandDetector", "MediaPipeDetector", "ReplayDetector", "SyntheticDetector",
    "OneEuroFilter", "ConstantVelocity",
    "HandPose", "estimate_pose", "palm_frame", "hand_scale", "grip_aperture",
    "HandTracker", "TrackFrame",
    "RetargetConfig", "RetargetedEpisode", "retarget", "integrate",
    "Episode", "segment_episodes", "write_demos", "wrist_view",
    "video_frames", "camera_frames", "probe_fps",
]
