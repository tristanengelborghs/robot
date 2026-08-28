"""The 21-point hand skeleton, and the rigid subset everything else leans on.

Index order is MediaPipe's, because every practical hand detector either emits that
order or documents its mapping onto it. Nothing below this module assumes MediaPipe
specifically — `robobench.hand.detect.HandDetector` is the seam.

The five palm points (wrist + four MCP knuckles) are the only landmarks that stay
approximately rigid with respect to each other. Fingers articulate, so anything
that needs a *stable* quantity — the hand's orientation, its metric scale, the
normalization that makes one person's hand comparable to another's — is computed
from the palm alone and never from fingertips. Fingertips are used for exactly one
thing: the grip signal, which is supposed to move.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

PALM_IDS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
MCP_IDS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
TIP_IDS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

FINGERS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# Drawing topology, for the overlay video only.
BONES = tuple(
    [(WRIST, INDEX_MCP), (WRIST, THUMB_CMC), (WRIST, PINKY_MCP),
     (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP)]
    + [(chain[i], chain[i + 1]) for chain in FINGERS.values() for i in range(len(chain) - 1)]
)

# A nominal adult RIGHT palm, in metres, expressed in the canonical hand frame:
#   +x  wrist -> middle knuckle  (the pointing direction)
#   +z  palm normal, out through the BACK of the hand
#   +y  z cross x, which for a right hand points to the thumb side
# The numbers are anthropometric averages, not a calibration: `palm_frame` only
# uses this template up to scale and rotation, and `hand_scale` divides the scale
# back out. Its job is to fix the frame convention and the correspondence order,
# so a left hand and a right hand doing the same motion retarget the same way.
CANONICAL_PALM_RIGHT = np.array(
    [
        [0.000, 0.000, 0.0],  # wrist
        [0.090, 0.022, 0.0],  # index MCP
        [0.095, 0.000, 0.0],  # middle MCP
        [0.090, -0.020, 0.0],  # ring MCP
        [0.083, -0.041, 0.0],  # pinky MCP
    ],
    dtype=np.float64,
)

#: Median wrist->MCP distance of the canonical palm. The unit of "one hand".
#: Median, matching `pose.hand_scale`, so that a perfect observation of this exact
#: template has a Kabsch residual of zero rather than a constant 1% floor.
CANONICAL_SCALE = float(np.median(np.linalg.norm(CANONICAL_PALM_RIGHT[1:], axis=-1)))


def canonical_palm(handedness: str = "right") -> np.ndarray:
    """Canonical palm template, mirrored for a left hand.

    Mirroring in y (rather than letting Kabsch's determinant correction sort it
    out) matters: a reflection is not a rotation, so fitting a right-hand template
    to a left hand yields the nearest *proper* rotation to a mirrored point set,
    which is off by a large and motion-dependent amount. Left-hand demos retargeted
    that way look plausible frame by frame and are wrong throughout.
    """
    h = handedness.lower()
    if h not in ("left", "right"):
        raise ValueError(f"handedness must be 'left' or 'right', got {handedness!r}")
    palm = CANONICAL_PALM_RIGHT.copy()
    if h == "left":
        palm[:, 1] *= -1.0
    return palm
