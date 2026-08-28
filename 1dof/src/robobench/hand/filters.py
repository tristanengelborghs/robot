"""Temporal filtering for landmark streams.

A per-frame hand detector has no memory, so its output jitters by a few millimetres
even on a perfectly still hand. That jitter is the single most damaging thing you
can hand to behaviour cloning: the retargeted action is a *difference* of
consecutive poses, so detector noise of amplitude e becomes action noise of
amplitude ~2e while the real signal — the hand's actual motion between frames — is
often smaller than that. Filter first, differentiate second, or the policy spends
its capacity learning to reproduce noise.

The filter here is the 1-Euro filter (Casiello, Roussel & Lecolinet, CHI 2012): a
low-pass whose cutoff rises with the observed speed. A fixed low-pass has to choose
between jitter at rest and lag during fast motion; this one gets both, which for
retargeting means a still hand yields exactly-zero actions and a fast reach is not
smeared across frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["LowPass", "OneEuroFilter", "ConstantVelocity"]


class LowPass:
    """Exponential low-pass with a per-call smoothing factor."""

    def __init__(self) -> None:
        self.value: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray, alpha: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self.value = x.copy() if self.value is None else alpha * x + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


def _alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor of a first-order low-pass at `cutoff` Hz sampled every `dt` s."""
    tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
    return float(dt / (dt + tau))


@dataclass
class OneEuroFilter:
    """Speed-adaptive low-pass, applied elementwise to an array of any shape.

    min_cutoff  cutoff in Hz at zero speed. Lower = steadier hand at rest, more lag.
    beta        how fast the cutoff opens up with speed. Higher = less lag on fast
                motion, more jitter passed through. This is the knob you actually
                tune; 0 degrades to a plain low-pass.
    d_cutoff    cutoff of the low-pass applied to the speed estimate itself. Rarely
                worth touching — it only stops the adaptive term from chattering.

    Defaults are tuned for 21 hand landmarks in metres at 30 fps: visually still at
    rest, under ~1 frame of lag on a reach. Timestamps are in seconds and may be
    irregular, which matters because dropped frames are normal in a webcam stream
    and a filter that assumes a fixed rate quietly changes its own bandwidth when
    the rate wobbles.
    """

    min_cutoff: float = 1.0
    beta: float = 0.7
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        self._x = LowPass()
        self._dx = LowPass()
        self._prev: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._prev = None
        self._t = None

    @property
    def initialized(self) -> bool:
        return self._prev is not None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._prev is None:
            self._prev, self._t = x.copy(), float(t)
            self._dx(np.zeros_like(x), 1.0)
            return self._x(x, 1.0).copy()

        dt = float(t) - float(self._t)
        if dt <= 0.0:
            # Duplicate or out-of-order timestamp: hold, rather than divide by zero
            # and blow the derivative up to infinity.
            return self._x.value.copy()

        speed = (x - self._prev) / dt
        speed_hat = self._dx(speed, _alpha(self.d_cutoff, dt))

        # One cutoff per element: a still finger stays still even while the wrist
        # sweeps. A single scalar cutoff for the whole hand would drag the steady
        # landmarks along with the fast ones.
        cutoff = self.min_cutoff + self.beta * np.abs(speed_hat)
        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
        alpha = dt / (dt + tau)

        prev = self._x.value
        out = alpha * x + (1.0 - alpha) * prev
        self._x.value = out
        self._prev, self._t = x.copy(), float(t)
        return out.copy()


@dataclass
class ConstantVelocity:
    """Damped constant-velocity extrapolation, used to coast through occlusions.

    When a hand passes behind an object the detector returns nothing for a few
    frames. Two bad options: drop those frames (the timeline develops holes and the
    action deltas across a hole are huge) or freeze the pose (the hand teleports on
    re-acquisition). Coasting on the last velocity is the third: it is right for the
    first few frames, and `damping` bleeds the velocity away so a long occlusion
    decays to a stationary guess instead of flying off screen.

    Coasted frames are always flagged downstream — they are an interpolation, not an
    observation, and long runs of them should be dropped from a demo rather than
    trained on.
    """

    damping: float = 0.85
    max_speed: float = 3.0  # m/s; above this the "velocity" is a detector glitch

    def __post_init__(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._t: Optional[float] = None
        self._coasts = 0

    def update(self, x: np.ndarray, t: float, smooth: float = 0.5) -> None:
        x = np.asarray(x, dtype=np.float64)
        if self._x is not None and self._t is not None and t > self._t:
            v = (x - self._x) / (t - self._t)
            speed = np.linalg.norm(v.reshape(-1, v.shape[-1]), axis=-1).max() if v.size else 0.0
            if speed > self.max_speed:
                v = v * (self.max_speed / speed)
            self._v = v if self._v is None else smooth * v + (1.0 - smooth) * self._v
        self._x, self._t, self._coasts = x.copy(), float(t), 0

    def predict(self, t: float) -> Optional[np.ndarray]:
        if self._x is None or self._t is None:
            return None
        if self._v is None:
            return self._x.copy()
        dt = float(t) - float(self._t)
        if dt <= 0.0:
            return self._x.copy()
        return self._x + self._v * dt * (self.damping ** max(self._coasts, 0))

    def coast(self, t: float) -> Optional[np.ndarray]:
        """Advance one step without an observation. Returns the extrapolated state."""
        pred = self.predict(t)
        if pred is None:
            return None
        self._coasts += 1
        self._x, self._t = pred, float(t)
        return pred.copy()
