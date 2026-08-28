"""Fingertip tactile signal, extracted from the contact solver.

Real tactile-rich hands report distributed pressure; the tractable simulated
analogue is the net contact force each fingertip is currently transmitting.
That scalar-per-finger signal is enough to answer the question that matters
in-hand: which fingers are load-bearing right now — which is exactly the
information a proprioception-only student is missing when the cube slips.

Extraction walks mjData.contact and calls mj_contactForce per active contact,
attributing force to a fingertip when either geom of the pair belongs to that
fingertip's body. Two things are deliberate:

Only contacts WITH THE CUBE count by default. A fingertip pressed against a
neighbouring finger produces real force and zero task information; folding it
in teaches the student that self-collision feels like grasping.

Forces are reported in log scale, log1p(|f|). Contact forces span three orders
of magnitude between a grazing touch and a full-weight press, and a network fed
raw Newtons spends its input range on the rare hard presses. log1p keeps zero
at zero and compresses the tail.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import mujoco
import numpy as np


def fingertip_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fingertip_geom_ids: Sequence[Tuple[int, ...]],
    against_geom: int | None = None,
    log_scale: bool = True,
) -> np.ndarray:
    """(n_fingertips,) net contact force magnitude per fingertip.

    against_geom: if given, only contacts against that geom (the cube) count.
    """
    geom_to_tip = {}
    for tip_idx, geoms in enumerate(fingertip_geom_ids):
        for g in geoms:
            geom_to_tip[g] = tip_idx

    out = np.zeros(len(fingertip_geom_ids), dtype=np.float64)
    force = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        tip = geom_to_tip.get(g1)
        other = g2
        if tip is None:
            tip = geom_to_tip.get(g2)
            other = g1
        if tip is None:
            continue
        if against_geom is not None and other != against_geom:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        # force[:3] is in the contact frame; the magnitude is frame-free
        out[tip] += float(np.linalg.norm(force[:3]))

    return np.log1p(out) if log_scale else out
