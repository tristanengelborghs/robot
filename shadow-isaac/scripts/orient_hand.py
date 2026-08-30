"""Measure which way the palm faces, and print the rotation that turns it up.

The stock Shadow Hand asset stands vertical with the fingers pointing up, so a
ball dropped onto it lands on fingertips and rolls off -- there is no basket.
The task needs the palm facing +z with the fingers roughly horizontal.

Rather than guess a quaternion and pay a simulator run per guess, this reads the
hand's geometry and computes the rotation directly:

* the palm plane is fitted to the knuckle row plus the palm body
* the finger direction is palm -> mean fingertip
* the palm normal is the plane normal, its sign resolved by the thumb, which
  sits on the palmar side of the hand

It then prints the delta rotation taking that normal to +z, composed with the
asset's current init rotation, ready to paste into CatchEnvCfg.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

KNUCKLES = ["robot0_ffknuckle", "robot0_mfknuckle", "robot0_rfknuckle", "robot0_lfmetacarpal"]
TIPS = ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal"]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_two_vectors(a, b):
    """wxyz rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -1.0 + 1e-9:                       # antiparallel: any perpendicular axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(a, b)
    s = np.sqrt((1.0 + d) * 2.0)
    return np.array([s * 0.5, *(axis / s)])


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240))
    hand = Articulation(SHADOW_HAND_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    for _ in range(10):                        # let the pose settle before reading
        sim.step()
        hand.update(1 / 240)

    names = list(hand.data.body_names)
    pos = hand.data.body_pos_w[0].cpu().numpy()

    def p(name):
        return pos[names.index(name)]

    palm = p("robot0_palm")
    knuckles = np.array([p(n) for n in KNUCKLES])
    tips = np.array([p(n) for n in TIPS])

    finger = tips.mean(axis=0) - palm
    finger /= np.linalg.norm(finger)
    across = knuckles[-1] - knuckles[0]
    across /= np.linalg.norm(across)

    normal = np.cross(across, finger)
    normal /= np.linalg.norm(normal)
    # The thumb sits on the palmar side, which fixes the otherwise free sign.
    if np.dot(p("robot0_thdistal") - palm, normal) < 0:
        normal = -normal

    print("\n=== measured, world frame ===")
    print(f"  finger direction (palm -> tips) : {np.round(finger, 3)}")
    print(f"  across the knuckles             : {np.round(across, 3)}")
    print(f"  palm normal (thumb side)        : {np.round(normal, 3)}")
    print(f"  fingertips above the palm       : {(tips[:, 2] - palm[2]).mean()*100:.1f} cm")

    delta = quat_from_two_vectors(normal, np.array([0.0, 0.0, 1.0]))
    current = np.array(SHADOW_HAND_CFG.init_state.rot, dtype=float)
    composed = quat_mul(delta, current)

    print("\n=== rotation to put the palm UP ===")
    print(f"  asset init rot (wxyz)  : {tuple(np.round(current, 6))}")
    print(f"  delta rot      (wxyz)  : {tuple(np.round(delta, 6))}")
    print(f"  USE THIS       (wxyz)  : {tuple(np.round(composed, 6))}")

    # Where the fingers end up: they should be roughly horizontal afterwards.
    def rotate(q, v):
        w, x, y, z = q
        u = np.array([x, y, z])
        return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)

    print(f"\n  after rotating: palm normal -> {np.round(rotate(delta, normal), 3)} "
          f"(want [0 0 1]), fingers -> {np.round(rotate(delta, finger), 3)} "
          f"(want z near 0)")


if __name__ == "__main__":
    main()
    simulation_app.close()
