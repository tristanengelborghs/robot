"""Search orientations for the one that puts the palm up, in one session.

Composing quaternions against this asset kept producing a hand that was not
where the algebra said it would be -- twice -- so this stops reasoning and
measures. It writes each candidate root orientation into the running simulator,
steps, and reads back where the palm actually points. One Isaac start-up
answers what a run-per-guess loop was answering one bit at a time.

The candidates are the 24 rotations of the cube (signed permutation matrices
with determinant +1), which is every axis-aligned way to stand a hand up. The
winner is the value to paste into CatchEnvCfg as the hand's rot.

Palm direction is signed by the CURL, not the thumb: with the reset pose cupped
the fingertips necessarily lie on the palmar side of the knuckle row. That is a
property of the pose, where "the thumb is on the palmar side" was an assumption
about anatomy -- and it was the wrong one.
"""

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from catching.tasks.direct.catch import grasp as gp  # noqa: E402

KNUCKLES = ["robot0_ffknuckle", "robot0_mfknuckle", "robot0_rfknuckle", "robot0_lfmetacarpal"]
TIPS = ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal"]


def cube_rotations():
    """The 24 signed permutation matrices with det +1, as wxyz quaternions."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            for i, j in enumerate(perm):
                m[i, j] = signs[i]
            if abs(np.linalg.det(m) - 1.0) > 1e-6:
                continue
            tr = m.trace()
            if tr > 0:
                s = np.sqrt(tr + 1.0) * 2
                q = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
                     (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
            elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
                q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                     (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
            elif m[1, 1] > m[2, 2]:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
                q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                     0.25 * s, (m[1, 2] + m[2, 1]) / s]
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
                q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s]
            out.append(np.round(np.array(q), 6))
    return out


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240))
    hand = Articulation(SHADOW_HAND_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    names = list(hand.data.joint_names)
    lim = hand.data.soft_joint_pos_limits[0].cpu().numpy()
    qpos = torch.as_tensor(gp.pose_to_qpos(gp.OPEN_POSE, names, lim[:, 0], lim[:, 1]),
                           device=hand.device).unsqueeze(0)

    body_names = list(hand.data.body_names)
    ki = [body_names.index(n) for n in KNUCKLES]
    ti = [body_names.index(n) for n in TIPS]
    pi = body_names.index("robot0_palm")

    results = []
    for q in cube_rotations():
        root = torch.zeros(1, 7, device=hand.device)
        root[0, :3] = torch.tensor([0.0, 0.0, 0.5], device=hand.device)
        root[0, 3:] = torch.as_tensor(q, dtype=torch.float32, device=hand.device)
        hand.write_root_pose_to_sim(root)
        hand.write_joint_state_to_sim(qpos, torch.zeros_like(qpos))
        for _ in range(4):
            sim.step()
            hand.update(1 / 240)

        bp = hand.data.body_pos_w[0].cpu().numpy()
        palm, knuck, tips = bp[pi], bp[ki], bp[ti]
        across = knuck[-1] - knuck[0]
        across /= np.linalg.norm(across)
        along = knuck.mean(axis=0) - palm
        along /= np.linalg.norm(along)
        n = np.cross(across, along)
        n /= np.linalg.norm(n)
        if np.dot(tips.mean(axis=0) - knuck.mean(axis=0), n) < 0:
            n = -n
        results.append((float(n[2]), q, n, along))

    results.sort(key=lambda r: -r[0])
    print("\n=== orientations, best palm-up first ===")
    print(f"{'rot (wxyz)':<40} {'palm normal':<22} {'fingers':<22} tilt")
    for up, q, n, along in results[:6]:
        tilt = np.degrees(np.arccos(np.clip(up, -1, 1)))
        print(f"{str(tuple(q)):<40} {str(np.round(n,3)):<22} "
              f"{str(np.round(along,3)):<22} {tilt:5.1f} deg")
    best = results[0]
    print(f"\nUSE: hand_rot = {tuple(float(v) for v in best[1])}")
    print(f"     palm normal {np.round(best[2],3)}, fingers {np.round(best[3],3)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
