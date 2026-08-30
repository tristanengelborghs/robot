"""Print the running hand's raw world geometry. No derived quantities.

Every orientation check so far has been a computed "palm normal", and at least
one of them was confidently wrong: it was built from `lfmetacarpal - ffknuckle`
crossed with `knuckles_mean - palm`, and neither vector is guaranteed to lie in
the palm plane -- lfmetacarpal is the proximal metacarpal rather than a knuckle,
and the palm body origin need not sit on the palm surface. A search then
optimised that wrong vector faithfully.

So this prints positions and lets a person read them. With the palm up:

  * the knuckle row sits at roughly the SAME height as the palm
  * the curled fingertips sit ABOVE both
  * the spread across the knuckles is horizontal

With the hand on its side, the knuckle row spreads vertically instead.

A plane is also fitted to palm + knuckles by SVD, which is a far better normal
than a cross product of two arbitrary vectors, and its sign is anchored on the
finger curl.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Catch-Shadow-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import catching  # noqa: E402, F401

KNUCKLES = ["robot0_ffknuckle", "robot0_mfknuckle", "robot0_rfknuckle", "robot0_lfknuckle"]
TIPS = ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal",
        "robot0_lfdistal", "robot0_thdistal"]
OTHER = ["robot0_palm", "robot0_wrist", "robot0_forearm", "robot0_thbase"]


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    env.reset()
    for _ in range(30):                       # let the reset pose take effect
        env.step(u._reset_qpos.new_zeros(1, u.cfg.action_space))

    names = list(u.hand.data.body_names)
    bp = u.hand.data.body_pos_w[0].cpu().numpy()
    origin = u.scene.env_origins[0].cpu().numpy()
    palm = bp[names.index("robot0_palm")]

    print("\n=== world positions, relative to the palm (cm) ===")
    print(f"{'body':<24} {'x':>8} {'y':>8} {'z':>8}")
    for n in OTHER + KNUCKLES + TIPS:
        d = (bp[names.index(n)] - palm) * 100
        print(f"{n:<24} {d[0]:8.1f} {d[1]:8.1f} {d[2]:8.1f}")
    print(f"\npalm world position: {np.round(palm - origin, 3)} (env-relative)")
    ball = u.ball.data.root_pos_w[0].cpu().numpy()
    print(f"ball  world position: {np.round(ball - palm, 3)} (palm-relative)")

    knuck = np.array([bp[names.index(n)] for n in KNUCKLES])
    tips = np.array([bp[names.index(n)] for n in TIPS[:4]])

    print("\n=== how to read it ===")
    print(f"  knuckle spread in z : {(knuck[:,2].max()-knuck[:,2].min())*100:6.1f} cm "
          f"(small => the knuckle row is horizontal)")
    print(f"  knuckles vs palm, z : {(knuck[:,2].mean()-palm[2])*100:6.1f} cm "
          f"(near 0 => the palm plane is horizontal)")
    print(f"  fingertips vs palm z: {(tips[:,2].mean()-palm[2])*100:6.1f} cm "
          f"(positive => the fingers curl UP, a basket)")

    pts = np.vstack([knuck, palm[None, :]])
    centred = pts - pts.mean(axis=0)
    normal = np.linalg.svd(centred)[2][-1]
    if np.dot(tips.mean(axis=0) - knuck.mean(axis=0), normal) < 0:
        normal = -normal
    tilt = np.degrees(np.arccos(np.clip(normal[2], -1, 1)))
    print(f"\n  SVD plane normal    : {np.round(normal, 3)}  ({tilt:.1f} deg off vertical)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
