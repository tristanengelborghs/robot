"""Print the Shadow Hand's real body and joint names, then exit.

The contact sensor and the palm lookup in `catch_env_cfg.py` are regexes over
body names, and a regex that matches nothing does not raise -- it reports no
contacts, and the failure surfaces weeks later as a policy that can never
catch. This is the cheapest possible way to replace those guesses with facts.

Run it with `make bodies`.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240))
    hand = Articulation(SHADOW_HAND_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    print("\n=== bodies ===")
    for i, name in enumerate(hand.data.body_names):
        print(f"{i:3d}  {name}")
    print("\n=== joints ===")
    for i, name in enumerate(hand.data.joint_names):
        print(f"{i:3d}  {name}")
    print(f"\nbodies={len(hand.data.body_names)}  joints={len(hand.data.joint_names)}")
    print("\nPut the fingertip pattern and num_hand_dofs from this list into "
          "catch_env_cfg.py.")


if __name__ == "__main__":
    main()
    simulation_app.close()
