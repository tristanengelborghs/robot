"""Watch a policy (or flailing random actions) in the interactive viewer.

macOS note: the interactive viewer must run under mjpython, which ships with
the mujoco wheel:

    .venv/bin/mjpython -m dexhand.view --ckpt runs/<run>/ckpt_last.pt
    .venv/bin/mjpython -m dexhand.view            # random actions, toy config
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hand", default="shadow")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.ckpt:
        from dexhand.eval import load_ckpt
        from dexhand.train import make_env_cfg
        from dexhand.env import ReorientEnv
        policy, norm_a, cfg = load_ckpt(args.ckpt)
        env = ReorientEnv(cfg["env"]["hand"], make_env_cfg(cfg), seed=args.seed)
    else:
        from dexhand.env import ReorientEnv
        policy, norm_a = None, None
        env = ReorientEnv(args.hand, seed=args.seed)

    obs, _ = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        while v.is_running():
            if policy is None:
                action = rng.uniform(-0.3, 0.3, env.nu)
            else:
                a_t = torch.as_tensor(norm_a.normalize(obs["actor"][None]), dtype=torch.float32)
                with torch.no_grad():
                    a, _, _ = policy.act(a_t, a_t, deterministic=True)
                action = a.numpy()[0]
            obs, r, term, trunc, info = env.step(action)
            v.sync()
            if term or trunc:
                obs, _ = env.reset()
            time.sleep(env.cfg.ctrl_dt * 0.5)


if __name__ == "__main__":
    main()
