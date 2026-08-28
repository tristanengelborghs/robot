"""Teacher -> student distillation by DAgger.

    python -m dexhand.distill --ckpt runs/<run>/ckpt_last.pt

The teacher acts from privileged state (exact cube pose). The student sees only
what plausible hardware provides: noisy joint positions, its own previous
action, fingertip tactile, and the goal — no cube pose at all. It closes the
gap with a short observation history, which is what turns "where is the cube"
from an input into something inferable from how the fingers moved and what
they are touching.

DAgger rather than behaviour cloning on teacher rollouts, for the standard
reason made concrete: a student trained only on the teacher's states has never
seen the states its own early mistakes produce, and in-hand manipulation
compounds mistakes within a few control steps (the cube slides, the tactile
signature changes, and the BC student is off-distribution while still holding
the cube). Rolling out the STUDENT and labelling with the teacher trains
exactly on the distribution the student will actually visit.

This is the recipe's scaffold on proprio+tactile. A hardware-deployable student
of the full task wants vision as well; the seam is the obs dict's "student"
key — widen it, and this file does not change.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from dexhand.env import ReorientEnv
from dexhand.eval import load_ckpt
from dexhand.networks import StudentPolicy
from dexhand.ppo import RunningNorm
from dexhand.train import make_env_cfg


def stack_history(buf: deque) -> np.ndarray:
    return np.concatenate(list(buf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--steps-per-iter", type=int, default=4000)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    teacher, norm_a, cfg = load_ckpt(args.ckpt)
    env = ReorientEnv(cfg["env"]["hand"], make_env_cfg(cfg), seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    s_dim = obs["student"].shape[0] * args.history
    act_dim = env.action_space.shape[0]
    student = StudentPolicy(s_dim, act_dim, hidden=(256, 256))
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    norm_s = RunningNorm((s_dim,))

    out = Path(args.ckpt).parent
    log = open(out / "distill_log.jsonl", "a", buffering=1)
    dataset_x, dataset_y = [], []

    for it in range(1, args.iters + 1):
        hist = deque([obs["student"]] * args.history, maxlen=args.history)
        ep_successes, episodes = [], 0

        for _ in range(args.steps_per_iter):
            s_obs = stack_history(hist)
            norm_s.update(s_obs[None])

            # label: what the teacher would do HERE (mean action)
            a_t = torch.as_tensor(norm_a.normalize(obs["actor"][None]), dtype=torch.float32)
            with torch.no_grad():
                teacher_a, _, _ = teacher.act(a_t, a_t, deterministic=True)
            # raw, not normalized: the stats keep moving during collection, and
            # rows normalized under last iteration's stats do not match what
            # the deployed student (saved with the FINAL stats) will see
            dataset_x.append(s_obs.copy())
            dataset_y.append(teacher_a.numpy()[0])

            # act: the student drives (after a warmup iteration on teacher control,
            # before the student has learned anything at all)
            if it == 1:
                action = teacher_a.numpy()[0]
            else:
                with torch.no_grad():
                    action = student(torch.as_tensor(
                        norm_s.normalize(s_obs)[None], dtype=torch.float32)).numpy()[0]

            obs, r, term, trunc, info = env.step(action)
            hist.append(obs["student"])
            if term or trunc:
                ep_successes.append(info["successes"])
                episodes += 1
                obs, _ = env.reset()
                hist = deque([obs["student"]] * args.history, maxlen=args.history)

        x = torch.as_tensor(norm_s.normalize(np.array(dataset_x)), dtype=torch.float32)
        y = torch.as_tensor(np.array(dataset_y), dtype=torch.float32)
        for _ in range(args.epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), 256):
                idx = perm[i:i + 256]
                loss = torch.nn.functional.mse_loss(student(x[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        row = {"iter": it, "dataset": len(x), "mse": float(loss.item()),
               "student_successes_per_ep": float(np.mean(ep_successes)) if ep_successes else None,
               "episodes": episodes}
        log.write(json.dumps(row) + "\n")
        print(f"  distill it {it}/{args.iters}  dataset {len(x):6d}  mse {row['mse']:.4f}  "
              f"student succ/ep {row['student_successes_per_ep']}")

    torch.save({"student": student.state_dict(), "norm_student": norm_s.state_dict(),
                "history": args.history, "config": cfg}, out / "student_last.pt")
    print(f"[done] -> {out / 'student_last.pt'}")


if __name__ == "__main__":
    main()
