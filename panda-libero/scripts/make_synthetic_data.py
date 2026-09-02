#!/usr/bin/env python3
"""Write LIBERO-shaped HDF5 files with random content.

Lets you develop and debug an architecture — shapes, loss, throughput, the whole
train loop — on a laptop with no simulator, no dataset download and no GPU. The
files match the real layout exactly, so code that runs here runs on real LIBERO.

    python scripts/make_synthetic_data.py
    python -m robobench.train --config configs/smoke.yaml
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

INSTRUCTIONS = [
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/libero_object")
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--demos", type=int, default=4)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="cache/lang_synthetic.npz",
                    help="where the matching language cache goes; the smoke config reads the default")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = args.image_size

    for t in range(args.tasks):
        instruction = INSTRUCTIONS[t % len(INSTRUCTIONS)]
        path = out / (instruction.replace(" ", "_") + "_demo.hdf5")
        with h5py.File(path, "w") as f:
            data = f.create_group("data")
            data.attrs["problem_info"] = json.dumps(
                {"language_instruction": instruction, "problem_name": "libero_object"}
            )
            data.attrs["total"] = args.demos * args.length
            for d in range(args.demos):
                g = data.create_group(f"demo_{d}")
                T = args.length

                # The trajectory phase is drawn into the image as a moving bright
                # square, and the action is a deterministic function of that phase.
                # So the loss can only fall if gradient actually flows image ->
                # action: a flat curve here means the vision path is broken, which
                # is exactly what a smoke test should be able to tell you.
                p_t = np.linspace(0.0, 1.0, T, dtype=np.float32)
                agent = np.zeros((T, s, s, 3), dtype=np.uint8)
                wrist = np.zeros((T, s, s, 3), dtype=np.uint8)
                box = max(4, s // 8)
                for i, pv in enumerate(p_t):
                    r = int(pv * (s - box))
                    c = int(((t + 1) * pv) % 1.0 * (s - box))
                    agent[i, r : r + box, c : c + box] = 255
                    wrist[i, c : c + box, r : r + box, (t % 3)] = 255
                agent = np.clip(agent.astype(np.int16) + rng.integers(0, 24, agent.shape), 0, 255).astype(np.uint8)
                wrist = np.clip(wrist.astype(np.int16) + rng.integers(0, 24, wrist.shape), 0, 255).astype(np.uint8)

                phase = 2 * np.pi * p_t
                acts = np.stack(
                    [np.sin(phase + t), np.cos(phase), np.sin(2 * phase),
                     0.5 * p_t, 0.2 * np.ones(T), -0.3 * np.ones(T),
                     np.where(p_t > 0.5, 1.0, -1.0)], axis=-1
                ).astype(np.float32)
                acts += rng.normal(0, 0.01, acts.shape).astype(np.float32)

                g.create_dataset("actions", data=np.clip(acts, -1, 1))
                g.create_dataset("dones", data=np.zeros(T, dtype=np.int64))
                g.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))
                obs = g.create_group("obs")
                obs.create_dataset("agentview_rgb", data=agent)
                obs.create_dataset("eye_in_hand_rgb", data=wrist)
                joints = np.stack([np.sin(phase + j) for j in range(7)], axis=-1).astype(np.float32)
                obs.create_dataset("joint_states", data=joints)
                obs.create_dataset("gripper_states", data=np.stack([p_t, -p_t], axis=-1).astype(np.float32))
        print(f"wrote {path}")

    # Matching language cache so the smoke config runs without transformers.
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    keys = INSTRUCTIONS[: args.tasks]
    vecs = rng.normal(0, 1, (len(keys), 512)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
    np.savez(cache, keys=np.array(keys, dtype=object), vectors=vecs)
    print(f"wrote {cache}")


if __name__ == "__main__":
    main()
