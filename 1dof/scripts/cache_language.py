#!/usr/bin/env python3
"""Precompute frozen CLIP text embeddings for every instruction, once.

Instructions are fixed per task, so running a text encoder inside the training loop
buys nothing and costs throughput. This caches them to an .npz keyed by the exact
instruction string — keyed by string rather than by index so that training order and
evaluation order can never silently disagree.

    python scripts/cache_language.py --data-root data/libero --suites libero_object libero_spatial
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def instructions_from_hdf5(root: Path, suites) -> list:
    found = []
    for suite in suites:
        d = root / suite
        if not d.is_dir():
            raise SystemExit(f"suite directory not found: {d}")
        for p in sorted(d.glob("*.hdf5")):
            with h5py.File(p, "r") as f:
                try:
                    text = json.loads(f["data"].attrs["problem_info"])["language_instruction"].strip()
                except Exception:
                    text = p.stem.replace("_demo", "").replace("_", " ").strip()
            found.append(text)
    return found


def instructions_from_benchmark(suites) -> list:
    """Also pull from the LIBERO task registry, so evaluation-time strings are covered
    even for tasks whose demo files you did not download."""
    try:
        from libero.libero import benchmark
    except ImportError:
        return []
    out = []
    for suite in suites:
        try:
            ts = benchmark.get_benchmark_dict()[suite]()
        except KeyError:
            continue
        out += [ts.get_task(i).language.strip() for i in range(ts.n_tasks)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/libero")
    ap.add_argument("--suites", nargs="+", default=["libero_object"])
    ap.add_argument("--out", default="cache/lang_clip.npz")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    args = ap.parse_args()

    texts = []
    root = Path(args.data_root)
    if root.is_dir():
        texts += instructions_from_hdf5(root, args.suites)
    texts += instructions_from_benchmark(args.suites)

    texts = sorted(set(t for t in texts if t))
    if not texts:
        raise SystemExit("found no instructions — check --data-root and --suites")
    print(f"embedding {len(texts)} unique instructions with {args.model}")

    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    tok = CLIPTokenizerFast.from_pretrained(args.model)
    enc = CLIPTextModelWithProjection.from_pretrained(args.model).eval()

    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        vecs = enc(**batch).text_embeds
        vecs = torch.nn.functional.normalize(vecs, dim=-1).cpu().numpy().astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, keys=np.array(texts, dtype=object), vectors=vecs)
    print(f"wrote {out}  ({vecs.shape[0]} x {vecs.shape[1]})")
    for t in texts[:5]:
        print(f"  - {t}")


if __name__ == "__main__":
    main()
