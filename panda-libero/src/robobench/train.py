"""Training loop. Shared by every architecture so comparisons stay apples-to-apples.

    python -m robobench.train --config configs/resnet_film_bc.yaml
    python -m robobench.train --config configs/resnet_film_bc.yaml model.name=my_arch seed=1
    python -m robobench.train --config configs/resnet_film_bc.yaml ablation=no_lang
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from robobench import ablate
from robobench.config import config_hash, load_config
from robobench.data.libero_dataset import LiberoChunkDataset, collate
from robobench.models import *  # noqa: F401,F403  (populates the registry)
from robobench.models.base import ObsSpec
from robobench.registry import build_model, list_models
from robobench.utils import (
    JsonlWriter,
    count_params,
    git_provenance,
    human,
    pick_device,
    save_json,
    set_seed,
)


class EMA:
    """Exponential moving average of weights. Standard in visuomotor BC and worth a
    few points; applied identically to every model so it never favours one."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(p)


def build_lr_schedule(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.02):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def load_language_cache(path: str | None, num_tasks_hint: int = 0):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"language cache not found at {p}. Build it once with:\n"
            "  python scripts/cache_language.py --data-root data/libero --suites libero_object libero_spatial"
        )
    blob = np.load(p, allow_pickle=True)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in zip(blob["keys"], blob["vectors"])}


def move_to(batch, device):
    """Host -> device.

    `non_blocking=True` is only safe when the source is pinned, which the loader
    arranges for CUDA alone. On MPS an async copy from pageable memory returns
    before it has finished and the source buffer is recycled underneath it, so the
    tensors arrive as garbage: masks full of 1e-23, an infinite language norm, and
    a loss that is silently meaningless rather than an error.
    """
    nb = device.type == "cuda"
    out = dict(batch)
    out["images"] = {k: v.to(device, non_blocking=nb) for k, v in batch["images"].items()}
    for k in ("proprio", "lang", "actions", "mask", "task_id"):
        out[k] = batch[k].to(device, non_blocking=nb)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a visuomotor policy on LIBERO.")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.batch_size=64")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    chash = config_hash(cfg)
    run_name = cfg.get("run_name") or f"{cfg['model']['name']}-{cfg['data']['suites'][0]}-s{cfg['seed']}-{chash}"
    run_dir = Path(cfg["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"], deterministic=cfg["train"]["deterministic"])
    device = pick_device(cfg["device"])

    # ---- data -------------------------------------------------------------
    lang_cache = load_language_cache(cfg["data"]["lang_cache"])
    dataset = LiberoChunkDataset(
        root=cfg["data"]["root"],
        suites=cfg["data"]["suites"],
        chunk_size=cfg["policy"]["chunk_size"],
        image_size=cfg["data"]["image_size"],
        cameras=cfg["data"]["cameras"],
        max_demos_per_task=cfg["data"]["max_demos_per_task"],
        lang_cache=lang_cache,
        lang_dim=cfg["data"]["lang_dim"],
        flip_images=cfg["data"]["flip_images"],
        augment=cfg["data"]["augment"],
        crop_ratio=cfg["data"]["crop_ratio"],
    )
    meta = dataset.describe()
    print(f"[data] {meta['num_tasks']} tasks | {human(meta['num_samples'])} samples | {cfg['data']['suites']}")

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )

    # ---- model ------------------------------------------------------------
    obs_spec = ObsSpec(
        cameras=list(cfg["data"]["cameras"]),
        image_size=cfg["data"]["image_size"],
        proprio_dim=meta["proprio_dim"],
        lang_dim=meta["lang_dim"],
    )
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(
        cfg["model"]["name"],
        obs_spec=obs_spec,
        action_dim=meta["action_dim"],
        chunk_size=cfg["policy"]["chunk_size"],
        **model_kwargs,
    ).to(device)

    params = count_params(model)
    print(f"[model] {cfg['model']['name']} | {human(params['total'])} params | device={device}")
    print(f"[model] registered: {list_models()}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.95),
    )
    total_steps = cfg["train"]["steps"]
    sched = build_lr_schedule(opt, cfg["train"]["warmup_steps"], total_steps)
    ema = EMA(model, cfg["train"]["ema_decay"]) if cfg["train"]["ema_decay"] else None

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_json(
        {
            "config": cfg,
            "config_hash": chash,
            "provenance": git_provenance(),
            "params": params,
            "dataset": {k: v for k, v in meta.items() if k != "instructions"},
            "instructions": meta["instructions"],
        },
        run_dir / "run.json",
    )

    log = JsonlWriter(run_dir / "train_log.jsonl")
    ablation = cfg["ablation"]
    if ablation != "none":
        print(f"[ablation] '{ablation}' is active — this input is destroyed in training AND eval")

    # ---- loop -------------------------------------------------------------
    step, t0, running = 0, time.time(), []
    pbar = tqdm(total=total_steps, desc=run_name, dynamic_ncols=True)
    model.train()

    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            batch = ablate.apply(move_to(batch, device), ablation)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model.loss(batch)
            else:
                out = model.loss(batch)
            loss = out["loss"]

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)

            running.append(loss.item())
            step += 1
            pbar.update(1)

            if step % cfg["train"]["log_every"] == 0:
                mean_loss = float(np.mean(running[-cfg["train"]["log_every"] :]))
                rec = {
                    "step": step,
                    "loss": mean_loss,
                    "lr": sched.get_last_lr()[0],
                    "elapsed_s": round(time.time() - t0, 1),
                    **{k: float(v) for k, v in out.items() if k != "loss"},
                }
                log.write(rec)
                pbar.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{rec['lr']:.2e}")

            if step % cfg["train"]["ckpt_every"] == 0 or step == total_steps:
                _save(run_dir / f"ckpt_{step:06d}.pt", model, ema, dataset, cfg, obs_spec, meta, step)
                _save(run_dir / "ckpt_last.pt", model, ema, dataset, cfg, obs_spec, meta, step)

    pbar.close()
    log.close()
    print(f"\n[done] {step} steps in {(time.time() - t0) / 60:.1f} min -> {run_dir}")
    print(f"[next] python -m robobench.eval --ckpt {run_dir / 'ckpt_last.pt'}")


def _save(path, model, ema, dataset, cfg, obs_spec, meta, step) -> None:
    torch.save(
        {
            "step": step,
            "config": cfg,
            "model_name": cfg["model"]["name"],
            "model_kwargs": {k: v for k, v in cfg["model"].items() if k != "name"},
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.shadow.state_dict() if ema else None,
            "normalizer": dataset.normalizer.state_dict(),
            "obs_spec": obs_spec.__dict__,
            "action_dim": meta["action_dim"],
            "chunk_size": cfg["policy"]["chunk_size"],
            "instructions": meta["instructions"],
            "provenance": git_provenance(),
        },
        path,
    )


if __name__ == "__main__":
    main()
