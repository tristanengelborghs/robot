"""Rollout evaluation on LIBERO, with per-episode outcomes written to disk.

    python -m robobench.eval --ckpt runs/<run>/ckpt_last.pt --suite libero_object

Every episode gets a row in results/<run>/<suite>_episodes.jsonl: task, episode index,
init-state index, seed, success, steps. Aggregate success rate is derived from that
file, never reported on its own. This is the single habit that lets anyone — including
a reviewer — run a significance test on your claim, and it cannot be retrofitted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from robobench import ablate
from robobench.data.normalizer import ActionNormalizer
from robobench.models import *  # noqa: F401,F403
from robobench.models.base import ObsSpec
from robobench.registry import build_model
from robobench.stats import bootstrap_ci
from robobench.utils import JsonlWriter, git_provenance, pick_device, save_json, set_seed

# LIBERO convention: 220 steps for the short suites, 520 for the long-horizon one.
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 220,
    "libero_goal": 220,
    "libero_10": 520,
    "libero_90": 220,
}


def _allow_libero_init_state_unpickling() -> None:
    """Let torch >= 2.6 read LIBERO's init-state files.

    LIBERO calls `torch.load(path)` with no `weights_only`, and torch 2.6 flipped
    that default to True, so loading the pickled numpy arrays now raises
    UnpicklingError. Rather than pinning torch backwards for the whole project,
    allowlist exactly the numpy reconstructors those files contain — they come
    from the LIBERO commit recorded in .libero_commit, not from anywhere untrusted.
    """
    try:
        import numpy as np
        from torch.serialization import add_safe_globals
    except ImportError:  # pragma: no cover - torch too old to need this
        return

    allowed = [np.ndarray, np.dtype]
    for path in ("numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct"):
        mod_name, _, attr = path.rpartition(".")
        try:
            mod = __import__(mod_name, fromlist=[attr])
            allowed.append(getattr(mod, attr))
        except (ImportError, AttributeError):
            continue
    # Concrete dtype classes (numpy >= 1.25) appear in the pickle as well.
    allowed += [getattr(np.dtypes, n) for n in dir(getattr(np, "dtypes", ())) if n.endswith("DType")]
    add_safe_globals(allowed)


def _import_libero():
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:  # pragma: no cover - depends on optional sim stack
        raise SystemExit(
            "LIBERO is not installed. Training does not need it, evaluation does.\n"
            "  bash scripts/setup_libero.sh\n"
            f"(original error: {e})"
        )
    _allow_libero_init_state_unpickling()
    return benchmark, get_libero_path, OffScreenRenderEnv


def load_policy(ckpt_path: str, device: torch.device, use_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_spec = ObsSpec(**ckpt["obs_spec"])
    model = build_model(
        ckpt["model_name"],
        obs_spec=obs_spec,
        action_dim=ckpt["action_dim"],
        chunk_size=ckpt["chunk_size"],
        **ckpt["model_kwargs"],
    )
    state = ckpt["ema_state_dict"] if (use_ema and ckpt.get("ema_state_dict")) else ckpt["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    normalizer = ActionNormalizer.from_state_dict(ckpt["normalizer"])
    return model, normalizer, obs_spec, ckpt


def prepare_obs(raw, cameras, image_size: int, flip: bool, device) -> Dict[str, torch.Tensor]:
    """Simulator observation -> exactly what the dataset produced during training.

    The vertical flip and the joints+gripper proprio vector must match
    robobench/data/libero_dataset.py or the policy sees a distribution it never
    trained on and quietly fails.
    """
    key_map = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
    images = {}
    for cam in cameras:
        img = raw[key_map[cam]]
        if flip:
            img = img[::-1]
        img = np.ascontiguousarray(img)
        if img.shape[0] != image_size:
            ys = (np.arange(image_size) * img.shape[0] / image_size).astype(np.int64)
            xs = (np.arange(image_size) * img.shape[1] / image_size).astype(np.int64)
            img = img[ys][:, xs]
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        images[cam] = t.unsqueeze(0).to(device)

    proprio = np.concatenate(
        [np.asarray(raw["robot0_joint_pos"], dtype=np.float32).ravel(),
         np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).ravel()]
    )
    return images, torch.from_numpy(proprio).unsqueeze(0).to(device)


def evaluate(args) -> None:
    device = pick_device(args.device)
    model, normalizer, obs_spec, ckpt = load_policy(args.ckpt, device, use_ema=not args.no_ema)
    cfg = ckpt["config"]
    ablation = args.ablation or cfg.get("ablation", "none")

    lang_cache_path = args.lang_cache or cfg["data"]["lang_cache"]
    blob = np.load(lang_cache_path, allow_pickle=True)
    lang_cache = {str(k): np.asarray(v, np.float32) for k, v in zip(blob["keys"], blob["vectors"])}

    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero()
    suite_name = args.suite or cfg["data"]["suites"][0]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = args.max_steps or DEFAULT_MAX_STEPS.get(suite_name, 220)

    run_name = Path(args.ckpt).parent.name
    out_dir = Path(args.out_dir) / run_name
    tag = f"{suite_name}" + (f"_{ablation}" if ablation != "none" else "")
    episodes_path = out_dir / f"{tag}_episodes.jsonl"
    if episodes_path.exists() and not args.append:
        episodes_path.unlink()

    print(f"[eval] {run_name} | suite={suite_name} | {num_tasks} tasks x {args.episodes} eps "
          f"| max_steps={max_steps} | ablation={ablation}")

    writer = JsonlWriter(episodes_path)
    per_task: Dict[str, List[int]] = {}

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        instruction = task.language.strip()
        if instruction not in lang_cache:
            raise KeyError(
                f"instruction {instruction!r} missing from the language cache.\n"
                "Rebuild it over the same suites you are evaluating:\n"
                f"  python scripts/cache_language.py --suites {suite_name}"
            )
        lang = torch.from_numpy(lang_cache[instruction]).unsqueeze(0).to(device)

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=args.render_size,
            camera_widths=args.render_size,
        )
        init_states = task_suite.get_task_init_states(task_id)
        successes: List[int] = []

        bar = tqdm(range(args.episodes), desc=f"[{task_id + 1}/{num_tasks}] {instruction[:44]}", leave=False)
        for ep in bar:
            seed = args.seed * 100_000 + task_id * 1_000 + ep
            set_seed(seed, deterministic=True)
            env.seed(seed)
            env.reset()
            # Fixed init states make the comparison paired across models: every policy
            # sees the identical starting configuration for episode `ep`.
            init_idx = ep % len(init_states)
            raw = env.set_init_state(init_states[init_idx])

            # Objects are dropped in at reset; let the scene settle before acting.
            for _ in range(args.settle_steps):
                raw, _, _, _ = env.step([0.0] * 6 + [-1.0])

            success, steps = False, 0
            while steps < max_steps:
                images, proprio = prepare_obs(raw, obs_spec.cameras, cfg["data"]["image_size"],
                                              cfg["data"]["flip_images"], device)
                batch = ablate.apply({"images": images, "proprio": proprio, "lang": lang}, ablation)
                with torch.no_grad():
                    chunk = model.predict(batch["images"], batch["proprio"], batch["lang"])
                actions = normalizer.denormalize(chunk[0].float().cpu().numpy())

                for a in actions[: args.exec_horizon]:
                    raw, _, done, _ = env.step(np.clip(a, -1.0, 1.0).tolist())
                    steps += 1
                    if done:
                        success = True
                        break
                    if steps >= max_steps:
                        break
                if success:
                    break

            successes.append(int(success))
            writer.write({
                "suite": suite_name, "task_id": task_id, "task": instruction,
                "episode": ep, "init_state_idx": init_idx, "seed": seed,
                "success": int(success), "steps": steps, "ablation": ablation,
                "model": ckpt["model_name"], "ckpt_step": ckpt["step"],
            })
            bar.set_postfix(sr=f"{np.mean(successes):.2f}")

        env.close()
        per_task[instruction] = successes
        print(f"  task {task_id:2d}  {np.mean(successes) * 100:5.1f}%   {instruction}")

    writer.close()

    flat = np.array([s for v in per_task.values() for s in v], dtype=float)
    lo, hi = bootstrap_ci(flat, n_boot=10_000, seed=0)
    summary = {
        "run": run_name, "suite": suite_name, "ablation": ablation,
        "model": ckpt["model_name"], "ckpt_step": ckpt["step"],
        "episodes_per_task": args.episodes, "num_tasks": num_tasks,
        "success_rate": float(flat.mean()),
        "ci95": [float(lo), float(hi)],
        "per_task": {k: float(np.mean(v)) for k, v in per_task.items()},
        "episodes_file": str(episodes_path),
        "provenance": git_provenance(),
    }
    save_json(summary, out_dir / f"{tag}_summary.json")

    print(f"\n[result] {suite_name}: {flat.mean() * 100:.1f}%  (95% CI {lo * 100:.1f}-{hi * 100:.1f}, "
          f"n={len(flat)})")
    print(f"[result] per-episode outcomes -> {episodes_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Roll out a trained policy on LIBERO.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default=None, help="defaults to the suite the checkpoint trained on")
    ap.add_argument("--episodes", type=int, default=50, help="per task; 50 is the informal floor")
    ap.add_argument("--exec-horizon", type=int, default=8, help="actions executed per predicted chunk")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--render-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ablation", default=None, choices=[None, *ablate.MODES])
    ap.add_argument("--lang-cache", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--append", action="store_true")
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()
