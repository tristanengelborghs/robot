"""Train the diffusion policy on recorded demonstrations.

Needs no simulator: a recorded dataset is an HDF5 file and training is torch, so
this runs on the GPU box or on a laptop with equal correctness and very
different speed. Evaluation is the part that needs Isaac Lab; see
scripts/eval_diffusion.py.

    PYTHONPATH=. .venv/bin/python scripts/train_diffusion.py \\
        --dataset datasets/lift_scripted.hdf5 --epochs 200

Both the observations and the actions are normalised, and the normalisers are
saved with the weights. Actions especially: this task's action space mixes
position deltas of order 0.01 with a gripper command of exactly +/-1, and asking
a network to predict noise on both at once, unnormalised, wastes most of its
capacity on the one with the larger scale.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.demo_data import Normalizer, build_dataset, load_episodes, split_episodes  # noqa: E402
from harness.diffusion import DiffusionConfig, DiffusionPolicy, NoiseSchedule, diffusion_loss  # noqa: E402
from harness.training import iterate_batches, pick_device, save_checkpoint  # noqa: E402

# The observation terms a pick-and-lift policy sees. `target_object_position` is
# deliberately absent: these demonstrations ignore the commanded goal and pick
# the block up, so feeding the goal in would train the policy to ignore it.
LIFT_OBS_KEYS = ("joint_pos", "joint_vel", "object_position", "actions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="datasets/lift_scripted.hdf5")
    parser.add_argument("--out", default="runs/diffusion", help="Directory for the checkpoint and metrics.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon", type=int, default=8, help="Action chunk length.")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def to_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def evaluate(policy, schedule, obs, actions, batch_size: int) -> float:
    """Mean denoising loss over a set, with no gradients.

    This is the honest number to watch, but note what it does and does not say:
    it measures how well the policy predicts noise on held-out demonstrations,
    not whether it can pick the block up. Only a rollout answers that.
    """
    policy.eval()
    total, seen = 0.0, 0
    rng = np.random.default_rng(0)

    with torch.no_grad():
        for index in iterate_batches(len(obs), batch_size, rng, shuffle=False):
            batch = torch.as_tensor(index, device=obs.device)
            loss = diffusion_loss(policy, schedule, obs[batch], actions[batch])
            total += loss.item() * len(index)
            seen += len(index)

    policy.train()
    return total / max(seen, 1)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    episodes = load_episodes(args.dataset, obs_keys=LIFT_OBS_KEYS)
    train_episodes, val_episodes = split_episodes(episodes, args.validation_episodes, seed=args.seed)
    train_set = build_dataset(train_episodes, args.horizon)
    val_set = build_dataset(val_episodes, args.horizon)

    print(f"device      {device}")
    print(f"train       {train_set.format()}")
    print(f"validation  {val_set.format()}", flush=True)

    # Fitted on the training split alone; the validation split must not inform
    # the scaling any more than it informs the weights.
    obs_normalizer = Normalizer.fit(train_set.obs)
    action_normalizer = Normalizer.fit(train_set.action_chunks.reshape(-1, train_set.action_dim))

    def prepare(dataset):
        obs = to_tensor(obs_normalizer.normalize(dataset.obs), device)
        flat = dataset.action_chunks.reshape(-1, dataset.action_dim)
        actions = action_normalizer.normalize(flat).reshape(dataset.action_chunks.shape)
        return obs, to_tensor(actions, device)

    train_obs, train_actions = prepare(train_set)
    val_obs, val_actions = prepare(val_set)

    cfg = DiffusionConfig(
        obs_dim=train_set.obs_dim,
        action_dim=train_set.action_dim,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        num_steps=args.diffusion_steps,
    )
    policy = DiffusionPolicy(cfg).to(device)
    schedule = NoiseSchedule(cfg.num_steps).to(device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    parameters = sum(p.numel() for p in policy.parameters())
    print(f"policy      {parameters:,} parameters\n", flush=True)

    out_dir = Path(args.out)
    rng = np.random.default_rng(args.seed)
    history: list[dict] = []
    best_val = float("inf")
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss, batches = 0.0, 0
        for index in iterate_batches(len(train_obs), args.batch_size, rng):
            batch = torch.as_tensor(index, device=device)
            loss = diffusion_loss(policy, schedule, train_obs[batch], train_actions[batch])

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            batches += 1

        train_loss = epoch_loss / max(batches, 1)
        val_loss = evaluate(policy, schedule, val_obs, val_actions, args.batch_size)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        # Keep the checkpoint that generalises best, not the last one. With a
        # hundred near-identical demonstrations, later is not better.
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "policy.pt",
                policy,
                obs_normalizer,
                action_normalizer,
                extra={"epoch": epoch, "val_loss": val_loss, "obs_keys": list(LIFT_OBS_KEYS)},
            )

        if epoch % args.log_every == 0 or epoch == 1:
            elapsed = time.time() - started
            print(
                f"epoch {epoch:4d}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}"
                f"  best {best_val:.4f}  {elapsed:.0f}s",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nbest validation loss {best_val:.4f}")
    print(f"checkpoint {out_dir / 'policy.pt'}")
    print("\nA low denoising loss says the policy predicts noise well on held-out")
    print("demonstrations. Whether it can lift the block is a different question;")
    print("run scripts/eval_diffusion.py on the box to answer it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
