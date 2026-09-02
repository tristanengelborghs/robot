#!/usr/bin/env python3
"""Load the real VLA backbone once and check everything that the offline tests cannot.

    python scripts/check_vla_backbone.py                 # defaults: vla_goal.yaml's backbone, CPU or GPU
    python scripts/check_vla_backbone.py --image-res 256

Verifies, in order: the Hub download and `from_pretrained` under the installed
transformers; that the hand-built prompt is token-for-token what the HF processor
produces (skipped if `num2words`, which the processor imports, is absent); one
forward + backward with finite gradients on every trainable tensor, timed; the
trainable-only checkpoint's size; and that the evaluator's non-strict loader
rebuilds the model from that checkpoint and predicts identically.

Needs a couple of GB of disk for the weights and a few GB of RAM. Run it on the
GPU box before the first real training run.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robobench.eval import load_policy  # noqa: E402
from robobench.models.base import ObsSpec  # noqa: E402
from robobench.models.vla import DEFAULT_BACKBONE, VLAPolicy  # noqa: E402
from robobench.utils import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--image-res", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = "auto" if device.type == "cuda" else "fp32"
    torch.manual_seed(0)
    spec = ObsSpec(cameras=["agentview", "wrist"], image_size=128, proprio_dim=9, lang_dim=512)

    t0 = time.time()
    model = VLAPolicy(spec, action_dim=7, chunk_size=8, backbone=args.backbone,
                      image_res=args.image_res, backbone_dtype=dtype).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[build] {time.time() - t0:.1f}s  {type(model.backbone).__name__}  dtype={model.backbone.dtype}  device={device}")
    print(f"[params] total {n_total / 1e6:.1f}M, trainable {n_train / 1e6:.2f}M, "
          f"{sum(1 for m in model.modules() if type(m).__name__ == 'LoRALinear')} LoRA layers, "
          f"{model.n_img_tok} tokens per camera")

    # --- prompt vs the HF processor ---------------------------------------
    instr = "put the bowl on the stove"
    try:
        import num2words  # noqa: F401  (the SmolVLM processor imports it)
        from PIL import Image
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(args.backbone)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": instr}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        imgs = [Image.fromarray(np.full((args.image_res, args.image_res, 3), 128, np.uint8))] * 2
        proc.image_processor.do_image_splitting = False
        proc.image_processor.max_image_size = {"longest_edge": args.image_res}
        proc.image_processor.size = {"longest_edge": args.image_res}
        enc = proc(text=prompt, images=imgs, return_tensors="pt")
        theirs, ours = enc["input_ids"][0], model._prompt_ids(instr)
        same = torch.equal(theirs, ours)
        print(f"[prompt] processor {theirs.numel()} ids, ours {ours.numel()} ids, identical={same}")
        if not same:
            print("   processor:", repr(proc.tokenizer.decode(theirs))[:400])
            print("   ours:     ", repr(model.tok.decode(ours))[:400])
            raise SystemExit("prompt mismatch — fix vla.PROMPT / img_block before training")
        pv = enc["pixel_values"]
        x = model._prep(torch.full((1, 3, 128, 128), 128, dtype=torch.uint8))
        print(f"[pixels] processor {tuple(pv.shape)} in [{pv.min():.2f}, {pv.max():.2f}]; "
              f"ours {tuple(x.shape)} in [{x.min():.2f}, {x.max():.2f}]")
    except (ImportError, ValueError) as e:
        # The processor pulls in num2words and torchvision; neither is needed to train.
        print(f"[prompt] processor comparison skipped: {str(e)[:160]}")
        print("         pip install num2words torchvision, then rerun, to check the prompt against the processor")

    # --- one training step ---------------------------------------------------
    B = args.batch_size
    batch = {
        "images": {c: torch.randint(0, 255, (B, 3, 128, 128), dtype=torch.uint8, device=device) for c in spec.cameras},
        "proprio": torch.randn(B, 9, device=device), "lang": torch.zeros(B, 512, device=device),
        "text": [instr, "turn on the stove"][:B] + [instr] * max(0, B - 2),
        "actions": torch.rand(B, 8, 7, device=device) * 2 - 1, "mask": torch.ones(B, 8, device=device),
    }
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model.loss(batch)
    out["loss"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, g in grads if g is None or not torch.isfinite(g).all()]
    print(f"[step] loss {out['loss'].item():.4f}, forward+backward {dt:.2f}s at batch {B}; "
          f"{len(grads)} trainable tensors, {len(bad)} without a finite grad")
    if device.type == "cuda":
        print(f"[memory] peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    if bad:
        raise SystemExit(f"non-finite or missing gradients: {bad[:5]}")

    # --- checkpoint round trip through the evaluator's loader -----------------
    sd = model.trainable_state_dict()
    mb = sum(v.numel() * v.element_size() for v in sd.values()) / 1e6
    tmp = Path(tempfile.mkdtemp()) / "ckpt_last.pt"
    torch.save({
        "step": 1, "config": {"data": {"lang_cache": None, "image_size": 128, "flip_images": True, "suites": ["libero_goal"]}},
        "model_name": "vla",
        "model_kwargs": {"backbone": args.backbone, "image_res": args.image_res, "backbone_dtype": dtype},
        "state_dict": sd, "ema_state_dict": None,
        "normalizer": {"low": [-1.0] * 7, "high": [1.0] * 7},
        "obs_spec": spec.__dict__, "action_dim": 7, "chunk_size": 8, "instructions": [instr],
    }, tmp)
    print(f"[ckpt] {len(sd)} tensors, {mb:.1f} MB of parameters, {tmp.stat().st_size / 1e6:.1f} MB on disk")
    t0 = time.time()
    model2, _, _, _ = load_policy(str(tmp), device)
    print(f"[eval loader] rebuilt and loaded in {time.time() - t0:.1f}s")
    model.eval()
    with torch.no_grad():
        a = model.predict(batch["images"], batch["proprio"], batch["lang"], text=batch["text"])
        b = model2.predict(batch["images"], batch["proprio"], batch["lang"], text=batch["text"])
    print(f"[eval loader] predictions identical: {torch.equal(a, b)}")
    tmp.unlink()
    print("OK")


if __name__ == "__main__":
    main()
