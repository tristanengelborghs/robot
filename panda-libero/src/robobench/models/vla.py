"""VLA — a pretrained vision-language model with an action head bolted on.

The other architectures in this directory are trained from scratch. This one is
the pretrained column of the same table: SmolVLM2 (a SigLIP vision tower feeding
a 360M-parameter SmolLM2 through a pixel-shuffle connector) reads both camera
frames and the raw instruction string, and a small head regresses the action
chunk from its final hidden states. The recipe is OpenVLA-OFT's — parallel
decoding of a chunk with L1 regression, no autoregressive action tokens — which
is what took OpenVLA from 76% to 97% on LIBERO, and it lands exactly on the
masked-L1 objective the harness already provides.

How the pieces fit:

    images (uint8)  -> resize to `image_res`, (x/255 - 0.5)/0.5 -> SigLIP -> connector
    instruction     -> "<|im_start|>User:<img><img>{text}<end_of_utterance>\\nAssistant:"
    proprio         -> one extra token (and a side channel into the head)
    K action slots  -> K learned query embeddings appended to the sequence
                       language model (causal)
    hidden states at the K slots -> LayerNorm -> concat proprio -> MLP -> (K, 7)

The prompt is built by hand rather than through the HF processor, byte for byte
what the processor would produce, so the model sees the format it was trained
on without the processor's dependencies (torchvision, num2words). Each `<image>`
placeholder expands to `(image_res/64)^2` tokens — 64 at 512 px, 16 at 256 px —
and the image features are computed here and handed to the language model as
`image_hidden_states`, bypassing the backbone's own image path. That path drops
any image whose normalised pixels are exactly zero as "padding", which a zeroed
camera in the `no_vision` ablation would otherwise trip; here a black frame is a
black frame.

The instruction enters as text. `lang`, the cached CLIP vector every other model
reads, is accepted and ignored; the `no_lang` ablation blanks the string.

What trains: LoRA adapters on the language model's attention projections (or
nothing in the backbone, with `finetune=frozen`), the action queries, the proprio
projections and the head. The backbone is loaded in bf16 and never touched, so
`trainable_state_dict` — what the checkpoint stores and the EMA tracks — is a few
million parameters, not the half billion that would otherwise be written twice
per checkpoint and averaged every step.

`backbone="tiny"` builds a random two-layer stand-in with the same architecture
and an in-memory tokenizer, so the tests exercise every line here without a
download and in under a second.
"""

from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from robobench.models.base import BasePolicy, ObsSpec
from robobench.models.modules import MLP
from robobench.registry import register_model

DEFAULT_BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TINY = "tiny"

# SmolVLM's tokenizer conventions. The processor wraps every image in
# <fake_token_around_image> and, when the image is not tiled, marks it <global-img>.
BOS = "<|im_start|>"
END_OF_UTTERANCE = "<end_of_utterance>"
FAKE_IMAGE = "<fake_token_around_image>"
IMAGE = "<image>"
GLOBAL_IMAGE = "<global-img>"
SPECIALS = ("<|endoftext|>", BOS, "<|im_end|>", FAKE_IMAGE, IMAGE, END_OF_UTTERANCE, GLOBAL_IMAGE)

PROMPT = BOS + "User:{images}{instruction}" + END_OF_UTTERANCE + "\nAssistant:"


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update, y = Wx + (B A x) * alpha/r.

    `B` starts at zero so the wrapped model is bit-identical to the pretrained one
    at step 0. The adapters are fp32 regardless of the base layer's dtype; the
    update is cast to the base output's dtype when added.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = (self.dropout(x).to(self.A.dtype) @ self.A.t()) @ self.B.t()
        return y + (delta * self.scale).to(y.dtype)


@register_model("vla")
class VLAPolicy(BasePolicy):
    strict_load = False  # the checkpoint holds the adapters and head; the backbone is rebuilt

    def __init__(
        self,
        obs_spec: ObsSpec,
        action_dim: int,
        chunk_size: int,
        backbone: str = DEFAULT_BACKBONE,      # HF repo id, or "tiny" for a random offline stand-in
        finetune: str = "lora",                # lora | frozen
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        lora_targets: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        hidden_dim: int = 512,                 # width of the proprio projections and the head
        image_res: int = 512,                  # what the vision tower sees; a multiple of 64
        freeze_vision: bool = True,            # vision tower + connector under no_grad
        gradient_checkpointing: bool = False,
        backbone_dtype: str = "auto",          # auto (bf16 on CUDA, fp32 elsewhere) | bf16 | fp32
        proprio_token: bool = True,            # also feed proprio to the LM as one token
        drop_lm_head: bool = True,             # the 47M-parameter vocabulary head is never used
    ):
        super().__init__(obs_spec, action_dim, chunk_size)
        self.cameras = list(obs_spec.cameras)
        if finetune not in ("lora", "frozen"):
            raise ValueError(f"finetune must be 'lora' or 'frozen', got {finetune!r}")
        self.finetune = finetune
        self.freeze_vision = freeze_vision

        if backbone == TINY:
            self.backbone, self.tok = _tiny_backbone()
            image_res = self.backbone.config.vision_config.image_size
        else:
            self.backbone, self.tok = _pretrained_backbone(backbone, backbone_dtype)

        cfg = self.backbone.config
        patch = cfg.vision_config.patch_size
        scale = cfg.scale_factor
        if image_res % (patch * scale) != 0:
            raise ValueError(f"image_res must be a multiple of {patch * scale}, got {image_res}")
        self.image_res = image_res
        self.n_img_tok = (image_res // patch) ** 2 // scale**2
        self.width = cfg.text_config.hidden_size
        self.image_token_id = cfg.image_token_id
        self.pad_id = cfg.text_config.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tok.pad_token_id
        if self.tok.convert_tokens_to_ids(IMAGE) != self.image_token_id:
            raise RuntimeError("tokenizer and config disagree on the <image> token id")

        # -- freeze, then open exactly what should train ----------------------
        self.backbone.requires_grad_(False)
        if drop_lm_head:
            self.backbone.lm_head = nn.Identity()
        if finetune == "lora":
            self._wrap_lora(tuple(lora_targets), lora_rank, lora_alpha, lora_dropout)
        if not freeze_vision:
            self.backbone.model.vision_model.requires_grad_(True)
            self.backbone.model.connector.requires_grad_(True)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # -- the trainable part, fp32 -----------------------------------------
        D = self.width
        self.action_queries = nn.Parameter(torch.randn(chunk_size, D) * 0.02)
        self.proprio_token_proj = MLP([obs_spec.proprio_dim, hidden_dim, D]) if proprio_token else None
        self.proprio_proj = MLP([obs_spec.proprio_dim, hidden_dim, hidden_dim])
        self.out_norm = nn.LayerNorm(D)
        self.head = MLP([D + hidden_dim, hidden_dim, action_dim])
        last = [m for m in self.head.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.bias)
        nn.init.normal_(last.weight, std=0.01)

        # -- prompts ----------------------------------------------------------
        self.img_block = FAKE_IMAGE + GLOBAL_IMAGE + IMAGE * self.n_img_tok + FAKE_IMAGE
        self._prompt_cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ setup

    def _wrap_lora(self, targets: Tuple[str, ...], rank: int, alpha: float, dropout: float) -> None:
        wrapped = 0
        for layer in self.backbone.model.text_model.layers:
            for name in targets:
                for block in (layer.self_attn, layer.mlp):
                    if hasattr(block, name):
                        setattr(block, name, LoRALinear(getattr(block, name), rank, alpha, dropout))
                        wrapped += 1
        if wrapped == 0:
            raise ValueError(f"no modules named {targets} in the language model")

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in keep}

    # ---------------------------------------------------------------- prompts

    def _prompt_ids(self, text: str) -> torch.Tensor:
        ids = self._prompt_cache.get(text)
        if ids is None:
            prompt = PROMPT.format(images=self.img_block * len(self.cameras), instruction=text)
            ids = torch.tensor(self.tok(prompt, add_special_tokens=False)["input_ids"], dtype=torch.long)
            n_img = int((ids == self.image_token_id).sum())
            if n_img != self.n_img_tok * len(self.cameras):
                raise RuntimeError(f"prompt has {n_img} image tokens, expected {self.n_img_tok * len(self.cameras)}")
            self._prompt_cache[text] = ids
        return ids

    @property
    def n_tail(self) -> int:
        return self.chunk_size + (1 if self.proprio_token_proj is not None else 0)

    def _batch_prompts(self, text: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-pad the prompts and reserve `n_tail` attended slots at the end, so
        the proprio token and the K action queries always sit at positions -n_tail:."""
        seqs = [self._prompt_ids(t) for t in text]
        T = max(s.numel() for s in seqs) + self.n_tail
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            start = T - self.n_tail - s.numel()
            ids[i, start : T - self.n_tail] = s
            attn[i, start:] = 1
        return ids.to(device), attn.to(device)

    # ----------------------------------------------------------------- images

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        """uint8 (B, 3, H, W) -> what SigLIP was trained on: `image_res` square, in [-1, 1]."""
        x = img.float() / 255.0
        R = self.image_res
        if x.shape[-2] != R or x.shape[-1] != R:
            x = F.interpolate(x, size=(R, R), mode="bicubic", align_corners=False, antialias=R < x.shape[-1])
        return (x - 0.5) / 0.5

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang: torch.Tensor,
        text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        B = proprio.shape[0]
        K = self.chunk_size
        if text is None:
            text = [""] * B
        if len(text) != B:
            raise ValueError(f"got {len(text)} instructions for a batch of {B}")
        model = self.backbone.model

        # images -> features, computed here so a black frame is a frame, not padding
        pv = torch.stack([self._prep(images[c]) for c in self.cameras], dim=1)  # (B, n_cam, 3, R, R)
        with torch.no_grad() if self.freeze_vision else contextlib.nullcontext():
            vis = model.vision_model(pixel_values=pv.flatten(0, 1).to(model.dtype))
            img_h = model.connector(vis.last_hidden_state)  # (B*n_cam, n_img_tok, D)

        # prompt tokens, then overwrite the reserved tail with proprio + action queries
        ids, attn = self._batch_prompts(text, proprio.device)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        emb = model.get_input_embeddings()(ids)
        tail = [] if self.proprio_token_proj is None else [self.proprio_token_proj(proprio)[:, None]]
        tail.append(self.action_queries[None].expand(B, -1, -1))
        tail = torch.cat(tail, dim=1).to(emb.dtype)
        emb = torch.cat([emb[:, : -self.n_tail], tail], dim=1)

        out = model(
            input_ids=ids,
            inputs_embeds=emb,
            attention_mask=attn,
            position_ids=pos,
            image_hidden_states=img_h,
            use_cache=False,
            return_dict=True,
        )
        h = self.out_norm(out.last_hidden_state[:, -K:].float())  # (B, K, D)
        p = self.proprio_proj(proprio)[:, None].expand(-1, K, -1)
        return self.head(torch.cat([h, p], dim=-1)).float()


# ---------------------------------------------------------------- backbones


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[name]


def _pretrained_backbone(repo: str, dtype_name: str):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=_resolve_dtype(dtype_name))
    model.config.use_cache = False
    return model, tok


def _tiny_backbone():
    """A random SmolVLM the size of a unit test: 32-px images, one layer each side.

    The tokenizer is character-level over printable ASCII with SmolVLM's special
    tokens on top, built in memory, so instructions tokenise to something an
    ablation can destroy and nothing is downloaded.
    """
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, SmolVLMConfig, SmolVLMForConditionalGeneration

    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    vocab = {t: i for i, t in enumerate(list(SPECIALS) + ["<pad>", "[UNK]"] + chars)}
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        pad_token="<pad>",
        bos_token=BOS,
        eos_token=END_OF_UTTERANCE,
        additional_special_tokens=list(SPECIALS),
    )
    cfg = SmolVLMConfig(
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, image_size=32, patch_size=16,
        ),
        text_config=dict(
            model_type="llama", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=len(tok),
            max_position_embeddings=1024, pad_token_id=tok.pad_token_id,
        ),
        image_token_id=tok.convert_tokens_to_ids(IMAGE),
        scale_factor=2,
        pad_token_id=tok.pad_token_id,
        use_cache=False,
    )
    torch.manual_seed(0)
    return SmolVLMForConditionalGeneration(cfg), tok
