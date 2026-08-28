"""YAML config + dotted CLI overrides, the sibling project's pattern.

One resolved dict fully describes a run and is written into every checkpoint
and log, so any number in a results table traces back to the exact settings
that produced it. Deliberately not Hydra, for the same reason as before: the
whole surface fits in a page and never surprises anyone.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULTS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _coerce(text: str) -> Any:
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config(path: str | None = None, overrides: List[str] | None = None) -> Dict[str, Any]:
    with open(DEFAULTS_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    if path:
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got '{item}'")
        dotted, raw = item.split("=", 1)
        node = cfg
        keys = dotted.strip().split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _coerce(raw.strip())
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
