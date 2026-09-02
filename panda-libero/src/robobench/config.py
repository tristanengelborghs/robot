"""Tiny config system: YAML file + dotted CLI overrides.

Deliberately not Hydra. A run is fully described by one resolved dict, which gets
written into the checkpoint and into every results file, so any number in the
table can be traced back to the exact configuration that produced it.
"""

from __future__ import annotations

import copy
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULTS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(text: str) -> Any:
    """Parse a CLI override value: JSON first, bare string as fallback."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def _merge_model_block(cfg: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge, except that a `model:` block naming a different architecture replaces
    the old block outright.

    Everything under `model:` other than `name` goes straight to that model's
    constructor, so the keys belong to one architecture. Deep-merging the default
    ResNet's `num_keypoints` into a config that switches to `vla` would hand the
    VLA a kwarg it has never heard of.
    """
    new_name = override.get("model", {}).get("name") if isinstance(override.get("model"), dict) else None
    if new_name is not None and new_name != cfg.get("model", {}).get("name"):
        cfg = dict(cfg)
        cfg["model"] = {}
    return _deep_merge(cfg, override)


def load_config(path: str | None = None, overrides: List[str] | None = None) -> Dict[str, Any]:
    """Load defaults, merge an optional config file, then apply `key.sub=value` overrides.

    Switching architectures — `model.name` in the file or on the command line —
    starts the `model:` block fresh; the other keys under it are then whatever
    that file or those overrides say, in any order.
    """
    with open(DEFAULTS_PATH) as f:
        cfg = yaml.safe_load(f) or {}

    if path:
        with open(path) as f:
            cfg = _merge_model_block(cfg, yaml.safe_load(f) or {})

    parsed = []
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got '{item}'")
        dotted, raw = item.split("=", 1)
        parsed.append((dotted.strip(), _coerce(raw.strip())))

    # The architecture switch goes first so it cannot wipe out sibling overrides.
    for dotted, value in sorted(parsed, key=lambda kv: kv[0] != "model.name"):
        if dotted == "model.name" and value != cfg.get("model", {}).get("name"):
            cfg["model"] = {}
        _set_dotted(cfg, dotted, value)

    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable short hash of a resolved config — goes in run names and results files."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
