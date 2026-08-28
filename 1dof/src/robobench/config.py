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


def load_config(path: str | None = None, overrides: List[str] | None = None) -> Dict[str, Any]:
    """Load defaults, merge an optional config file, then apply `key.sub=value` overrides."""
    with open(DEFAULTS_PATH) as f:
        cfg = yaml.safe_load(f) or {}

    if path:
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got '{item}'")
        dotted, raw = item.split("=", 1)
        _set_dotted(cfg, dotted.strip(), _coerce(raw.strip()))

    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable short hash of a resolved config — goes in run names and results files."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
