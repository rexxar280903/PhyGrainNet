"""YAML config loading with `base:` inheritance and dotted CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("base", None)
    if base:
        bases = base if isinstance(base, list) else [base]
        merged: dict = {}
        for b in bases:
            merged = deep_merge(merged, load_config((path.parent / b).resolve()))
        cfg = deep_merge(merged, cfg)
    for item in overrides or []:
        apply_override(cfg, item)
    return cfg


def apply_override(cfg: dict, item: str) -> None:
    """Apply `a.b.c=value` (value parsed as YAML, so 1e-3, true, [1,2] work)."""
    if "=" not in item:
        raise ValueError(f"override must look like key=value, got {item!r}")
    key, raw = item.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
