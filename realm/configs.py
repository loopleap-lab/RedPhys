"""Run settings: configs/*.yaml with overrides, the config recorded in manifests, per-sample seeding."""

import dataclasses
import json
from pathlib import Path

import yaml

from .data.files import PROJECT_ROOT, relpath

CONFIG_DIR = PROJECT_ROOT / "configs"


def seed_all(seed: int, key: str) -> None:
    """Seed python, numpy and torch (all devices) from (seed, key): a sample's randomness does not depend on run order."""
    import random
    import numpy as np
    import torch
    s = random.Random(f"{seed}-{key}").getrandbits(32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def seeded(fn, seed: int, key: str, label: str):
    """fn() seeded from (seed, key); NaN output (to_pil raises FloatingPointError): once more from (seed, key-retry)."""
    try:
        seed_all(seed, key)
        return fn()
    except FloatingPointError as e:
        print(f"\nWARNING: {label}: {e}, retrying once")
        seed_all(seed, f"{key}-retry")
        return fn()


def _fields(config_class) -> dict:
    return {f.name: f for f in dataclasses.fields(config_class)}


def _tuples(data: dict, fields: dict) -> dict:
    """Lists -> tuples for tuple-valued fields."""
    return {k: tuple(v) if isinstance(v, list) and isinstance(fields[k].default, tuple) else v for k, v in data.items()}


def load_config(path, config_class) -> dict:
    """YAML config fields (lists -> tuples where needed); {} without a path."""
    data = (yaml.safe_load(Path(path).read_text()) or {}) if path else {}
    fields = _fields(config_class)
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"{path}: {unknown} not in {config_class.__name__} fields: {', '.join(sorted(fields))}")
    return _tuples(data, fields)


def with_overrides(config: dict, config_class, params: dict) -> dict:
    """`config` with keyword settings of `config_class` on top."""
    fields = _fields(config_class)
    unknown = sorted(set(params) - set(fields))
    if unknown:
        raise ValueError(f"{config_class.__name__} has no setting {unknown}; its settings: {', '.join(fields)}")
    return {**config, **_tuples(params, fields)}


def default_config_path(kind: str, name: str):
    """configs/<kind>/<name>.yaml, or None."""
    path = CONFIG_DIR / kind / f"{name}.yaml"
    return path if path.exists() else None


def build_config(kind: str, name: str, config_class, path, params: dict, base=None) -> dict:
    """Config kwargs: base (run inputs) < the YAML (path, else configs/<kind>/<name>.yaml) < params."""
    path = path or default_config_path(kind, name)
    print(f"Config:  {path or '(code defaults)'}")
    return with_overrides({**(base or {}), **load_config(path, config_class)}, config_class, params)


def config_snapshot(config, drop=()) -> dict:
    """A config as manifest.json records it (JSON types, so a resume compares equal): no device or key fields, paths project-relative."""
    return json.loads(json.dumps({k: relpath(v) if isinstance(v, str) and Path(v).is_absolute() else v
                                  for k, v in dataclasses.asdict(config).items() if k not in drop and k != "device" and "key" not in k}))
