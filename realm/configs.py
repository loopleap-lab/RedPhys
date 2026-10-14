"""Hyperparameters from configs/*.yaml."""

import dataclasses
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(path, config_class) -> dict:
    """YAML config fields (lists -> tuples where needed)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    fields = {f.name: f for f in dataclasses.fields(config_class)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        sys.exit(f"ERROR: {path}: {unknown} not in {config_class.__name__} fields: {', '.join(sorted(fields))}")
    return {k: tuple(v) if isinstance(v, list) and isinstance(fields[k].default, tuple) else v for k, v in data.items()}


def with_overrides(config: dict, config_class, params: dict) -> dict:
    """`config` with keyword settings of `config_class` on top."""
    unknown = sorted(set(params) - {f.name for f in dataclasses.fields(config_class)})
    if unknown:
        raise ValueError(f"{config_class.__name__} has no setting {unknown}; its settings: {', '.join(f.name for f in dataclasses.fields(config_class))}")
    return {**config, **params}


def default_config_path(kind: str, name: str):
    """configs/<kind>/<name>.yaml, or None."""
    path = CONFIG_DIR / kind / f"{name}.yaml"
    return path if path.exists() else None
