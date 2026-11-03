"""RedPhys: red-teaming benchmark for physical-world VLMs."""

__version__ = "0.2.0"

import os

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")

from .data import Sample

_RUNNERS = {"attack": ".attacks.run", "defend": ".defense.run", "target": ".agent.scenario_agent.run",
            "evaluate": ".evaluation", "score": ".evaluation", "report": ".evaluation"}


def __getattr__(name):
    if name in _RUNNERS:
        import importlib

        return getattr(importlib.import_module(_RUNNERS[name], __name__), name)
    raise AttributeError(f"module 'redphys' has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_RUNNERS})


def methods() -> dict[str, list[str]]:
    """Registered names: {"attack": [...], "defense": [...]}."""
    from .attacks import AttackRegistry
    from .defense import DefenseRegistry

    return {"attack": AttackRegistry.names(), "defense": DefenseRegistry.names()}


__all__ = ["Sample", "attack", "defend", "target", "evaluate", "score", "report", "methods"]
