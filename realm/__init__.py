"""REALM: red-teaming benchmark for physical-world VLMs."""

__version__ = "0.2.0"

import os

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")

from .data import Sample

_RUNNERS = {"attack": ".attacks.run", "defend": ".defense.run", "target": ".agent.scenario_agent.run",
            "evaluate": ".evaluation", "score": ".evaluation"}


def __getattr__(name):
    if name in _RUNNERS:
        import importlib

        return getattr(importlib.import_module(_RUNNERS[name], __name__), name)
    raise AttributeError(f"module 'realm' has no attribute {name!r}")


__all__ = ["Sample", "attack", "defend", "target", "evaluate", "score"]
