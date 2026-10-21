"""Input defenses."""

from .base_defense import BaseDefense, DefenseConfig, DefenseResult
from .registry import DefenseRegistry, register_all_defenses

register_all_defenses()

__all__ = ["BaseDefense", "DefenseConfig", "DefenseResult", "DefenseRegistry"]
