"""Adversarial attacks."""

from .base_attack import AttackConfig, AttackResult, BaseAttack
from .registry import AttackRegistry, register_all_attacks

register_all_attacks()

__all__ = ["AttackConfig", "AttackResult", "BaseAttack", "AttackRegistry"]
