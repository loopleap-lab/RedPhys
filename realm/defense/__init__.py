"""Defense methods."""

__version__ = "0.1.0"

from .base_defense import DefenseConfig, DefenseResult, BaseDefense
from .pad.pad_defense import PADDefenseConfig, PADDefense
from .freqpure.freqpure_defense import FreqPureDefenseConfig, FreqPureDefense
from .bluesuffix.bluesuffix_defense import BlueSuffixDefenseConfig, BlueSuffixDefense
from .et3.et3_defense import ET3DefenseConfig, ET3Defense
from .oscp.oscp_defense import OSCPDefenseConfig, OSCPDefense
from .registry import DefenseRegistry, DefenseSpec, register_all_defenses

register_all_defenses()

__all__ = [
    "DefenseConfig", "DefenseResult", "BaseDefense",
    "PADDefenseConfig", "PADDefense",
    "FreqPureDefenseConfig", "FreqPureDefense",
    "BlueSuffixDefenseConfig", "BlueSuffixDefense",
    "ET3DefenseConfig", "ET3Defense",
    "OSCPDefenseConfig", "OSCPDefense",
    "DefenseRegistry", "DefenseSpec",
]
