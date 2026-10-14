"""Defense registry."""

from dataclasses import dataclass
from typing import Dict, Type

from .base_defense import BaseDefense, DefenseConfig


@dataclass
class DefenseSpec:
    name: str
    category: str                     # "purification", "detection", ...
    defense_class: Type[BaseDefense]
    config_class: Type[DefenseConfig]


class DefenseRegistry:
    _registry: Dict[str, DefenseSpec] = {}

    @classmethod
    def register(cls, spec: DefenseSpec) -> None:
        if spec.name in cls._registry:
            raise ValueError(f"Defense '{spec.name}' already registered")
        cls._registry[spec.name] = spec

    @classmethod
    def create(cls, name: str, **config_kwargs) -> BaseDefense:
        spec = cls.get_spec(name)
        return spec.defense_class(spec.config_class(**config_kwargs))

    @classmethod
    def get_spec(cls, name: str) -> DefenseSpec:
        if name not in cls._registry:
            raise ValueError(f"Unknown defense: {name}. Available: {list(cls._registry)}")
        return cls._registry[name]


def register_all_defenses():
    from .pad.pad_defense import PADDefense, PADDefenseConfig
    from .freqpure.freqpure_defense import FreqPureDefense, FreqPureDefenseConfig
    from .bluesuffix.bluesuffix_defense import BlueSuffixDefense, BlueSuffixDefenseConfig
    from .et3.et3_defense import ET3Defense, ET3DefenseConfig
    from .oscp.oscp_defense import OSCPDefense, OSCPDefenseConfig

    for name, defense_class, config_class in [
        ("pad", PADDefense, PADDefenseConfig),
        ("freqpure", FreqPureDefense, FreqPureDefenseConfig),
        ("bluesuffix", BlueSuffixDefense, BlueSuffixDefenseConfig),
        ("et3", ET3Defense, ET3DefenseConfig),
        ("oscp", OSCPDefense, OSCPDefenseConfig),
    ]:
        DefenseRegistry.register(DefenseSpec(name, "purification", defense_class, config_class))
