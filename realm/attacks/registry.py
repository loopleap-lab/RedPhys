"""Attack registry."""

from dataclasses import dataclass
from typing import Type, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_attack import BaseAttack, AttackConfig


@dataclass
class AttackSpec:
    name: str
    category: str
    attack_class: Type["BaseAttack"]
    config_class: Type["AttackConfig"]


class AttackRegistry:

    _registry: Dict[str, AttackSpec] = {}

    @classmethod
    def register(cls, spec: AttackSpec) -> None:
        cls._registry[spec.name] = spec

    @classmethod
    def create(cls, name: str, **config_kwargs) -> "BaseAttack":
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown attack: {name}. Available: {available}")

        spec = cls._registry[name]
        config = spec.config_class(**config_kwargs)
        return spec.attack_class(config)

    @classmethod
    def get_spec(cls, name: str) -> Optional[AttackSpec]:
        return cls._registry.get(name)


def register_all_attacks():
    from .physpatch.physpatch_attack import PhysPatchAttack, PhysPatchConfig
    from .foa.foa_attack import FOAAttack, FOAAttackConfig
    from .mattack.mattack_attack import MAttack, MAttackConfig
    from .oattack.oattack_attack import OAttack, OAttackConfig
    from .mpcattack.mpcattack_attack import MPCAttack, MPCAttackConfig
    from .coa.coa_attack import COAAttack, COAAttackConfig
    from .advdiffvlm.advdiffvlm_attack import AdvDiffVLMAttack, AdvDiffVLMConfig

    AttackRegistry.register(AttackSpec(
        name="physpatch",
        category="physical",
        attack_class=PhysPatchAttack,
        config_class=PhysPatchConfig,
    ))

    AttackRegistry.register(AttackSpec(
        name="foa",
        category="physical",
        attack_class=FOAAttack,
        config_class=FOAAttackConfig,
    ))

    AttackRegistry.register(AttackSpec(
        name="mattack",
        category="physical",
        attack_class=MAttack,
        config_class=MAttackConfig,
    ))

    AttackRegistry.register(AttackSpec(
        name="oattack",
        category="physical",
        attack_class=OAttack,
        config_class=OAttackConfig,
    ))

    AttackRegistry.register(AttackSpec(
        name="mpcattack",
        category="physical",
        attack_class=MPCAttack,
        config_class=MPCAttackConfig,
    ))

    AttackRegistry.register(AttackSpec(
        name="coa",
        category="physical",
        attack_class=COAAttack,
        config_class=COAAttackConfig,
    ))

    AttackRegistry.register(AttackSpec(
        name="advdiffvlm",
        category="diffusion",
        attack_class=AdvDiffVLMAttack,
        config_class=AdvDiffVLMConfig,
    ))

    from .advedm.advedm_attack import (
        ADVEDMAttack,
        ADVEDMConfig,
        ADVEDMRAttack,
        ADVEDMRConfig,
    )
    AttackRegistry.register(AttackSpec(
        name="advedm",
        category="visual",
        attack_class=ADVEDMAttack,
        config_class=ADVEDMConfig,
    ))

    AttackRegistry.register(AttackSpec(
        name="advedm_r",
        category="visual",
        attack_class=ADVEDMRAttack,
        config_class=ADVEDMRConfig,
    ))

    from .figstep.figstep_attack import FigStepAttack
    from .figstep.config import FigStepConfig

    AttackRegistry.register(AttackSpec(
        name="figstep",
        category="typographic",
        attack_class=FigStepAttack,
        config_class=FigStepConfig,
    ))

    from .vattack.vattack_attack import VAttack, VAttackConfig
    AttackRegistry.register(AttackSpec(
        name="vattack",
        category="physical",
        attack_class=VAttack,
        config_class=VAttackConfig,
    ))

    from .promptinject.promptinject_attack import PromptInjectAttack
    from .promptinject.config import PromptInjectConfig

    AttackRegistry.register(AttackSpec(
        name="promptinject",
        category="text",
        attack_class=PromptInjectAttack,
        config_class=PromptInjectConfig,
    ))

    from .anyattack.anyattack_attack import AnyAttack, AnyAttackConfig

    AttackRegistry.register(AttackSpec(
        name="anyattack",
        category="physical",
        attack_class=AnyAttack,
        config_class=AnyAttackConfig,
    ))

    from .paattack.paattack_attack import PAAttack, PAAttackConfig

    AttackRegistry.register(AttackSpec(
        name="paattack",
        category="visual",
        attack_class=PAAttack,
        config_class=PAAttackConfig,
    ))

    from .imagemix.imagemix_attack import ImageMixAttack, ImageMixConfig

    AttackRegistry.register(AttackSpec(
        name="imagemix",
        category="physical",
        attack_class=ImageMixAttack,
        config_class=ImageMixConfig,
    ))
