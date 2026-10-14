"""Adversarial attacks."""

__version__ = "0.1.0"

from .base_attack import (
    AttackConfig,
    AttackResult,
    BaseAttack,
)

from .physpatch.physpatch_attack import (
    PhysPatchConfig,
    PhysPatchAttack,
)
from .foa.foa_attack import (
    FOAAttackConfig,
    FOAAttack,
)
from .mattack.mattack_attack import (
    MAttackConfig,
    MAttack,
)
from .oattack.oattack_attack import (
    OAttackConfig,
    OAttack,
)
from .mpcattack.mpcattack_attack import (
    MPCAttackConfig,
    MPCAttack,
)
from .coa.coa_attack import (
    COAAttackConfig,
    COAAttack,
)
from .advdiffvlm.advdiffvlm_attack import (
    AdvDiffVLMConfig,
    AdvDiffVLMAttack,
)
from .advedm.advedm_attack import (
    ADVEDMConfig,
    ADVEDMAttack,
    ADVEDMRConfig,
    ADVEDMRAttack,
)

from .vattack.vattack_attack import (
    VAttackConfig,
    VAttack,
)

from .anyattack.anyattack_attack import (
    AnyAttackConfig,
    AnyAttack,
)

from .paattack.paattack_attack import (
    PAAttackConfig,
    PAAttack,
)

from .figstep.figstep_attack import (
    FigStepConfig,
    FigStepAttack,
)

from .promptinject.promptinject_attack import (
    PromptInjectConfig,
    PromptInjectAttack,
)

from .imagemix.imagemix_attack import (
    ImageMixConfig,
    ImageMixAttack,
)

from .registry import AttackRegistry, AttackSpec, register_all_attacks

register_all_attacks()

__all__ = [
    "AttackConfig",
    "AttackResult",
    "BaseAttack",
    "PhysPatchConfig",
    "PhysPatchAttack",
    "FOAAttackConfig",
    "FOAAttack",
    "MAttackConfig",
    "MAttack",
    "OAttackConfig",
    "OAttack",
    "MPCAttackConfig",
    "MPCAttack",
    "COAAttackConfig",
    "COAAttack",
    "VAttackConfig",
    "VAttack",
    "AnyAttackConfig",
    "AnyAttack",
    "AdvDiffVLMConfig",
    "AdvDiffVLMAttack",
    "ADVEDMConfig",
    "ADVEDMAttack",
    "ADVEDMRConfig",
    "ADVEDMRAttack",
    "PAAttackConfig",
    "PAAttack",
    "FigStepConfig",
    "FigStepAttack",
    "PromptInjectConfig",
    "PromptInjectAttack",
    "ImageMixConfig",
    "ImageMixAttack",
    "AttackRegistry",
    "AttackSpec",
]
