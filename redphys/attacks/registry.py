"""Attack registry."""

from ..registry import Registry

AttackRegistry = Registry("attack")


def register_all_attacks():
    from .advdiffvlm.advdiffvlm_attack import AdvDiffVLMAttack, AdvDiffVLMConfig
    from .advedm.advedm_attack import ADVEDMAttack, ADVEDMConfig, ADVEDMRAttack, ADVEDMRConfig
    from .anyattack.anyattack_attack import AnyAttack, AnyAttackConfig
    from .coa.coa_attack import COAAttack, COAAttackConfig
    from .figstep.config import FigStepConfig
    from .figstep.figstep_attack import FigStepAttack
    from .foa.foa_attack import FOAAttack, FOAAttackConfig
    from .imagemix.imagemix_attack import ImageMixAttack, ImageMixConfig
    from .leadingq.config import LeadingQConfig
    from .leadingq.leadingq_attack import LeadingQAttack
    from .mattack.mattack_attack import MAttack, MAttackConfig
    from .mpcattack.mpcattack_attack import MPCAttack, MPCAttackConfig
    from .oattack.oattack_attack import OAttack, OAttackConfig
    from .paattack.paattack_attack import PAAttack, PAAttackConfig
    from .physpatch.physpatch_attack import PhysPatchAttack, PhysPatchConfig
    from .promptinject.config import PromptInjectConfig
    from .promptinject.promptinject_attack import PromptInjectAttack
    from .scenetap.config import SceneTAPConfig
    from .scenetap.scenetap_attack import SceneTAPAttack
    from .vattack.vattack_attack import VAttack, VAttackConfig

    for name, attack_class, config_class in [
        ("physpatch", PhysPatchAttack, PhysPatchConfig),
        ("foa", FOAAttack, FOAAttackConfig),
        ("mattack", MAttack, MAttackConfig),
        ("oattack", OAttack, OAttackConfig),
        ("mpcattack", MPCAttack, MPCAttackConfig),
        ("coa", COAAttack, COAAttackConfig),
        ("advdiffvlm", AdvDiffVLMAttack, AdvDiffVLMConfig),
        ("advedm", ADVEDMAttack, ADVEDMConfig),
        ("advedm_r", ADVEDMRAttack, ADVEDMRConfig),
        ("figstep", FigStepAttack, FigStepConfig),
        ("vattack", VAttack, VAttackConfig),
        ("promptinject", PromptInjectAttack, PromptInjectConfig),
        ("leadingq", LeadingQAttack, LeadingQConfig),
        ("scenetap", SceneTAPAttack, SceneTAPConfig),
        ("anyattack", AnyAttack, AnyAttackConfig),
        ("paattack", PAAttack, PAAttackConfig),
        ("imagemix", ImageMixAttack, ImageMixConfig),
    ]:
        AttackRegistry.register(name, attack_class, config_class)
