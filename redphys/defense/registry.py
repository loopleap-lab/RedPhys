"""Defense registry."""

from ..registry import Registry

DefenseRegistry = Registry("defense")


def register_all_defenses():
    from .pad.pad_defense import PADDefense, PADDefenseConfig
    from .freqpure.freqpure_defense import FreqPureDefense, FreqPureDefenseConfig
    from .bluesuffix.bluesuffix_defense import BlueSuffixDefense, BlueSuffixDefenseConfig
    from .et3.et3_defense import ET3Defense, ET3DefenseConfig
    from .oscp.oscp_defense import OSCPDefense, OSCPDefenseConfig
    from .adpo import AdPODefense, AdPOConfig

    for name, defense_class, config_class in [
        ("pad", PADDefense, PADDefenseConfig),
        ("freqpure", FreqPureDefense, FreqPureDefenseConfig),
        ("bluesuffix", BlueSuffixDefense, BlueSuffixDefenseConfig),
        ("et3", ET3Defense, ET3DefenseConfig),
        ("oscp", OSCPDefense, OSCPDefenseConfig),
        ("adpo", AdPODefense, AdPOConfig),  # training-based: fits the victim's vision encoder
    ]:
        DefenseRegistry.register(name, defense_class, config_class)
