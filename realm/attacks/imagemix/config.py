"""ImageMix attack configuration."""

from dataclasses import dataclass

from ..base_attack import AttackConfig


def resolve_cli_kwargs(attack_name, args, context):
    return {"target_images_dir": context["target_dir"]}


@dataclass
class ImageMixConfig(AttackConfig):
    alpha: float = 0.3            # mixup: target weight; else patch side ratio
    mix_type: str = "mixup"       # "mixup", "cutmix" or "resizemix" (JOOD)
    target_images_dir: str = ""   # flat dir with {sample_id}.{jpg,png,jpeg} targets

    def __post_init__(self):
        super().__post_init__()
        if self.mix_type not in ("mixup", "cutmix", "resizemix"):
            raise ValueError(f"mix_type must be 'mixup', 'cutmix' or 'resizemix', got {self.mix_type!r}")
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
