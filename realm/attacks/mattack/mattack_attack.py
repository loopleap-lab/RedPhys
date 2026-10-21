"""M-Attack: cosine similarity over a CLIP ensemble."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..base_attack import BaseAttack, AttackConfig, AttackResult, open_by_stem, load_surrogates, pixel_linf, random_crops, resize_center_crop, to_pil, to_tensor
from ...data import Sample


@dataclass
class MAttackConfig(AttackConfig):
    epsilon: float = 16.0
    max_iterations: int = 300
    alpha: float = 1.0

    attack_method: str = "fgsm"         # "fgsm" (paper default), "mifgsm", "pgd" (Adam)

    backbone: List[str] = field(
        default_factory=lambda: ["B16", "B32", "Laion"]
    )

    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: Tuple[float, float] = (0.5, 0.9)
    crop_ratio: Tuple[float, float] = (3 / 4, 4 / 3)  # x image W/H
    momentum: float = 0.9  # MI-FGSM

    target_images_dir: Optional[str] = None

    input_res: int = 224

    def __post_init__(self):
        super().__post_init__()
        if self.attack_method not in ("fgsm", "mifgsm", "pgd"):
            raise ValueError(f"attack_method must be 'fgsm', 'mifgsm' or 'pgd', got {self.attack_method!r}")


class MAttack(BaseAttack):
    config: MAttackConfig

    def __init__(self, config: MAttackConfig):
        super().__init__(config)
        self._ensemble_extractor = None
        self._ensemble_loss = None

    def _initialize_models(self):
        if self._ensemble_extractor is not None:
            return

        print("Loading CLIP ensemble models...")
        from .surrogates import (ClipB16FeatureExtractor, ClipB32FeatureExtractor, ClipL336FeatureExtractor, ClipLaionFeatureExtractor,
                                 EnsembleFeatureExtractor, EnsembleFeatureLoss)
        models = load_surrogates({"B16": ClipB16FeatureExtractor, "B32": ClipB32FeatureExtractor, "L336": ClipL336FeatureExtractor,
                                  "Laion": ClipLaionFeatureExtractor}, self.config.backbone, self.config.device)
        self._ensemble_extractor = EnsembleFeatureExtractor(models)
        self._ensemble_loss = EnsembleFeatureLoss(models)
        print("✓ Ensemble ready\n")

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()
        sample_idx = kwargs.get("sample_idx", 0)
        clean_image = to_tensor(sample.images[0], 255).to(self.config.device)
        target_image = to_tensor(resize_center_crop(open_by_stem(self.config.target_images_dir, str(sample.id)), self.config.input_res), 255).to(self.config.device)

        source_crop, target_crop = random_crops(self.config, clean_image.shape[-1] / clean_image.shape[-2])

        from .core import attacks
        attack_fn = getattr(attacks, f"{self.config.attack_method}_attack")  # fgsm, mifgsm or pgd

        adv_image = attack_fn(
            image_tensor=clean_image,
            tgt_tensor=target_image,
            ensemble_extractor=self._ensemble_extractor,
            ensemble_loss=self._ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            img_index=sample_idx,
            num_iters=self.config.max_iterations,
            epsilon=self.config.epsilon,
            alpha=self.config.alpha,
            **({"momentum_decay": self.config.momentum} if self.config.attack_method == "mifgsm" else {}),
        )

        adv_pil = to_pil(adv_image)

        return AttackResult(adversarial_sample=adv_pil, perturbation_norm=pixel_linf(adv_pil, sample.images[0]))
