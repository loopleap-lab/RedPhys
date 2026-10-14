"""V-Attack: text-guided adversarial attack on CLIP Value features."""

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torchvision.transforms as transforms

from ..base_attack import AttackConfig, AttackResult, BaseAttack, pixel_linf, to_pil, to_tensor
from ...data.base_dataset import Sample


@dataclass
class VAttackConfig(AttackConfig):
    epsilon: float = 16.0
    max_iterations: int = 200  # paper
    alpha: float = 0.75
    backbone: List[str] = field(default_factory=lambda: ["B16", "B32", "Laion"])
    input_res: int = 224
    crop_scale: Tuple[float, float] = (0.75, 1.0)
    crop_ratio: Tuple[float, float] = (3 / 4, 4 / 3)  # x image W/H
    use_source_crop: bool = True
    vattack: bool = True
    enhance: bool = True
    both: bool = False
    vision_attack: bool = False
    target_text_flag: bool = True
    source_text: str = ""
    target_text: str = ""


class VAttack(BaseAttack):
    def __init__(self, config: VAttackConfig):
        super().__init__(config)
        self.config: VAttackConfig = config
        self._ensemble_extractor = None
        self._ensemble_loss = None
        self._models = None

    def _initialize_models(self) -> None:
        if self._ensemble_extractor is not None:
            return

        from .core.surrogates.FeatureExtractors import (
            ClipB16FeatureExtractor,
            ClipB32FeatureExtractor,
            ClipLaionFeatureExtractor,
            EnsembleFeatureExtractor,
            EnsembleFeatureLoss,
        )

        class_map = {
            "B16": ClipB16FeatureExtractor,
            "B32": ClipB32FeatureExtractor,
            "Laion": ClipLaionFeatureExtractor,
        }

        models = []
        for name in self.config.backbone:
            cls = class_map[name]
            model = cls().eval().to(self.config.device).requires_grad_(False)
            models.append(model)

        self._models = models
        self._ensemble_extractor = EnsembleFeatureExtractor(models)
        self._ensemble_loss = EnsembleFeatureLoss(models)

    def _dict_to_list(self, features_dict):
        return [features_dict[i] for i in range(len(features_dict))]

    def _resolve_texts(self, sample: Sample) -> tuple[str, str]:
        """Get source/target text from config or per-sample metadata."""
        target = self.config.target_text
        source = self.config.source_text
        if not target and sample.metadata.get("attack_target_text"):
            target = sample.metadata["attack_target_text"]
        if not source and sample.metadata.get("attack_source_text"):
            source = sample.metadata["attack_source_text"]
        if not target or not source:
            raise ValueError(
                f"V-Attack needs both texts: pass labels (or source_text/target_text in the YAML) or provide "
                f"metadata['attack_source_text'/'attack_target_text'] on sample '{sample.id}'."
            )
        return source, target

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        cfg = self.config
        device = cfg.device

        image_org = to_tensor(sample.images[0], 255).to(device)

        if cfg.use_source_crop:
            a = image_org.shape[-1] / image_org.shape[-2]  # W/H-scaled
            source_crop = transforms.RandomResizedCrop(cfg.input_res, scale=cfg.crop_scale, ratio=(a * cfg.crop_ratio[0], a * cfg.crop_ratio[1]))
        else:
            source_crop = torch.nn.Identity()

        source_text, target_text = self._resolve_texts(sample)

        source_text_enc = self._dict_to_list(
            self._ensemble_extractor.tforward([source_text])
        )
        target_text_enc = self._dict_to_list(
            self._ensemble_extractor.tforward([target_text])
        )

        from .core.attacks import pgd_attack
        adv_tensor = pgd_attack(
            image_org=image_org,
            ensemble_extractor=self._ensemble_extractor,
            ensemble_loss=self._ensemble_loss,
            source_crop=source_crop,
            source_text=source_text_enc,
            target_text=target_text_enc,
            steps=cfg.max_iterations,
            epsilon=cfg.epsilon,
            alpha=cfg.alpha,
            vattack=cfg.vattack,
            enhance=cfg.enhance,
            both=cfg.both,
            vision_attack=cfg.vision_attack,
            target_text_flag=cfg.target_text_flag,
            device=device,
        )

        adv_pil = to_pil(adv_tensor)

        return AttackResult(
            adversarial_sample=adv_pil,
            perturbation_norm=pixel_linf(adv_pil, to_pil(image_org, 255)),
            metadata={
                "source_text": source_text,
                "target_text": target_text,
                "steps": cfg.max_iterations,
                "epsilon": cfg.epsilon,
            },
        )
