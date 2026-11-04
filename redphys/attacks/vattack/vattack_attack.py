"""V-Attack: text-guided attack on CLIP value features."""

from dataclasses import dataclass, field
from typing import ClassVar, List, Tuple

import torch

from ..base_attack import AttackConfig, AttackResult, BaseAttack, aspect_crop, load_surrogates, pixel_linf, target_text, to_pil, to_tensor
from ...data import Sample


@dataclass
class VAttackConfig(AttackConfig):
    needs_source_text: ClassVar[bool] = True
    epsilon: float = 16.0
    max_iterations: int = 200  # paper
    alpha: float = 0.75
    backbone: List[str] = field(default_factory=lambda: ["B16", "B32", "Laion"])
    input_res: int = 224
    resize_source: bool = False  # as upstream: the source is resized and center-cropped to input_res; the output has that size
    crop_scale: Tuple[float, float] = (0.75, 1.0)
    crop_ratio: Tuple[float, float] = (3 / 4, 4 / 3)  # x image W/H
    use_source_crop: bool = True
    vattack: bool = True
    enhance: bool = True
    both: bool = False
    target_text_flag: bool = True
    source_text: str = ""
    target_text: str = ""


class VAttack(BaseAttack):
    config: VAttackConfig

    def __init__(self, config: VAttackConfig):
        super().__init__(config)
        self._ensemble_extractor = None
        self._ensemble_loss = None

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

        models = load_surrogates(class_map, self.config.backbone, self.config.device)
        self._ensemble_extractor = EnsembleFeatureExtractor(models)
        self._ensemble_loss = EnsembleFeatureLoss(models)

    def _resolve_texts(self, sample: Sample) -> tuple[str, str]:
        """Source/target text from config or sample metadata."""
        target = target_text(self.config, sample)
        source = self.config.source_text or sample.metadata.get("attack_source_text")
        if not source:  # target_text() raises on a missing target
            raise ValueError(
                f"V-Attack needs a source text for sample '{sample.id}': labels= or source_text in the YAML."
            )
        return source, target

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        cfg = self.config
        device = cfg.device

        image_org = to_tensor(sample.images[0], 255).to(device)

        source_crop = aspect_crop(cfg.input_res, cfg.crop_scale, cfg.crop_ratio, image_org.shape[-1] / image_org.shape[-2]) \
            if cfg.use_source_crop else torch.nn.Identity()

        source_text, target_text = self._resolve_texts(sample)

        from .core.attacks import _dict_to_list, pgd_attack
        source_text_enc = _dict_to_list(self._ensemble_extractor.tforward([source_text]))
        target_text_enc = _dict_to_list(self._ensemble_extractor.tforward([target_text]))

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
            target_text_flag=cfg.target_text_flag,
            device=device,
        )

        adv_pil = to_pil(adv_tensor)

        return AttackResult(adversarial_sample=adv_pil, perturbation_norm=pixel_linf(adv_pil, sample.images[0]))
