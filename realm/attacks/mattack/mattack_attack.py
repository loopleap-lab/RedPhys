"""M-Attack: adversarial perturbations via cosine similarity over a CLIP ensemble."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

from ..base_attack import BaseAttack, AttackConfig, AttackResult, pixel_linf, to_pil, to_tensor
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

    source_images_dir: Optional[str] = None
    target_images_dir: Optional[str] = None

    input_res: int = 224


class MAttack(BaseAttack):
    def __init__(self, config: MAttackConfig):
        super().__init__(config)
        self.config: MAttackConfig = config

        self._models_initialized = False
        self._ensemble_extractor = None
        self._ensemble_loss = None

    def _initialize_models(self):
        if self._models_initialized:
            return

        print("Loading CLIP ensemble models...")

        from .surrogates import (
            ClipB16FeatureExtractor,
            ClipB32FeatureExtractor,
            ClipL336FeatureExtractor,
            ClipLaionFeatureExtractor,
            EnsembleFeatureExtractor,
            EnsembleFeatureLoss,
        )

        BACKBONE_MAP = {
            "B16": ClipB16FeatureExtractor,
            "B32": ClipB32FeatureExtractor,
            "L336": ClipL336FeatureExtractor,
            "Laion": ClipLaionFeatureExtractor,
        }

        models = []
        for backbone in self.config.backbone:
            if backbone not in BACKBONE_MAP:
                raise ValueError(f"Unknown backbone: {backbone}")

            model_class = BACKBONE_MAP[backbone]
            model = model_class().eval().to(self.config.device)
            model.requires_grad_(False)
            models.append(model)
            print(f"  ✓ Loaded {backbone}")

        self._ensemble_extractor = EnsembleFeatureExtractor(models)
        self._ensemble_loss = EnsembleFeatureLoss(models)

        self._models_initialized = True
        print("✓ Ensemble ready\n")

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """PIL -> unnormalized [0, 255] tensor."""
        image = transforms.Resize(
            self.config.input_res,
            interpolation=transforms.InterpolationMode.BICUBIC
        )(image)
        image = transforms.CenterCrop(self.config.input_res)(image)
        return to_tensor(image, 255).to(self.config.device)

    def _load_image_from_dir(self, images_dir: str, sample_id: str) -> Image.Image:
        """Load image from a flat directory by sample id stem (e.g. '00', '02')."""
        img_dir = Path(images_dir)
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / f"{sample_id}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"No image found for id '{sample_id}' in {img_dir}")

    def _create_crops(self, aspect: float = 1.0):
        if self.config.use_source_crop:
            source_crop = transforms.RandomResizedCrop(  # W/H-scaled
                size=self.config.input_res,
                scale=self.config.crop_scale,
                ratio=(aspect * self.config.crop_ratio[0], aspect * self.config.crop_ratio[1]),
            )
        else:
            source_crop = nn.Identity()

        if self.config.use_target_crop:
            target_crop = transforms.RandomResizedCrop(
                size=self.config.input_res,
                scale=self.config.crop_scale
            )
        else:
            target_crop = nn.Identity()

        return source_crop, target_crop

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        if not self.config.source_images_dir:
            raise ValueError("source_images_dir must be set in config")
        if not self.config.target_images_dir:
            raise ValueError("target_images_dir must be set in config")
        sample_idx = kwargs.get("sample_idx", 0)
        clean_image = to_tensor(sample.images[0], 255).to(self.config.device)
        target_image = self._prepare_image(self._load_image_from_dir(self.config.target_images_dir, str(sample.id)))

        source_crop, target_crop = self._create_crops(clean_image.shape[-1] / clean_image.shape[-2])

        from .core.attacks import fgsm_attack, mifgsm_attack, pgd_attack

        attack_fn_map = {
            "fgsm": fgsm_attack,
            "mifgsm": mifgsm_attack,
            "pgd": pgd_attack,
        }
        attack_fn = attack_fn_map[self.config.attack_method]

        adv_image = attack_fn(
            image_tensor=clean_image,
            tgt_tensor=target_image,
            ensemble_extractor=self._ensemble_extractor,
            ensemble_loss=self._ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            img_index=sample_idx if sample_idx is not None else 0,
            num_iters=self.config.max_iterations,
            epsilon=self.config.epsilon,
            alpha=self.config.alpha,
            device=self.config.device,
            **({"momentum_decay": self.config.momentum} if self.config.attack_method == "mifgsm" else {}),
        )

        adv_pil = to_pil(adv_image)

        return AttackResult(
            adversarial_sample=adv_pil,
            perturbation_norm=pixel_linf(adv_pil, to_pil(clean_image, 255)),
            metadata={
                "attack_method": self.config.attack_method,
                "backbone": self.config.backbone,
                "epsilon": self.config.epsilon,
                "max_iterations": self.config.max_iterations,
                "alpha": self.config.alpha,
            }
        )
