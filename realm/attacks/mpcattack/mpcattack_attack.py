"""MPCAttack: multi-paradigm collaborative adversarial attack (CVPR 2026, arXiv 2603.04846)."""

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from ..base_attack import AttackConfig, AttackResult, BaseAttack, pixel_linf, to_pil, to_tensor
from ...data.base_dataset import Sample


@dataclass
class MPCAttackConfig(AttackConfig):
    epsilon: float = 16.0
    max_iterations: int = 300
    alpha: float = 1.0
    momentum: float = 0.9
    clip_backbones: List[str] = field(default_factory=lambda: [
        "openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32", "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"])
    internvl_model: str = "OpenGVLab/InternVL3-1B-hf"
    dino_model: str = "facebook/dinov2-base"
    describe_prompt: str = "Describe this image in one concise sentence, no longer than 20 words."
    describe_max_tokens: int = 50
    lam: float = 0.6  # image vs text weight in the CLIP feature
    tau: float = 0.2
    omega: float = 2.0
    input_res: int = 224
    crop_scale: Tuple[float, float] = (0.5, 0.9)
    crop_ratio: Tuple[float, float] = (3 / 4, 4 / 3)  # x image W/H
    source_images_dir: Optional[str] = None
    target_images_dir: Optional[str] = None


class MPCAttack(BaseAttack):
    def __init__(self, config: MPCAttackConfig):
        super().__init__(config)
        self.config: MPCAttackConfig = config
        self._models = None

    def _load_models(self):
        if self._models is None:
            from .core import CLIPSurrogate, DINOv2Surrogate, InternVLSurrogate
            c = self.config
            clips = [CLIPSurrogate(m).to(c.device) for m in c.clip_backbones]
            internvl = InternVLSurrogate(c.internvl_model, c.describe_prompt, c.describe_max_tokens).to(c.device)
            dino = DINOv2Surrogate(c.dino_model).to(c.device)
            for m in (*clips, internvl, dino):
                m.requires_grad_(False)
            self._models = clips, internvl, dino

    def _target_image(self, sid: str) -> Image.Image:
        for ext in (".jpg", ".jpeg", ".png"):
            p = Path(self.config.target_images_dir) / f"{sid}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"No target image for '{sid}' in {self.config.target_images_dir}")

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        c = self.config
        if not c.target_images_dir:
            raise ValueError("target_images_dir must be set in config")
        self._load_models()
        clips, internvl, dino = self._models
        from .core import features, mpcattack

        src_pil = sample.images[0].convert("RGB")
        tgt_pil = T.Compose([T.Resize(c.input_res, interpolation=T.InterpolationMode.BICUBIC),
                             T.CenterCrop(c.input_res)])(self._target_image(str(sample.id)))
        image, target = to_tensor(src_pil, 255).to(c.device), to_tensor(tgt_pil, 255).to(c.device)
        src_text, tgt_text = internvl.describe(src_pil), internvl.describe(tgt_pil)
        extract = partial(features, clips, internvl, dino)
        with torch.no_grad():
            z_src = extract(image, [m.text(src_text) for m in clips], c.lam)
            z_tgt = extract(target, [m.text(tgt_text) for m in clips], c.lam)

        aspect = image.shape[-1] / image.shape[-2]
        source_crop = T.RandomResizedCrop(c.input_res, scale=c.crop_scale,
                                          ratio=(aspect * c.crop_ratio[0], aspect * c.crop_ratio[1]))
        adv = mpcattack(image, z_src, z_tgt, extract, source_crop, c.max_iterations, c.epsilon, c.alpha,
                        c.momentum, c.tau, c.omega, kwargs.get("sample_idx") or 0)
        adv_pil = to_pil(adv)
        return AttackResult(
            adversarial_sample=adv_pil,
            perturbation_norm=pixel_linf(adv_pil, src_pil),
            metadata={"epsilon": c.epsilon, "max_iterations": c.max_iterations,
                      "source_description": src_text, "target_description": tgt_text},
        )
