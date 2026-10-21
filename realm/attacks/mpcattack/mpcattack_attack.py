"""MPCAttack: multi-paradigm collaborative attack (CVPR 2026, arXiv 2603.04846)."""

from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import torch

from ..base_attack import AttackConfig, AttackResult, BaseAttack, aspect_crop, find_by_stem, pixel_linf, resize_center_crop, to_pil, to_tensor
from ...data import Sample, load_rgb


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
    target_images_dir: Optional[str] = None


class MPCAttack(BaseAttack):
    config: MPCAttackConfig

    def __init__(self, config: MPCAttackConfig):
        super().__init__(config)
        self._models = None
        self._target_texts = {}  # resolved target file -> description: one shared target is described once

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

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        c = self.config
        self._load_models()
        clips, internvl, dino = self._models
        from .core import features, mpcattack

        src_pil = sample.images[0]
        tgt_path = find_by_stem(c.target_images_dir, str(sample.id), required=True)
        tgt_pil = resize_center_crop(load_rgb(tgt_path), c.input_res)
        image, target = to_tensor(src_pil, 255).to(c.device), to_tensor(tgt_pil, 255).to(c.device)
        src_text = internvl.describe(src_pil)
        if (key := str(tgt_path.resolve())) not in self._target_texts:
            self._target_texts[key] = internvl.describe(tgt_pil)
        tgt_text = self._target_texts[key]
        extract = partial(features, clips, internvl, dino)
        with torch.no_grad():
            z_src = extract(image, [m.text(src_text) for m in clips], c.lam)
            z_tgt = extract(target, [m.text(tgt_text) for m in clips], c.lam)

        source_crop = aspect_crop(c.input_res, c.crop_scale, c.crop_ratio, image.shape[-1] / image.shape[-2])
        adv = mpcattack(image, z_src, z_tgt, extract, source_crop, c.max_iterations, c.epsilon, c.alpha,
                        c.momentum, c.tau, c.omega, kwargs.get("sample_idx", 0))
        adv_pil = to_pil(adv)
        return AttackResult(
            adversarial_sample=adv_pil,
            perturbation_norm=pixel_linf(adv_pil, src_pil),
            metadata={"source_description": src_text, "target_description": tgt_text},
        )
