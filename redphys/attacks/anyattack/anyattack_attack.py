"""AnyAttack: one forward pass of a learned decoder."""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torchvision.transforms as transforms
from PIL import Image

from ..base_attack import AttackConfig, AttackResult, BaseAttack, open_by_stem, pixel_linf, to_pil, to_tensor
from ...data import Sample, weight

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


@dataclass
class AnyAttackConfig(AttackConfig):
    epsilon: float = 16.0
    decoder_checkpoint: str = "coco_cos"  # L_Cos: general tasks
    target_images_dir: Optional[str] = None


class AnyAttack(BaseAttack):
    config: AnyAttackConfig

    def __init__(self, config: AnyAttackConfig):
        super().__init__(config)
        self._clip_encoder = None
        self._decoder = None

    def _initialize_models(self) -> None:
        if self._decoder is not None:  # set last: a failed load retries
            return

        from .core.model import CLIPEncoder, Decoder

        device = self.config.device

        self._clip_encoder = CLIPEncoder(model_name="ViT-B/32").to(device).eval()

        ckpt_name = self.config.decoder_checkpoint
        if os.path.isfile(ckpt_name):
            ckpt_path = ckpt_name
        else:
            ckpt_path = os.path.join(_ASSETS_DIR, "checkpoints", f"{ckpt_name}.pt")

        checkpoint = torch.load(weight(ckpt_path, "anyattack"), map_location="cpu")

        state = {k.removeprefix("module."): v for k, v in checkpoint.get("decoder_state_dict", checkpoint).items()}  # DDP prefix
        self._decoder = Decoder(embed_dim=state["fc.0.weight"].shape[1])
        self._decoder.load_state_dict(state)
        self._decoder = self._decoder.to(device).eval()

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Squash to the decoder's size, as demo.py."""
        size = self._decoder.img_size
        return to_tensor(image.resize((size, size), Image.BILINEAR))

    def _load_target_image(self, sample: Sample) -> torch.Tensor:
        return self._prepare_image(open_by_stem(self.config.target_images_dir, sample.id))

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        device = self.config.device
        eps = self.config.epsilon / 255.0

        clean = sample.images[0]
        target_tensor = self._load_target_image(sample).to(device)

        with torch.no_grad():
            noise = self._decoder(self._clip_encoder.encode_img(target_tensor)).clamp(-eps, eps)
            noise = transforms.functional.resize(noise, [clean.height, clean.width],  # onto the native image
                                                 interpolation=transforms.InterpolationMode.BICUBIC, antialias=True).clamp(-eps, eps)
            adv_tensor = (to_tensor(clean).to(device) + noise).clamp(0, 1)

        adv_pil = to_pil(adv_tensor)
        return AttackResult(adversarial_sample=adv_pil, perturbation_norm=pixel_linf(adv_pil, clean))
