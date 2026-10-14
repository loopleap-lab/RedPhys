"""AnyAttack: one forward pass of a learned decoder."""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from ..base_attack import AttackConfig, AttackResult, BaseAttack, pixel_linf, to_pil, to_tensor
from ...data.base_dataset import Sample

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


@dataclass
class AnyAttackConfig(AttackConfig):
    epsilon: float = 16.0
    decoder_checkpoint: str = "coco_cos"  # L_Cos: general tasks
    target_images_dir: str = ""


class AnyAttack(BaseAttack):
    def __init__(self, config: AnyAttackConfig):
        super().__init__(config)
        self.config: AnyAttackConfig = config
        self._clip_encoder = None
        self._decoder = None

    def _initialize_models(self) -> None:
        if self._clip_encoder is not None:
            return

        from .core.model import CLIPEncoder, Decoder

        device = self.config.device

        self._clip_encoder = CLIPEncoder(model_name="ViT-B/32")
        self._clip_encoder = self._clip_encoder.to(device).eval()

        ckpt_name = self.config.decoder_checkpoint
        if os.path.isfile(ckpt_name):
            ckpt_path = ckpt_name
        else:
            ckpt_path = os.path.join(_ASSETS_DIR, "checkpoints", f"{ckpt_name}.pt")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Decoder checkpoint not found: {ckpt_path}\n"
                f"Available: {os.listdir(os.path.join(_ASSETS_DIR, 'checkpoints'))}"
            )

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        state = {k.removeprefix("module."): v for k, v in checkpoint.get("decoder_state_dict", checkpoint).items()}  # DDP prefix
        self._decoder = Decoder(embed_dim=state["fc.0.weight"].shape[1])
        self._decoder.load_state_dict(state)

        self._decoder = self._decoder.to(device).eval()

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Squash to the decoder's size, as demo.py."""
        size = self._decoder.img_size
        return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])(image.convert("RGB")).unsqueeze(0)

    def _load_target_image(self, sample: Sample) -> torch.Tensor:
        tgt_dir = self.config.target_images_dir
        if not tgt_dir:
            raise ValueError("target_images_dir not set in AnyAttackConfig")

        tgt_path = Path(tgt_dir) / f"{sample.id}.jpg"
        if not tgt_path.exists():
            tgt_path = Path(tgt_dir) / f"{sample.id}.png"
        if not tgt_path.exists():
            tgt_path = Path(tgt_dir) / f"{sample.id}.jpeg"
        if not tgt_path.exists():
            raise FileNotFoundError(f"Target image not found: {tgt_dir}/{sample.id}.*")

        return self._prepare_image(Image.open(tgt_path))

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        device = self.config.device
        eps = self.config.epsilon / 255.0

        clean = sample.images[0].convert("RGB")
        target_tensor = self._load_target_image(sample).to(device)

        with torch.no_grad():
            noise = self._decoder(self._clip_encoder.encode_img(target_tensor)).clamp(-eps, eps)
            noise = transforms.functional.resize(noise, [clean.height, clean.width],  # onto the native image
                                                 interpolation=transforms.InterpolationMode.BICUBIC, antialias=True).clamp(-eps, eps)
            adv_tensor = (to_tensor(clean).to(device) + noise).clamp(0, 1)

        adv_pil = to_pil(adv_tensor)
        return AttackResult(
            adversarial_sample=adv_pil,
            perturbation_norm=pixel_linf(adv_pil, clean),
            metadata={
                "epsilon": self.config.epsilon,
                "decoder_checkpoint": self.config.decoder_checkpoint,
            },
        )
