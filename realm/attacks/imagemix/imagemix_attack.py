"""ImageMix: alpha-blend / cutmix baseline."""

import random
from pathlib import Path

from PIL import Image

from ..base_attack import AttackResult, BaseAttack, pixel_linf
from ...data.base_dataset import Sample
from .config import ImageMixConfig


class ImageMixAttack(BaseAttack):

    def _load_target(self, sample_id: str) -> Image.Image:
        target_dir = Path(self.config.target_images_dir)
        for ext in (".jpg", ".png", ".jpeg"):
            p = target_dir / f"{sample_id}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(
            f"No target image for '{sample_id}' in {target_dir}"
        )

    def _mixup(self, clean: Image.Image, target: Image.Image) -> Image.Image:
        target = target.resize(clean.size, Image.LANCZOS)
        return Image.blend(clean, target, self.config.alpha)

    def _patch(self, clean: Image.Image, target: Image.Image, rng: random.Random, resize: bool) -> Image.Image:
        """CutMix / ResizeMix (JOOD): alpha-sided patch at a random position."""
        w, h = clean.size
        pw, ph = max(1, int(w * self.config.alpha)), max(1, int(h * self.config.alpha))
        x, y = rng.randint(0, w - pw), rng.randint(0, h - ph)
        if resize:  # ResizeMix: the whole target, shrunk
            patch = target.resize((pw, ph), Image.LANCZOS)
        else:  # CutMix: the target's region at the same position
            patch = target.resize(clean.size, Image.LANCZOS).crop((x, y, x + pw, y + ph))
        result = clean.copy()
        result.paste(patch, (x, y))
        return result

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        if not self.config.target_images_dir:
            raise ValueError("ImageMix needs target_images_dir")
        clean = sample.images[0]
        target = self._load_target(sample.id)

        if self.config.mix_type == "mixup":
            adv = self._mixup(clean, target)
        else:
            adv = self._patch(clean, target, random.Random(f"{self.config.seed}-{sample.id}"),  # per sample
                              resize=self.config.mix_type == "resizemix")

        return AttackResult(
            adversarial_sample=adv,
            perturbation_norm=pixel_linf(adv, clean),
            metadata={
                "mix_type": self.config.mix_type,
                "alpha": self.config.alpha,
                "attack": "imagemix",
            },
        )
