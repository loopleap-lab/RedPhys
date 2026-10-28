"""ImageMix: mixup / cutmix / resizemix baseline."""

import random
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from ..base_attack import AttackConfig, AttackResult, BaseAttack, open_by_stem, pixel_linf
from ...data import Sample


@dataclass
class ImageMixConfig(AttackConfig):
    alpha: float = 0.3            # mixup: target weight; else patch side ratio
    mix_type: str = "mixup"       # "mixup", "cutmix" or "resizemix" (JOOD)
    target_images_dir: Optional[str] = None  # flat dir with {sample_id}.{jpg,png,jpeg} targets

    def __post_init__(self):
        super().__post_init__()
        if self.mix_type not in ("mixup", "cutmix", "resizemix"):
            raise ValueError(f"mix_type must be 'mixup', 'cutmix' or 'resizemix', got {self.mix_type!r}")
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")


class ImageMixAttack(BaseAttack):

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
        clean = sample.images[0]
        target = open_by_stem(self.config.target_images_dir, sample.id)

        if self.config.mix_type == "mixup":
            adv = self._mixup(clean, target)
        else:
            adv = self._patch(clean, target, random.Random(f"{self.config.seed}-{sample.id}"),  # per sample
                              resize=self.config.mix_type == "resizemix")

        return AttackResult(adversarial_sample=adv, perturbation_norm=pixel_linf(adv, clean))
