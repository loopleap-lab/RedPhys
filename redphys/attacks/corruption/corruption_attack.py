"""Non-adversarial corruptions at severity 1-5: blur, noise, JPEG and lighting as ImageNet-C (Hendrycks & Dietterich, ICLR 2019),
occlusion as Random Erasing (Zhong et al., AAAI 2020) with a fixed area."""

import io
import math
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from ..base_attack import AttackConfig, AttackResult, BaseAttack, pixel_linf
from ...data import Sample

KINDS = ("blur", "noise", "jpeg", "occlusion", "lighting")


@dataclass
class CorruptionConfig(AttackConfig):
    kind: str = "blur"  # one of KINDS
    severity: int = 3  # 1-5: which of the kind's levels
    blur: list = field(default_factory=lambda: [1, 2, 3, 4, 6])  # gaussian_blur sigma at 224 px, scaled with the short side
    noise: list = field(default_factory=lambda: [.08, .12, .18, .26, .38])  # gaussian_noise std, 0-1 scale
    jpeg: list = field(default_factory=lambda: [25, 18, 15, 10, 7])  # jpeg_compression quality
    occlusion: list = field(default_factory=lambda: [.05, .1, .2, .3, .4])  # gray rectangle, fraction of the image area
    lighting: list = field(default_factory=lambda: [.1, .2, .3, .4, .5])  # brightness: HSV value shift
    brighter: bool = False  # lighting: ImageNet-C's brightness; default darker (low light)

    def __post_init__(self):
        super().__post_init__()
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}, got {self.kind!r}")
        if bad := [k for k in KINDS if len(getattr(self, k)) != 5]:
            raise ValueError(f"{', '.join(bad)}: give five levels")
        if self.severity not in range(1, 6):
            raise ValueError(f"severity must be 1-5, got {self.severity}")


class CorruptionAttack(BaseAttack):
    """Untargeted, no optimization: a control for the adversarial attacks."""

    def __init__(self, config: CorruptionConfig):
        super().__init__(config)
        self.run_name = f"corruption-{config.kind}"  # the manifest's name: kinds stay apart in evaluate and report

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        clean, kind = sample.images[0].convert("RGB"), self.config.kind
        c, rng = getattr(self.config, kind)[self.config.severity - 1], random.Random(f"{self.config.seed}-{sample.id}")  # per sample
        x = np.asarray(clean, np.float32) / 255
        if kind == "blur":
            out = clean.filter(ImageFilter.GaussianBlur(c * min(clean.size) / 224))
        elif kind == "jpeg":
            buf = io.BytesIO()
            clean.save(buf, "JPEG", quality=c)
            out = Image.open(buf).convert("RGB")
        elif kind == "occlusion":
            w, h = clean.size
            r = math.exp(rng.uniform(math.log(0.3), math.log(1 / 0.3)))  # aspect ratio, Random Erasing's range
            pw = min(w, round(math.sqrt(c * w * h * r)))
            ph = min(h, round(c * w * h / pw))
            out = clean.copy()
            out.paste((128, 128, 128), (x0 := rng.randint(0, w - pw), y0 := rng.randint(0, h - ph), x0 + pw, y0 + ph))
        else:
            if kind == "noise":
                y = x + np.random.default_rng(rng.getrandbits(64)).normal(scale=c, size=x.shape)
            else:  # new HSV value, same hue and saturation: rgb scales by v'/v (black stays gray)
                v = x.max(2, keepdims=True)
                v2 = np.clip(v + (c if self.config.brighter else -c), 0, 1)
                y = np.where(v > 0, x * v2 / np.maximum(v, 1e-8), v2)
            out = Image.fromarray(np.round(np.clip(y, 0, 1) * 255).astype(np.uint8))
        return AttackResult(adversarial_sample=out, perturbation_norm=pixel_linf(out, clean),
                            metadata={"corruption": kind, "severity": self.config.severity})
