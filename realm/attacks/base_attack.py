"""Attack base class and image/tensor helpers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from ..data.base_dataset import Sample


def to_tensor(img: Image.Image, scale: float = 1.0) -> torch.Tensor:
    """Image -> [1,3,H,W] in [0, scale]."""
    return torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).contiguous().unsqueeze(0).float() / (255 / scale)


def to_pil(x: torch.Tensor, scale: float = 1.0) -> Image.Image:
    """Tensor in [0, scale] -> uint8 RGB; raises on NaN."""
    x = x.detach().float().cpu().squeeze(0) / scale
    if not torch.isfinite(x).all():
        raise FloatingPointError("non-finite adversarial image")
    return Image.fromarray(x.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy())


def pixel_linf(adv: Image.Image, clean: Image.Image) -> float:
    """L-inf of two images (0-255)."""
    return float(np.abs(np.asarray(adv.convert("RGB"), np.int16) - np.asarray(clean.convert("RGB"), np.int16)).max())


@dataclass
class AttackConfig:
    epsilon: float = 8.0  # L-inf, pixel units (0-255)
    seed: int = 42
    max_iterations: int = 10
    alpha: Optional[float] = None  # None: epsilon/10
    device: str = "cuda"
    surrogates: str = "default"  # see surrogates.py
    precision: str = "bf16"  # or "fp32"

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.epsilon / 10.0


@dataclass
class AttackResult:
    adversarial_sample: Any
    perturbation_norm: Optional[float]  # None: not a perturbation
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):

    def __init__(self, config: AttackConfig):
        self.config = config

        import random
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        if config.precision not in ("fp32", "bf16"):
            raise ValueError(f"precision must be 'fp32' or 'bf16', got {config.precision!r}")
        if config.precision == "bf16":
            self.generate = torch.autocast("cuda", dtype=torch.bfloat16)(self.generate)

    @abstractmethod
    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        """Adversarial example for a sample (kwargs: sample_idx)."""
