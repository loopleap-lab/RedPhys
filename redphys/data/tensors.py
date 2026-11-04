"""PIL <-> tensor and precision (kept out of redphys.data's import: torch is heavy)."""

import numpy as np
import torch
from PIL import Image


def to_tensor(img: Image.Image, scale: float = 1.0) -> torch.Tensor:
    """Image -> [1,3,H,W] in [0, scale]."""
    return torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).contiguous().unsqueeze(0).float() / (255 / scale)


def to_pil(x: torch.Tensor, scale: float = 1.0) -> Image.Image:
    """Tensor in [0, scale] -> uint8 RGB; raises on NaN."""
    x = x.detach().float().cpu().squeeze(0) / scale
    if not torch.isfinite(x).all():
        raise FloatingPointError("non-finite adversarial image")
    return Image.fromarray(x.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy())


def with_precision(obj, method: str) -> None:
    """Check obj.config.precision; bf16 on a CUDA device runs obj.<method> under bf16 autocast."""
    c = obj.config
    if c.precision not in ("fp32", "bf16"):
        raise ValueError(f"precision must be 'fp32' or 'bf16', got {c.precision!r}")
    if c.precision == "bf16" and str(c.device).startswith("cuda"):
        setattr(obj, method, torch.autocast("cuda", dtype=torch.bfloat16)(getattr(obj, method)))
