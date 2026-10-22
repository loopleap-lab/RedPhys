from dataclasses import dataclass
from typing import Optional, Tuple

from ..base_defense import DefenseConfig


@dataclass
class AdPOConfig(DefenseConfig):
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    adv_source: str = "realm"  # or "pgd"
    holdout_attacks: Tuple[str, ...] = ()
    test_fraction: float = 0.2
    max_pairs: Optional[int] = None
    prompt: Optional[str] = "What is the content of the image?"  # None: case question
    max_new_tokens: int = 256
    max_pixels: Optional[int] = None  # Qwen only
    beta: float = 0.1
    lam: float = 1.0
    aio_reduction: str = "sum"  # or "mean"
    lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 2
    batch_size: int = 128  # effective
    micro_batch: int = 1
    train_projector: bool = False
    pgd_eps: float = 2.0  # /255
    pgd_steps: int = 10
    pgd_alpha: float = 1.0  # /255
    server_url: Optional[str] = None

    def __post_init__(self):
        if self.adv_source not in ("realm", "pgd"):
            raise ValueError(f"adv_source must be 'realm' or 'pgd', got {self.adv_source!r}")
        if self.aio_reduction not in ("sum", "mean"):
            raise ValueError(f"aio_reduction must be 'sum' or 'mean', got {self.aio_reduction!r}")
