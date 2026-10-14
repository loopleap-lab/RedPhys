"""Base classes for defense methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict
import torch
import random
import numpy as np


@dataclass
class DefenseConfig:
    device: str = "cuda"
    seed: int = 42
    precision: str = "bf16"  # "bf16" or "fp32"


@dataclass
class DefenseResult:
    cleaned_sample: Any
    original_image_path: str
    detection_confidence: float = 0.0
    regions_removed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDefense(ABC):
    def __init__(self, config: DefenseConfig):
        self.config = config
        self._set_seed(config.seed)
        if config.precision not in ("fp32", "bf16"):
            raise ValueError(f"precision must be 'fp32' or 'bf16', got {config.precision!r}")
        if config.precision == "bf16":
            self.clean = torch.autocast("cuda", dtype=torch.bfloat16)(self.clean)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @abstractmethod
    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        pass
