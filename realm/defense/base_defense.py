"""Base classes for defense methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from ..data.tensors import with_precision


@dataclass
class DefenseConfig:
    device: str = "cuda"
    seed: int = 42  # run.py seeds each image from (seed, id); OSCP draws its noise from this fixed seed, as in the paper
    precision: str = "bf16"  # "bf16" or "fp32"


@dataclass
class DefenseResult:
    cleaned_sample: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDefense(ABC):
    def __init__(self, config: DefenseConfig):
        self.config = config
        with_precision(self, "clean")

    @abstractmethod
    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        pass
