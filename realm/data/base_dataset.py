"""Sample and BaseDataset."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional
from PIL import Image, ImageOps


def load_rgb(path) -> Image.Image:
    """RGB, upright per EXIF; 16-bit gray scaled to 8-bit."""
    img = ImageOps.exif_transpose(Image.open(path))
    return (img.point(lambda v: v / 257, "L") if img.mode.startswith("I") else img).convert("RGB")


@dataclass
class Sample:
    id: str
    images: List[Image.Image]
    question: str
    ground_truth: str
    task_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):

    def __init__(self, data_root: str, max_samples: Optional[int] = None):
        self.data_root = data_root
        self.max_samples = max_samples
        self.entries: List[Any] = []
        self.load_data()
        if self.max_samples is not None:
            self.entries = self.entries[: self.max_samples]

    @abstractmethod
    def load_data(self) -> None:
        """Fill self.entries, one record per sample."""

    @abstractmethod
    def get_sample(self, idx: int) -> Sample:
        """Sample for idx (loads its images)."""

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Sample:
        return self.get_sample(idx)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self.get_sample(i)
