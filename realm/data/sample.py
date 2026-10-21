"""Sample and image loading."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from PIL import Image, ImageOps

DEFAULT_QUESTION = "What is the main object in this image?"


def load_rgb(path) -> Image.Image:
    """RGB, upright per EXIF; 16-bit gray scaled to 8-bit."""
    img = ImageOps.exif_transpose(Image.open(path))
    return (img.point(lambda v: v / 257, "L") if img.mode.startswith("I") else img).convert("RGB")


@dataclass
class Sample:
    id: str
    images: List[Image.Image]
    question: str
    metadata: Dict[str, Any] = field(default_factory=dict)
