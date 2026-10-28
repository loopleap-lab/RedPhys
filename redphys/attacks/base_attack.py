"""Attack base class and image/tensor helpers."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ..data import Sample, load_rgb
from ..data.files import IMAGE_SUFFIXES
from ..data.tensors import to_pil, to_tensor, with_precision  # to_pil, to_tensor: re-exported for the attacks


def load_surrogates(classes: dict, names, device) -> list:
    """Frozen eval-mode surrogates, one per name, from {name: class}."""
    if bad := [n for n in names if n not in classes]:
        raise ValueError(f"Unknown backbone(s) {bad}; available: {list(classes)}")
    models = []
    for n in names:
        models.append(classes[n]().eval().to(device).requires_grad_(False))
        print(f"  ✓ Loaded {n}")
    return models


def find_by_stem(directory, stem: str, exts=IMAGE_SUFFIXES, required: bool = False) -> Optional[Path]:
    """directory/<stem><ext> for the first ext that exists, else None (required: an error)."""
    if required and not directory:
        raise ValueError("target_images_dir is not set")
    p = next((p for ext in exts if (p := Path(directory) / f"{stem}{ext}").exists()), None)
    if required and p is None:
        raise FileNotFoundError(f"No image for '{stem}' in {directory}")
    return p


def open_by_stem(directory, stem: str, exts=IMAGE_SUFFIXES) -> Image.Image:
    """The RGB image directory/<stem>.<ext> (EXIF-upright, as sources); an error if there is none."""
    return load_rgb(find_by_stem(directory, stem, exts, required=True))


def resize_center_crop(img: Image.Image, res: int) -> Image.Image:
    """Bicubic resize of the short side to res, then a res x res center crop."""
    return T.CenterCrop(res)(T.Resize(res, interpolation=T.InterpolationMode.BICUBIC)(img))


def aspect_crop(res: int, scale, ratio, aspect: float) -> T.RandomResizedCrop:
    """RandomResizedCrop with the aspect-ratio range scaled by the image's W/H."""
    return T.RandomResizedCrop(res, scale=scale, ratio=(aspect * ratio[0], aspect * ratio[1]))


def random_crops(cfg, aspect: float = 1.0):
    """(source, target) RandomResizedCrop to cfg.input_res (identity if use_*_crop is off); the source ratio is x image W/H."""
    source = aspect_crop(cfg.input_res, cfg.crop_scale, cfg.crop_ratio, aspect) if cfg.use_source_crop else torch.nn.Identity()
    target = T.RandomResizedCrop(cfg.input_res, scale=cfg.crop_scale) if cfg.use_target_crop else torch.nn.Identity()
    return source, target


def pixel_linf(adv: Image.Image, clean: Image.Image) -> float:
    """L-inf of two images (0-255)."""
    return float(np.abs(np.asarray(adv.convert("RGB"), np.int16) - np.asarray(clean.convert("RGB"), np.int16)).max())


def target_text(config, sample: Sample) -> str:
    """config.target_text, else the sample's attack_target_text."""
    if text := getattr(config, "target_text", None) or sample.metadata.get("attack_target_text"):
        return text
    raise ValueError(f"No target text for sample '{sample.id}': set target_text or pass labels")


class LLMClientMixin:
    """Lazy OpenAI-compatible client from config.llm_server_url (None: OpenAI's API) and llm_api_key_env; checks llm_model if declared."""

    _llm_client = None

    def llm(self):
        if self._llm_client is None:
            url, env = self.config.llm_server_url, self.config.llm_api_key_env
            if hasattr(self.config, "llm_model") and not self.config.llm_model:
                raise ValueError(f"{type(self).__name__} needs llm_model in its YAML")
            if not url and not os.environ.get(env):
                raise ValueError(f"{type(self).__name__} calls an LLM: set {env}, or llm_server_url in its YAML")
            from openai import OpenAI
            self._llm_client = OpenAI(base_url=f"{url}/v1" if url else None, api_key=os.environ.get(env, "dummy"))
        return self._llm_client

    def llm_text(self, prompt: str) -> str:
        """The LLM's reply to prompt (concise system prompt, greedy); raises if empty."""
        from ..judge import clean_llm_text, no_reasoning
        client = self.llm()
        resp = client.chat.completions.create(
            model=self.config.llm_model, max_tokens=256, temperature=0.0, extra_body=no_reasoning(client),
            messages=[{"role": "system", "content": "You are a concise assistant. /no_think"}, {"role": "user", "content": prompt}])
        if not (text := clean_llm_text(resp.choices[0].message.content)):
            raise RuntimeError(f"LLM returned empty text for: {prompt[:80]!r}")
        return text


@dataclass
class AttackConfig:
    """Shared hyperparameters. A subclass also declares the run inputs it needs, filled in by redphys.attack (attacks/run.py):
    target_images_dir (stem-paired targets; required then), clean_images_dir / source_images_dir, source_images (file names),
    target_text (required then, or per-case labels) and gradcam_masks_dir (under the output). The YAML overrides them."""

    uses_question: ClassVar[bool] = False  # attacks the question: DEFAULT_QUESTION when none is given
    needs_source_text: ClassVar[bool] = False  # needs the correct answer: per-case labels (or a source_text setting)
    perturbs_image: ClassVar[bool] = True  # False: the attack leaves the image (the question) or replaces it (FigStep)

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
        self.config = config  # randomness: run.py seeds each sample (seed_all)
        with_precision(self, "generate")

    @abstractmethod
    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        """Adversarial example for a sample (kwargs: sample_idx)."""
