"""ET3: energy-guided test-time transformation (CVPR 2026, arXiv 2603.26984)."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

from ...data import load_rgb
from ...data.tensors import to_pil, to_tensor
from ..base_defense import BaseDefense, DefenseConfig, DefenseResult

ASSETS = Path(__file__).parent / "assets"


@dataclass
class ET3DefenseConfig(DefenseConfig):
    clip_model: str = "hf-hub:chs20/fare4-clip"  # open_clip surrogate whose energy is minimized (robust FARE4)
    clip_pretrained: Optional[str] = None
    input_res: int = 224
    epsilon: float = 4.0  # L2 radius on [0, 1] at input_res^2 (FARE setting)
    alpha: float = 2.0  # step on the L2-normalized gradient, same scale
    steps: int = 2  # T
    temperature: float = 1.0
    logit_scale: float = 100.0
    labels_file: Optional[str] = None  # default: assets/imagenet21k_wordnet_lemmas.txt
    text_batch: int = 2560


@torch.no_grad()
def _label_embeddings(model, tokenizer, names, device, batch):
    """Class embedding: normalized text features averaged over the 80 OpenAI ImageNet templates."""
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES as templates
    from tqdm import tqdm
    step = max(1, batch // len(templates))
    out = []
    for i in tqdm(range(0, len(names), step), desc="ET3 label embeddings"):
        with torch.autocast("cuda"):
            e = model.encode_text(tokenizer([t(n) for n in names[i:i + step] for t in templates]).to(device))
        e = F.normalize(e.float(), dim=-1).view(-1, len(templates), e.shape[-1]).mean(1)
        out.append(F.normalize(e, dim=-1).half().cpu())
    return torch.cat(out)


class ET3Defense(BaseDefense):
    def __init__(self, config: ET3DefenseConfig):
        super().__init__(config)
        self._model = self._norm = self._text = None

    def _load(self):
        if self._model is not None:
            return
        import open_clip
        c = self.config
        model = open_clip.create_model_and_transforms(c.clip_model, pretrained=c.clip_pretrained)[0]
        self._model = model.eval().requires_grad_(False).to(c.device)
        self._norm = Normalize(model.visual.preprocess_cfg["mean"], model.visual.preprocess_cfg["std"])
        labels = Path(c.labels_file or ASSETS / "imagenet21k_wordnet_lemmas.txt")
        cache = ASSETS / re.sub(r"[^\w.-]", "_", f"text_{c.clip_model}_{c.clip_pretrained}_{labels.stem}.pt")
        if not cache.exists():
            names = [l.strip() for l in labels.read_text().splitlines() if l.strip()]
            torch.save(_label_embeddings(self._model, open_clip.get_tokenizer(c.clip_model), names, c.device, c.text_batch), cache)
        self._text = torch.load(cache).float().to(c.device)

    def _features(self, x):
        """L2-normalized image features of the robust CLIP."""
        x = F.interpolate(x, size=(self.config.input_res,) * 2, mode="bicubic", align_corners=False, antialias=True)
        return F.normalize(self._model.encode_image(self._norm(x.clamp(0, 1))), dim=-1)

    def _energy(self, x):
        """Eq. 1: E(x) = -t * logsumexp(f(x) / t), f = zero-shot logits over the label set."""
        c = self.config
        return -c.temperature * torch.logsumexp(c.logit_scale * self._features(x) @ self._text.T / c.temperature, dim=-1)

    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        self._load()
        c = self.config
        x0 = to_tensor(load_rgb(image_path)).to(c.device)
        scale = (x0[0, 0].numel() / c.input_res ** 2) ** 0.5  # same per-pixel budget at any size
        eps, alpha = c.epsilon * scale, c.alpha * scale
        x, energies = x0.clone(), []
        for _ in range(c.steps):  # Eq. 2
            x.requires_grad_(True)
            e = self._energy(x).sum()
            g = torch.autograd.grad(e, x)[0]
            energies.append(e.item())
            with torch.no_grad():
                d = (x - alpha * g / g.norm().clamp_min(1e-8)).clamp(0, 1) - x0
                x = (x0 + d * (eps / d.norm().clamp_min(1e-8)).clamp(max=1)).clamp(0, 1)  # l2 ball
        with torch.no_grad():
            energies.append(self._energy(x).item())
        return DefenseResult(cleaned_sample=to_pil(x), metadata={"energy": energies})
