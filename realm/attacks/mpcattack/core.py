"""MPCAttack surrogates and optimization (adapted from the official MPCAttack)."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from tqdm import tqdm

_CLIP_NORM = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def _normalizer(res):
    return transforms.Compose([
        transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
        _CLIP_NORM,
    ])


class CLIPSurrogate(nn.Module):
    """Cross-modal paradigm: CLIP image/text embeddings."""

    def __init__(self, model_id):
        super().__init__()
        from transformers import AutoTokenizer, CLIPModel
        self.model = CLIPModel.from_pretrained(model_id).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.normalizer = _normalizer(224)

    def forward(self, x):
        return F.normalize(self.model.get_image_features(pixel_values=self.normalizer(x)), dim=-1)

    @torch.no_grad()
    def text(self, text):
        tok = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        return F.normalize(self.model.get_text_features(**tok), dim=-1)


class InternVLSurrogate(nn.Module):
    """Multi-modal paradigm: InternVL3 features; also the text generator."""

    def __init__(self, model_id, prompt, max_new_tokens):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).eval()
        self.prompt, self.max_new_tokens = prompt, max_new_tokens
        self.normalizer = _normalizer(448)

    def forward(self, x):
        x = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(x)
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)  # one 448 tile, as upstream
        feats = self.model.get_image_features(pixel_values=self.normalizer(x).to(torch.bfloat16))
        return (feats / feats.norm(dim=1, keepdim=True)).mean(1).float()

    @torch.no_grad()
    def describe(self, image):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class DINOv2Surrogate(nn.Module):
    """Self-supervised paradigm: DINOv2 CLS token."""

    def __init__(self, model_id):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.normalizer = _normalizer(224)

    def forward(self, x):
        return F.normalize(self.model(pixel_values=self.normalizer(x)).last_hidden_state[:, 0], dim=-1)


def features(clips, internvl, dino, x, texts=None, lam=1.0):
    """Eq. 2-5: per-paradigm L2-normalized features, concatenated; CLIP fused with text features (Eq. 4)."""
    clip = [m(x) for m in clips]
    if texts is not None:
        clip = [lam * c + (1 - lam) * t for c, t in zip(clip, texts)]
    return torch.cat([F.normalize(torch.cat(clip, 1), dim=-1), F.normalize(internvl(x), dim=-1), dino(x)], 1)


def info_nce(z_adv, z_tgt, z_src, tau, omega):
    """Eq. 6 (as upstream): -CE over [omega * sim(adv, tgt), sim(adv, src)] / tau."""
    z_adv, z_tgt, z_src = (F.normalize(z, dim=-1) for z in (z_adv, z_tgt, z_src))
    logits = torch.stack([(z_adv * z_tgt).sum(-1) * omega / tau, (z_adv * z_src).sum(-1) / tau], 1)
    return -F.cross_entropy(logits, torch.zeros(len(z_adv), dtype=torch.long, device=z_adv.device))


def mpcattack(image, z_src, z_tgt, extract, source_crop, num_iters, epsilon, alpha, momentum, tau, omega, img_index=0):
    """Algorithm 1: MI-FGSM on the contrastive loss; images in [0, 255]."""
    delta = (epsilon / 255 * torch.randn_like(image)).requires_grad_(True)  # small random init, as upstream
    g = torch.zeros_like(image)
    pbar = tqdm(range(num_iters), desc=f"MPCAttack #{img_index}")
    for i in pbar:
        z_adv = extract(source_crop(image + delta))
        loss = info_nce(z_adv, z_tgt, z_src, tau, omega)
        grad = torch.autograd.grad(loss, delta)[0]
        g = momentum * g + grad / grad.abs().sum().clamp_min(1e-12)  # Alg. 1 line 6
        delta.data = torch.clamp(delta + alpha * torch.sign(g), -epsilon, epsilon)
        if i % 10 == 0 or i == num_iters - 1:  # .item() syncs the GPU
            with torch.no_grad():
                sims = [F.cosine_similarity(z_adv, z, dim=-1).item() for z in (z_tgt, z_src)]
            pbar.set_postfix(loss=f"{loss.item():.3f}", sim_tgt=f"{sims[0]:.3f}", sim_src=f"{sims[1]:.3f}")
    return torch.clamp((image + delta.detach()) / 255.0, 0.0, 1.0)
