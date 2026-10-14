"""O-Attack core (adapted from the official O-Attack G3 configuration)."""

import random

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from tqdm import tqdm


class CLIPAnchor(nn.Module):
    """CLIP surrogate returning projected late-layer CLS/EOS features [B, K, D] (CSA), with dropout (PSS)."""

    def __init__(self, model_id: str, image_k: int, text_k: int, res: int = 224):
        super().__init__()
        from transformers import AutoTokenizer, CLIPModel
        self.model = CLIPModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.image_k, self.text_k = image_k, text_k
        self.normalizer = transforms.Compose([
            transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(res),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.attn = [m for m in self.model.modules() if m.__class__.__name__ == "CLIPAttention"]
        self.ffn_p = 0.0
        for m in self.model.modules():
            if m.__class__.__name__ == "CLIPMLP":
                m.register_forward_hook(lambda _m, _i, out: F.dropout(out, p=self.ffn_p, training=True) if self.ffn_p > 0 else out)
        self.model.train()  # attention dropout is active only in train mode
        self.requires_grad_(False)

    def sample_dropout(self, cap: float):
        """Eq. 5: attention / FFN probabilities ~ U(0, cap), fixed for the next forwards."""
        p = random.uniform(0, cap)
        for m in self.attn:
            m.dropout = p
        self.ffn_p = random.uniform(0, cap)

    def image(self, x):
        hs = self.model.vision_model(pixel_values=self.normalizer(x), output_hidden_states=True).hidden_states
        feats = [self.model.visual_projection(self.model.vision_model.post_layernorm(h[:, 0])) for h in hs[-self.image_k:]]
        return F.normalize(torch.stack(feats, 1), dim=-1)  # Eq. 3

    @torch.no_grad()
    def text(self, texts):
        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        hs = self.model.text_model(**tok, output_hidden_states=True).hidden_states
        eos = tok["input_ids"].argmax(-1)
        idx = torch.arange(len(texts), device=eos.device)
        feats = [self.model.text_projection(self.model.text_model.final_layer_norm(h[idx, eos])) for h in hs[-self.text_k:]]
        z = F.normalize(torch.stack(feats, 1), dim=-1).mean(1)
        return F.normalize(z, dim=-1)  # Eq. 6


def mix_captions(bank):
    """Mix the words of two random captions: beta-prefix of one, (1-beta)-suffix of the other."""
    if len(bank) == 1:
        return bank[0]
    c1, c2 = (c.split() for c in random.sample(bank, 2))
    beta = random.uniform(0.0, 1.0)
    p1, p2 = int(beta * len(c1)), int((1 - beta) * len(c2))
    return " ".join(c1[:p1] + (c2[-p2:] if p2 > 0 else [])) or " ".join(c1 or c2)


def _moments(x):
    return x.mean(), x.var(unbiased=False)


def oattack(image, target, models, caps, src_bank, tgt_bank, source_aug, target_aug,
            num_iters, epsilon, alpha, views, text_alpha, lambda_v, lambda_t, img_index=0):
    """SCO (Eq. 10) with Adam and L-inf projection; images in [0, 255]."""
    delta = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=alpha)
    pbar = tqdm(range(num_iters), desc=f"O-Attack #{img_index}")
    for t in pbar:
        prog = t / max(num_iters - 1, 1)
        cap_t = [c[0] + (c[1] - c[0]) * prog for c in caps]  # Eq. 4
        adv = image + delta
        adv_views = torch.cat([source_aug(adv) for _ in range(views)])
        with torch.no_grad():
            tgt_views = torch.cat([target_aug(target) for _ in range(views)])
        src_txt = [mix_captions(src_bank) for _ in range(views)]
        tgt_txt = [mix_captions(tgt_bank) for _ in range(views)]
        vis, txt = [], []
        for m, cap in zip(models, cap_t):
            m.sample_dropout(cap)  # shared by the adv/target images of this step
            with torch.no_grad():
                z_tgt = m.image(tgt_views)
            z_adv = m.image(adv_views)
            vis.append((z_adv * z_tgt).sum(-1).mean(1))  # Eq. 7-8
            m.sample_dropout(cap)  # captions
            t_tgt, t_src = m.text(tgt_txt), m.text(src_txt)
            txt.append(((z_adv * t_tgt[:, None]).sum(-1) - (z_adv * t_src[:, None]).sum(-1)).mean(1))
        mu_v, var_v = _moments(torch.cat(vis))  # Eq. 9
        mu_t, var_t = _moments(torch.cat(txt))
        objective = mu_v - lambda_v * var_v + text_alpha * (mu_t - lambda_t * var_t)  # -Eq. 10
        optimizer.zero_grad()
        delta.grad = -torch.autograd.grad(objective, delta)[0]
        optimizer.step()
        delta.data.clamp_(-epsilon, epsilon)
        if t % 10 == 0 or t == num_iters - 1:  # .item() syncs the GPU
            pbar.set_postfix(vis=f"{mu_v.item():.3f}", txt=f"{mu_t.item():.3f}", max_delta=f"{delta.abs().max().item():.1f}")
    return torch.clamp((image + delta.detach()) / 255.0, 0.0, 1.0)
