"""TextDiffuser-2 inpainting (microsoft/unilm, textdiffuser-2), the path SceneTAP runs: one text box, regional diffusion."""

import string

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + " "


def square_region(left: int, top: int, right: int, bottom: int, width: int, height: int, scale: float) -> tuple:
    """Square crop of `scale` x the box's long side around it, shifted inside the image where it fits."""
    half = max(right - left, bottom - top) * scale // 2
    cx, cy = (left + right) // 2, (top + bottom) // 2
    l, r, t, b = max(cx - half, 0), min(cx + half, width), max(cy - half, 0), min(cy + half, height)
    if r - l > b - t:
        d = r - l - (b - t)
        if t - d / 2 < 0:
            t, b = 0, r - l
        elif b + d / 2 > height:
            t, b = height - (r - l), height
        else:
            t, b = t - d / 2, b + d / 2
    elif r - l < b - t:
        d = b - t - (r - l)
        if l - d / 2 < 0:
            l, r = 0, b - t
        elif r + d / 2 > width:
            l, r = width - (b - t), width
        else:
            l, r = l - d / 2, r + d / 2
    return int(l), int(t), int(r), int(b)


def _grid(v: int, origin: int, edge: int, size: int) -> int:
    """Image coordinate v -> the crop's 128 grid (via the 512 canvas), with the official float steps."""
    return int((v - origin) / edge * size / size * 512) // 4


class TextDiffuser:

    def __init__(self, sd_model: str, td_model: str, device: str, td_revision: str = None):
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer")
        self.tokenizer.add_tokens([f"{p}{i}" for i in range(520) for p in "ltrb"] + [f"[{c}]" for c in ALPHABET])  # box coordinates, characters
        self.text_encoder = CLIPTextModel.from_pretrained(td_model, subfolder="text_encoder", revision=td_revision).to(device).half()
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.vae = AutoencoderKL.from_pretrained(sd_model, subfolder="vae").to(device).half()
        # the fork's U-Net takes [noise, mask, masked latent] through a 9-channel conv_in
        self.unet = UNet2DConditionModel.from_pretrained(td_model, subfolder="unet", in_channels=9, revision=td_revision).to(device).half()
        self.scheduler = DDPMScheduler.from_pretrained(sd_model, subfolder="scheduler", steps_offset=0)  # the fork ignores the SD config's offset: t = 950 ... 0

    def _ids(self, caption: str, box: tuple, text: str) -> list:
        x0, y0, x1, y1 = box
        chars = " ".join(f"[{c}]" for c in text)
        ids = self.tokenizer.encode(f"{caption} <|endoftext|><|startoftext|> l{x0} t{y0} r{x1} b{y1} {chars} <|endoftext|>")[:77]
        return ids + [self.tokenizer.pad_token_id] * (77 - len(ids))

    @torch.no_grad()
    def write(self, image: Image.Image, box: tuple, text: str, caption: str, *, steps: int, guidance: float, scale: float,
              n: int, generator: torch.Generator) -> list:
        """n images with `text` written in box (l, t, r, b) of the original image; only the crop around it is regenerated."""
        W, H = image.size
        x0, y0, x1, y1 = box
        crop = square_region(x0, y0, x1, y1, W, H, scale)
        edge = crop[2] - crop[0]
        if crop[3] - crop[1] != edge:
            raise RuntimeError(f"crop {crop} is not square")
        q = [_grid(v, crop[k % 2], edge, (W, H)[k % 2]) for k, v in enumerate(box)]
        mask = Image.new("L", (512, 512), 0)
        ImageDraw.Draw(mask).rectangle([4 * v for v in q], fill=1)

        dev = self.device
        ids = torch.tensor([self._ids(caption, tuple(q), text)] * n, device=dev)
        none = torch.full_like(ids, self.tokenizer.pad_token_id)
        cond, uncond = self.text_encoder(ids)[0].half(), self.text_encoder(none)[0].half()

        src = torch.from_numpy(np.asarray(image.crop(crop).resize((512, 512)), np.float32) / 255).permute(2, 0, 1)[None].to(dev)
        m = torch.from_numpy(np.asarray(mask, np.float32)).half().to(dev)[None, None].repeat(n, 1, 1, 1)
        masked = (src.sub(0.5).div(0.5) * (1 - m)).half()
        feature = (self.vae.encode(masked).latent_dist.sample(generator) * self.vae.config.scaling_factor).half()
        feature_mask = F.interpolate(m, size=(64, 64), mode="nearest")

        x = torch.randn((n, 4, 64, 64), generator=generator).to(dev).half()
        self.scheduler.set_timesteps(steps)
        for t in self.scheduler.timesteps:
            inp = torch.cat([x, feature_mask, feature], dim=1)
            eps_c = self.unet(inp, t, encoder_hidden_states=cond).sample
            eps_u = self.unet(inp, t, encoder_hidden_states=uncond).sample
            x = self.scheduler.step(eps_u + guidance * (eps_c - eps_u), t, x, generator=generator).prev_sample
        out = self.vae.decode(x / self.vae.config.scaling_factor, return_dict=False)[0]

        results = []
        for patch in out.float().cpu():
            patch = Image.fromarray(((patch / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
            full = image.copy()
            full.paste(patch.resize((edge, edge)), crop[:2])
            results.append(full)
        return results
