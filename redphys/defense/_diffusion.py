"""Guided-diffusion purification shared by FreqPure and BlueSuffix (DiffPure-style)."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import weight
from ._guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

CHECKPOINT = Path(__file__).parent / "freqpure" / "assets" / "pretrained" / "guided_diffusion" / "256x256_diffusion_uncond.pt"
MODEL = dict(attention_resolutions="32,16,8", class_cond=False, diffusion_steps=1000, rescale_timesteps=True, timestep_respacing="1000",
             image_size=256, learn_sigma=True, noise_schedule="linear", num_channels=256, num_head_channels=64, num_res_blocks=2,
             resblock_updown=True, use_scale_shift_norm=True)  # DiffPure's configs/imagenet.yml


def load_diffusion(checkpoint, device, use_fp16: bool):
    """(unconditional ImageNet 256 model, gaussian diffusion)."""
    checkpoint = weight(checkpoint or CHECKPOINT, "freqpure")
    model, diffusion = create_model_and_diffusion(**{**model_and_diffusion_defaults(), **MODEL, "use_fp16": use_fp16})
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    if use_fp16:
        model.convert_to_fp16()  # as DiffPure
    return model.eval().to(device), diffusion


def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    """Linear betas, float32."""
    return torch.from_numpy(np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)).float()


def get_diffusion_params(max_timesteps: str, num_denoising_steps: str):
    """Comma lists -> (max timestep per round - 1, denoising timesteps per round)."""
    ts, ns = ([int(i) for i in s.split(",")] for s in (max_timesteps, num_denoising_steps))
    assert len(ts) == len(ns)
    return [t - 1 for t in ts], [[j - 1 for j in range(t // n, t + 1, t // n)] for t, n in zip(ts, ns)]


def diff2clf(x):
    """[-1, 1] -> [0, 1]."""
    return torch.clamp(x / 2 + 0.5, 0, 1)


def clf2diff(x):
    """[0, 1] -> [-1, 1]."""
    return torch.clamp((x - 0.5) * 2, -1, 1)


class Purifier(torch.nn.Module):
    """Rounds of noising to max_timestep and DDPM/DDIM denoising over attack_steps, at 256 x 256."""

    def __init__(self, model, max_timestep, attack_steps, sampling_method, device):
        super().__init__()
        assert sampling_method in ("ddim", "ddpm")
        self.diffusion, self.device = model, device
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.alphas = (1 - self.betas).cumprod(dim=0)
        self.alphas_pad = (1 - torch.cat([torch.zeros(1).to(device), self.betas], dim=0)).cumprod(dim=0)
        self.max_timestep, self.attack_steps = max_timestep, attack_steps
        self.eta = 0 if sampling_method == "ddim" else 1

    def compute_alpha(self, t):
        return self.alphas_pad.index_select(0, t + 1).view(-1, 1, 1, 1)

    def get_noised_x(self, x, t: int):
        e = torch.randn_like(x)
        a = self.alphas.index_select(0, (torch.ones(x.shape[0]) * t).to(x.device).long()).view(-1, 1, 1, 1)
        return x * a.sqrt() + e * (1.0 - a).sqrt()

    def eps(self, x, t):
        return torch.split(self.diffusion(x, t), 3, dim=1)[0]  # learn_sigma: drop the variance channels

    def x0_hook(self, ori_x, x0_t):
        return x0_t

    def denoising_process(self, ori_x, x, seq):
        n = x.size(0)
        xt = x
        for i, j in zip(reversed(seq), reversed([-1] + list(seq[:-1]))):
            t = (torch.ones(n) * i).to(x.device)
            at, at_next = self.compute_alpha(t.long()), self.compute_alpha((torch.ones(n) * j).to(x.device).long())
            et = self.eps(xt, t)
            x0_t = self.x0_hook(ori_x, (xt - et * (1 - at).sqrt()) / at.sqrt())
            c1 = self.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt

    def forward(self, x):
        """[1,3,H,W] in [0, 1] -> purified, same size."""
        size = x.shape[-2:]
        x_diff = clf2diff(F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True))
        for t, seq in zip(self.max_timestep, self.attack_steps):
            x_diff = self.denoising_process(x_diff, self.get_noised_x(x_diff, t), seq)
        return diff2clf(F.interpolate(x_diff, size=size, mode="bilinear", align_corners=False, antialias=True))
