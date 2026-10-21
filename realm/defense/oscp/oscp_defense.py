"""OSCP: one-step control purification (CVPR 2025, arXiv 2408.17064)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from ...data import load_rgb
from ..base_defense import BaseDefense, DefenseConfig, DefenseResult

ASSETS = Path(__file__).parent / "assets"


@dataclass
class OSCPDefenseConfig(DefenseConfig):
    seed: int = 100  # paper App. 7.1
    precision: str = "fp32"  # as upstream
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet: str = "lllyasviel/sd-controlnet-canny"
    lora: Optional[str] = None  # GAND-trained LCM LoRA; default: assets/checkpoints/oscp_lora.safetensors
    resolution: int = 512
    strength: float = 0.2  # t* = 200
    num_inference_steps: int = 5  # x strength = 1 denoising step
    guidance_scale: float = 1.0
    control_scale: float = 0.8
    canny_low: int = 100
    canny_high: int = 200


class OSCPDefense(BaseDefense):
    def __init__(self, config: OSCPDefenseConfig):
        super().__init__(config)
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetImg2ImgPipeline
        from safetensors.torch import load_file
        c = self.config
        controlnet = ControlNetModel.from_pretrained(c.controlnet, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            c.sd_model, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        ).to(c.device, dtype=torch.float32)  # fp32 inference, as upstream
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights({k.replace("base_model.model.", "unet."): v for k, v in load_file(c.lora or ASSETS / "checkpoints" / "oscp_lora.safetensors").items()})
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        self._load()
        c = self.config
        image = load_rgb(image_path)
        small = image.resize((c.resolution, c.resolution), Image.BILINEAR)
        edges = cv2.Canny(np.array(small), c.canny_low, c.canny_high)
        out = self._pipe(
            prompt="", image=small, control_image=Image.fromarray(np.stack([edges] * 3, -1)),
            num_inference_steps=c.num_inference_steps, strength=c.strength, guidance_scale=c.guidance_scale,
            controlnet_conditioning_scale=c.control_scale, generator=torch.Generator().manual_seed(c.seed),  # the paper's fixed seed; the global RNG is left alone
        ).images[0]
        return DefenseResult(cleaned_sample=out.resize(image.size, Image.BICUBIC))  # back to the input size
