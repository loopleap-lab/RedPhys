"""FreqPure: frequency-domain diffusion purification."""

from dataclasses import dataclass
from typing import Optional

import torch

from ...data import load_rgb
from ...data.tensors import to_pil, to_tensor
from ..base_defense import BaseDefense, DefenseConfig, DefenseResult


@dataclass
class FreqPureDefenseConfig(DefenseConfig):
    amplitude_cut_range: int = 10
    phase_cut_range: int = 10
    delta: float = 0.3  # phase clip around the input's low frequencies (the original's --delta)
    forward_noise_steps: int = 50
    max_timesteps: str = "50,50,50,50,50,50,50,50"
    num_denoising_steps: str = "5,5,5,5,5,5,5,5"
    sampling_method: str = "ddpm"
    diffusion_checkpoint: Optional[str] = None  # default: freqpure/assets/pretrained/guided_diffusion/


class FreqPureDefense(BaseDefense):
    def __init__(self, config: FreqPureDefenseConfig):
        super().__init__(config)
        self._purifier = None

    def _initialize_models(self):
        if self._purifier is not None:
            return
        from .._diffusion import get_diffusion_params, load_diffusion
        from .core.purification import PurificationForward

        print("Loading FreqPure diffusion model...")
        c, device = self.config, torch.device(self.config.device)
        model, _ = load_diffusion(c.diffusion_checkpoint, device, use_fp16=False)
        max_timestep, steps = get_diffusion_params(c.max_timesteps, c.num_denoising_steps)
        self._purifier = PurificationForward(model, max_timestep, steps, c.sampling_method, device, c.amplitude_cut_range, c.phase_cut_range,
                                             c.delta, int(c.forward_noise_steps))  # YAML may give 50.0

    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        self._initialize_models()
        with torch.no_grad():
            purified = self._purifier(to_tensor(load_rgb(image_path)).to(self.config.device))
        return DefenseResult(cleaned_sample=to_pil(purified))
