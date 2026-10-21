import torch

from ..._diffusion import Purifier


class PurificationForward(Purifier):
    """Purifier that runs DDPM at stride 1 with upstream's p_sample."""

    def __init__(self, model, max_timestep, attack_steps, sampling_method, device, gaussian_diffusion):
        super().__init__(model, max_timestep, attack_steps, sampling_method, device)
        self.gaussian_diffusion = gaussian_diffusion

    def denoising_process(self, ori_x, x, seq):
        if self.eta == 1 and len(seq) == seq[-1] + 1:  # ddpm, stride 1: upstream p_sample
            for i in reversed(seq):
                t = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
                x = self.gaussian_diffusion.p_sample(self.diffusion, x, t, clip_denoised=True)["sample"]
            return x
        return super().denoising_process(ori_x, x, seq)
