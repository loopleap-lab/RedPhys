import numpy as np
import torch

from ..._diffusion import Purifier, clf2diff, diff2clf


class PurificationForward(Purifier):
    """Purifier whose x0 estimate takes the low-frequency amplitude and phase of the input (FreqPure)."""

    def __init__(self, model, max_timestep, attack_steps, sampling_method, device, amplitude_cut_range, phase_cut_range, delta,
                 forward_noise_steps):
        super().__init__(model, max_timestep, attack_steps, sampling_method, device)
        self.amplitude_cut_range, self.phase_cut_range, self.delta = amplitude_cut_range, phase_cut_range, delta
        self.forward_noise_steps, self._disks = forward_noise_steps, {}

    def compute_fft(self, image):
        amplitude_channels = []
        phase_channels = []
        for channel in range(3):
            fshift = torch.fft.fftshift(torch.fft.fft2(image[channel, :, :]))
            amplitude_channels.append(torch.abs(fshift))
            phase_channels.append(torch.angle(fshift) + torch.pi)
        return amplitude_channels, phase_channels

    def low_frequency(self, rows, cols, cut_range):
        """Centered disk mask of radius cut_range, cached per instance (an lru_cache would keep the model alive)."""
        if (k := (rows, cols, cut_range)) not in self._disks:
            U, V = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
            self._disks[k] = torch.from_numpy(np.sqrt(U ** 2 + V ** 2) <= cut_range).to(self.device)
        return self._disks[k]

    def low_pass_exchange(self, src, dst, cut_range):
        """dst channels with src inside the low-frequency disk."""
        return [torch.where(self.low_frequency(*s.shape, cut_range), s, d) for s, d in zip(src, dst)]

    def phase_low_pass_exchange(self, src, dst):
        """The phase exchange, then the original's clip to src +- delta inside the disk (a no-op for delta >= 0)."""
        out = []
        for s, p in zip(src, self.low_pass_exchange(src, dst, self.phase_cut_range)):
            low = self.low_frequency(*s.shape, self.phase_cut_range)
            p[low] = torch.clip(p[low], s[low] - self.delta, s[low] + self.delta)
            out.append(p)
        return out

    def reconstruct_image(self, filtered_amplitude_channels, phase_channels):
        reconstructed_image = []
        for channel in range(3):
            fshift_filtered = filtered_amplitude_channels[channel] * torch.exp(1j * (phase_channels[channel] - torch.pi))
            img_reconstructed = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fshift_filtered)))
            reconstructed_image.append(torch.clip(img_reconstructed, 0, 255) / 255)
        return torch.stack(reconstructed_image, dim=2)

    def x0_hook(self, x, x_0_t):
        """Amplitude-phase exchange with a one-step denoised x."""
        x_t = self.get_noised_x(x, self.forward_noise_steps)
        t = (torch.ones(x.size(0)) * self.forward_noise_steps).to(x.device)
        at = self.compute_alpha(t.long())
        x = diff2clf((x_t - self.eps(x_t, t) * (1 - at).sqrt()) / at.sqrt()) * 255
        x_0_t = diff2clf(x_0_t) * 255

        batch, channel, height, width = x.shape
        new_x_0_t = torch.zeros(size=(batch, height, width, channel), device=self.device)
        for b in range(batch):
            amplitude, phase = self.compute_fft(x[b])
            amplitude_0_t, phase_0_t = self.compute_fft(x_0_t[b])
            new_x_0_t[b] = self.reconstruct_image(self.low_pass_exchange(amplitude, amplitude_0_t, self.amplitude_cut_range),
                                                  self.phase_low_pass_exchange(phase, phase_0_t))
        return clf2diff(new_x_0_t.permute(0, 3, 1, 2))
