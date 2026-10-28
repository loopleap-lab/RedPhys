"""Eq. 4: l_inf PGD maximizing ||phi(x_adv) - phi_org(x)||^2, as FARE (uniform start; its sign-momentum is a plain sign step)."""

import torch


def pgd(vlm, encoder, target: torch.Tensor, x: torch.Tensor, eps: float, alpha: float, steps: int) -> torch.Tensor:
    """x [3, H, W], eps and alpha in [0, 1]."""
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = (x + delta).clamp(0, 1) - x
    for _ in range(steps):
        delta.requires_grad_(True)
        with torch.enable_grad():
            loss = (vlm.features(encoder, vlm.pixel_inputs(x + delta)) - target).pow(2).sum()
            grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()
