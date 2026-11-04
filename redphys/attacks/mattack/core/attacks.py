"""M-Attack functions (adapted from original M-Attack)."""

import torch
from torch import nn
from typing import Optional
import torchvision.transforms as transforms
from tqdm import tqdm

from ... import trace


def fgsm_attack(
    image_tensor: torch.Tensor,
    tgt_tensor: torch.Tensor,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    num_iters: int,
    epsilon: float,
    alpha: float,
) -> torch.Tensor:
    """Run an FGSM attack to generate an adversarial example."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)

    pbar = tqdm(range(num_iters), desc=f"M-Attack FGSM #{img_index}")

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        sim = ensemble_loss(ensemble_extractor(source_crop(image_tensor + delta)))
        loss = sim

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        delta.data = torch.clamp(
            delta + alpha * torch.sign(grad),
            min=-epsilon,
            max=epsilon,
        )

        if epoch % 10 == 0 or epoch == num_iters - 1:  # .item() syncs the GPU
            pbar.set_postfix(max_delta=f"{delta.abs().max().item():.3f}", sim=f"{sim.item():.3f}")
            trace.log(epoch, sim=sim, max_delta=delta.abs().max())

    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


def mifgsm_attack(
    image_tensor: torch.Tensor,
    tgt_tensor: torch.Tensor,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    num_iters: int,
    epsilon: float,
    alpha: float,
    momentum_decay: float = 0.9,
) -> torch.Tensor:
    """Run an MI-FGSM attack to generate an adversarial example."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    momentum = torch.zeros_like(image_tensor, requires_grad=False)

    pbar = tqdm(range(num_iters), desc=f"M-Attack MI-FGSM #{img_index}")

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        sim = ensemble_loss(ensemble_extractor(source_crop(image_tensor + delta)))
        loss = sim

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        momentum = momentum * momentum_decay + grad
        delta.data = torch.clamp(
            delta + alpha * torch.sign(momentum),
            min=-epsilon,
            max=epsilon,
        )

        if epoch % 10 == 0 or epoch == num_iters - 1:  # .item() syncs the GPU
            pbar.set_postfix(max_delta=f"{delta.abs().max().item():.3f}", sim=f"{sim.item():.3f}")
            trace.log(epoch, sim=sim, max_delta=delta.abs().max())

    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


def pgd_attack(
    image_tensor: torch.Tensor,
    tgt_tensor: torch.Tensor,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    num_iters: int,
    epsilon: float,
    alpha: float,
) -> torch.Tensor:
    """Run a PGD attack to generate an adversarial example."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=alpha)

    pbar = tqdm(range(num_iters), desc=f"M-Attack PGD #{img_index}")

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        sim = ensemble_loss(ensemble_extractor(source_crop(image_tensor + delta)))
        loss = -sim

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-epsilon,
            max=epsilon,
        )

        if epoch % 10 == 0 or epoch == num_iters - 1:  # .item() syncs the GPU
            pbar.set_postfix(max_delta=f"{delta.abs().max().item():.3f}", sim=f"{sim.item():.3f}")
            trace.log(epoch, sim=sim, max_delta=delta.abs().max())

    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image
