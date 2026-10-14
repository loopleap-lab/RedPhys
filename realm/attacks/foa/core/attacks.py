"""FOA attack functions (from original FOA-Attack)."""

import torch
from torch import nn
from typing import Optional
import torchvision.transforms as transforms
from tqdm import tqdm


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
    device: str,
    use_source_crop: bool = True,
    use_target_crop: bool = True,
) -> torch.Tensor:
    delta = torch.zeros_like(image_tensor, requires_grad=True)

    pbar = tqdm(range(num_iters), desc=f"Attack progress")

    for epoch in pbar:

        ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        adv_image = image_tensor + delta
        x = source_crop(adv_image) if use_source_crop else adv_image  # Alg. 1
        sim = ensemble_loss(*ensemble_extractor(x))

        loss = sim

        if epoch % 10 == 0 or epoch == num_iters - 1:  # .item() syncs the GPU
            pbar.set_postfix(max_delta=f"{delta.abs().max().item():.3f}", mean_delta=f"{delta.abs().mean().item():.3f}",
                             similarity=f"{sim.item():.5f}")

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        delta.data = torch.clamp(
            delta + alpha * torch.sign(grad),
            min=-epsilon,
            max=epsilon,
        )

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
    device: str,
    use_source_crop: bool = True,
    use_target_crop: bool = True,
    momentum_decay: float = 0.9,
) -> torch.Tensor:
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    momentum = torch.zeros_like(image_tensor, requires_grad=False)

    pbar = tqdm(range(num_iters), desc=f"Attack progress")

    for epoch in pbar:

        ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        adv_image = image_tensor + delta
        x = source_crop(adv_image) if use_source_crop else adv_image  # Alg. 1
        sim = ensemble_loss(*ensemble_extractor(x))

        loss = sim

        if epoch % 10 == 0 or epoch == num_iters - 1:  # .item() syncs the GPU
            pbar.set_postfix(max_delta=f"{delta.abs().max().item():.3f}", mean_delta=f"{delta.abs().mean().item():.3f}",
                             similarity=f"{sim.item():.5f}")

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        momentum = momentum * momentum_decay + grad
        delta.data = torch.clamp(
            delta + alpha * torch.sign(momentum),
            min=-epsilon,
            max=epsilon,
        )

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
    device: str,
    use_source_crop: bool = True,
    use_target_crop: bool = True,
) -> torch.Tensor:
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=alpha)

    pbar = tqdm(range(num_iters), desc=f"Attack progress")

    for epoch in pbar:

        ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        adv_image = image_tensor + delta
        x = source_crop(adv_image) if use_source_crop else adv_image  # Alg. 1
        sim = ensemble_loss(*ensemble_extractor(x))

        loss = -sim  # negate to maximize similarity

        if epoch % 10 == 0 or epoch == num_iters - 1:  # .item() syncs the GPU
            pbar.set_postfix(max_delta=f"{delta.abs().max().item():.3f}", mean_delta=f"{delta.abs().mean().item():.3f}",
                             similarity=f"{sim.item():.5f}")

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-epsilon,
            max=epsilon,
        )

    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


__all__ = ["fgsm_attack", "mifgsm_attack", "pgd_attack"]
