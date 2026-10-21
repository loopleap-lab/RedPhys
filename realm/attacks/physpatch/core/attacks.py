"""Physical adversarial patch attack algorithms (PGD and MI-FGSM)."""

import os
import random
import numpy as np
import torch

from .mask_generator import DynamicPatchGenerator
from .transforms import My_T


def set_environment(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pgd(
    image_tensor, tgt_tensor, ensemble_extractor, ensemble_loss,
    source_crop, target_crop, center=None,
    num_iters=10, epsilon=8, alpha=1.0, decay=1.0, device='cuda', seed=42,
    tau0=0.6, tau_step=0.02, sigma=0.2, field_lr=0.15, patch_area=0.01, full_weight=0.1,
):
    """PGD attack for physical adversarial patches."""
    set_environment(seed)
    init_patch = image_tensor.clone().detach().to(device)
    print("patch:", torch.min(init_patch), torch.max(init_patch), init_patch.shape)
    print("ori:", torch.min(image_tensor), torch.max(image_tensor), image_tensor.shape)
    print("tgt:", torch.min(tgt_tensor), torch.max(tgt_tensor), tgt_tensor.shape)

    delta = torch.zeros_like(init_patch, requires_grad=True).to(device)

    _, _, H, W = image_tensor.shape
    patch_generator = DynamicPatchGenerator(image_size=(W, H), learning_rate=field_lr, sigma=sigma, device=device).to(device)
    threshold_list = [round(tau0 + i * tau_step, 6) for i in range(num_iters)]
    mask_expanded = torch.ones([1, 1, H, W]).to(device)
    trans = My_T(num_copies=1)
    th = int(H * W * patch_area)  # 120*120 of 1600x900
    with torch.no_grad():
        ensemble_loss.set_full_feature(tgt_tensor)  # constant target
    for iter in range(num_iters):
        with torch.no_grad():
            tgt = target_crop(tgt_tensor)
            ensemble_loss.set_ground_truth(tgt)

        adv_patch = init_patch + delta
        if mask_expanded.sum() > th:
            mask_expanded = patch_generator(center, threshold_list[iter])
        adv_image = image_tensor * (1 - mask_expanded) + adv_patch * mask_expanded

        local = torch.cat([trans.transform(source_crop(adv_image)) for _ in range(1)])
        features, features_local = ensemble_extractor(local)
        full = ensemble_extractor.encode_img(adv_image)
        loss_a = ensemble_loss(features, features_local, full, full_weight=full_weight)
        loss = loss_a

        growing = bool(mask_expanded.sum() > th)
        if growing:
            grads = torch.autograd.grad(loss, [delta, mask_expanded], create_graph=False)
            grad = grads[0]
            grad_mask = grads[1]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy(), "mask_grads:", grad_mask.mean().detach().cpu().numpy())
        else:
            grads = torch.autograd.grad(loss, [delta], create_graph=False)
            grad = grads[0]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy())

        delta.data = delta + alpha * torch.sign(grad)
        delta.data = torch.clamp(delta.data, min=-epsilon, max=epsilon)
        if growing:
            with torch.no_grad():
                patch_generator.update_potential_field(grad_mask)

    final_patch = init_patch + delta
    final_image = image_tensor * (1 - mask_expanded) + final_patch * mask_expanded
    final_image = torch.clamp(final_image/255.0, 0.0, 1.0)
    return final_image.detach()


def mifgsm(
    image_tensor, tgt_tensor, ensemble_extractor, ensemble_loss,
    source_crop, target_crop, center=None,
    num_iters=10, epsilon=8, alpha=1.0, decay=1.0, device='cuda', seed=42,
    tau0=0.6, tau_step=0.02, sigma=0.2, field_lr=0.15, patch_area=0.01, full_weight=0.1,
):
    set_environment(seed)
    init_patch = image_tensor.clone().detach().to(device)
    print("patch:", torch.min(init_patch), torch.max(init_patch), init_patch.shape)
    print("ori:", torch.min(image_tensor), torch.max(image_tensor), image_tensor.shape)
    print("tgt:", torch.min(tgt_tensor), torch.max(tgt_tensor), tgt_tensor.shape)
    momentum = torch.zeros_like(init_patch).to(device)
    delta = torch.zeros_like(init_patch, requires_grad=True).to(device)
    _, _, H, W = image_tensor.shape
    patch_generator = DynamicPatchGenerator(image_size=(W, H), learning_rate=field_lr, sigma=sigma, device=device).to(device)
    threshold_list = [round(tau0 + i * tau_step, 6) for i in range(num_iters)]
    mask_expanded = torch.ones([1, 1, H, W]).to(device)
    trans = My_T(num_copies=1)
    th = int(H * W * patch_area)  # 120*120 of 1600x900
    with torch.no_grad():
        ensemble_loss.set_full_feature(tgt_tensor)  # constant target
    for iter in range(num_iters):
        with torch.no_grad():
            tgt = target_crop(tgt_tensor)
            ensemble_loss.set_ground_truth(tgt)

        adv_patch = init_patch + delta
        if mask_expanded.sum() > th:
            mask_expanded = patch_generator(center, threshold_list[iter])
        adv_image = image_tensor * (1 - mask_expanded) + adv_patch * mask_expanded

        local = torch.cat([trans.transform(source_crop(adv_image)) for _ in range(1)])
        features, features_local = ensemble_extractor(local)
        full = ensemble_extractor.encode_img(adv_image)
        loss_a = ensemble_loss(features, features_local, full, full_weight=full_weight)
        loss = loss_a

        growing = bool(mask_expanded.sum() > th)
        if growing:
            grads = torch.autograd.grad(loss, [delta, mask_expanded], create_graph=False)
            grad = grads[0]
            grad_mask = grads[1]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy(), "mask_grads:", grad_mask.mean().detach().cpu().numpy())
        else:
            grads = torch.autograd.grad(loss, [delta], create_graph=False)
            grad = grads[0]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy())

        momentum = decay * momentum + grad
        delta.data = torch.clamp(
            delta + alpha * torch.sign(momentum),
            min=-epsilon,
            max=epsilon,
        )
        if growing:
            with torch.no_grad():
                patch_generator.update_potential_field(grad_mask)

    final_patch = init_patch + delta
    final_image = image_tensor * (1 - mask_expanded) + final_patch * mask_expanded
    final_image = torch.clamp(final_image/255.0, 0.0, 1.0)
    return final_image.detach()
