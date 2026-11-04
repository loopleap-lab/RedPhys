"""Text-guided mask construction and pixel-space masking for ADVEDM-R."""

import torch
import torch.nn.functional as F

_COS_EPS = 1e-6


def compute_text_patch_similarity(patch_embeds: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between each patch and the target text."""
    patch_norm = F.normalize(patch_embeds, dim=-1, eps=_COS_EPS)
    text_norm = F.normalize(text_embed.to(patch_embeds.dtype), dim=-1, eps=_COS_EPS)
    return torch.matmul(patch_norm, text_norm.T).squeeze(-1)


def construct_top_k_mask(similarity: torch.Tensor, k_ratio: float = 0.2) -> torch.Tensor:
    """0 on the top-k most text-similar patches (Eq. 4 with xi between the k-th and k+1-th)."""
    if not 0.0 < k_ratio <= 1.0:
        raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}")
    B, N = similarity.shape
    k = max(1, min(int(round(N * k_ratio)), N))
    mask = torch.ones(B, N, device=similarity.device)
    for b in range(B):
        mask[b, torch.topk(similarity[b], k, largest=True).indices] = 0.0
    return mask


def create_masked_image(image: torch.Tensor, mask: torch.Tensor, patch_size: int, grid_size: int) -> torch.Tensor:
    """Zero the pixels of removal patches (mask=0); grid_size * patch_size must be the image size."""
    keep = F.interpolate(mask.view(-1, 1, grid_size, grid_size).to(image.dtype), scale_factor=patch_size, mode="nearest")
    if keep.shape[-2:] != image.shape[-2:]:
        raise ValueError(f"{grid_size}x{patch_size} patches do not tile the {tuple(image.shape[-2:])} image")
    return image * keep
