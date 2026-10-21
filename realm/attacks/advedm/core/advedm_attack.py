"""Paper-exact ADVEDM-A/-R attacks with attention reallocation and attention-weighted losses."""

import torch
import torch.nn.functional as F
from typing import Tuple
import torchvision.transforms as transforms

from ....data import load_rgb
from .attention_utils import (
    openai_tokens,
    reallocate_attention_vector,
    compute_attention_weighted_features_vector,
)

_COS_EPS = 1e-6


def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable normalization used by cosine-style losses."""
    return F.normalize(x, dim=-1, eps=_COS_EPS)


def _cosine_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity with stable epsilon."""
    return F.cosine_similarity(a, b, dim=-1, eps=_COS_EPS).mean()


def _cosine_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One cosine over the region (Eq. 7/11/12)."""
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0, eps=_COS_EPS)


class ADVEDMSemanticAdditionPaperExact:
    """ADVEDM-A semantic addition attack (paper Equations 8-12)."""

    def __init__(
        self,
        vision_encoder: torch.nn.Module,
        reference_image_path: str,
        device: str = "cuda",
        lambda_cls: float = 0.8,
        lambda_local: float = 2.0,
        lambda_fix: float = 0.3,
        alpha: float = 0.5,
        beta: float = 0.4,
        norm_mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        norm_std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
        image_size: int = 224,
    ):
        """Initialize the paper-exact semantic addition attack."""
        self.device = device
        self.vision_encoder = vision_encoder
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.image_size = image_size

        self.lambda_cls = lambda_cls
        self.lambda_local = lambda_local
        self.lambda_fix = lambda_fix
        self.alpha = alpha
        self.beta = beta

        self.reference_cls_embed = self._load_reference_image(reference_image_path)
        self.reference_patch_embeds = self.reference_attention = None  # per sample, from the composite (wrapper)
        print(f"Reference {reference_image_path}: CLS {tuple(self.reference_cls_embed.shape)}")

    def _load_reference_image(self, path: str) -> torch.Tensor:
        """The reference image's normalized CLS embedding."""
        ref_image = load_rgb(path)  # EXIF-upright, as the composite's reference

        image_size = self.image_size

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        ref_tensor = transform(ref_image).unsqueeze(0).to(self.device)

        mean = torch.tensor(self.norm_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.norm_std, device=self.device).view(1, 3, 1, 1)
        ref_normalized = (ref_tensor - mean) / std

        vis = self.vision_encoder
        trunk = getattr(vis, "trunk", vis)
        is_openai_style = hasattr(trunk, "conv1")

        with torch.no_grad():
            if is_openai_style:
                x = openai_tokens(trunk, ref_normalized)
                seq_first = not getattr(trunk.transformer, "batch_first", False)
                if seq_first:
                    x = x.permute(1, 0, 2)
                x = trunk.transformer(x)
                if seq_first:
                    x = x.permute(1, 0, 2)
                x = trunk.ln_post(x)
                x = x.float()

                cls_embed = x[:, 0, :]
            else:
                x = trunk.patch_embed(ref_normalized)
                has_cls = hasattr(trunk, "cls_token") and trunk.cls_token is not None
                if has_cls:
                    cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat([cls_token, x], dim=1)
                if hasattr(trunk, "pos_embed") and trunk.pos_embed is not None:
                    x = x + trunk.pos_embed
                if hasattr(trunk, "patch_drop"):
                    x = trunk.patch_drop(x)
                if hasattr(trunk, "norm_pre"):
                    x = trunk.norm_pre(x)
                for blk in trunk.blocks:
                    x = blk(x)
                if hasattr(trunk, "norm"):
                    x = trunk.norm(x)
                x = x.float()

                cls_embed = x[:, 0, :] if has_cls else vis.head(trunk.forward_head(x)).float()

        return _safe_normalize(cls_embed).detach()

    def compute_cls_loss(
        self,
        cls_embed_adv: torch.Tensor,
        cls_embed_orig: torch.Tensor
    ) -> torch.Tensor:
        """L_cls (Eq.9): CLS fusion loss between clean and reference images."""
        cls_adv_norm = _safe_normalize(cls_embed_adv)
        cls_orig_norm = _safe_normalize(cls_embed_orig)
        cls_ref_norm = _safe_normalize(self.reference_cls_embed.to(cls_embed_adv.dtype))

        cls_fused = (1 - self.alpha) * cls_orig_norm + self.alpha * cls_ref_norm
        cls_fused_norm = _safe_normalize(cls_fused)

        similarity = F.cosine_similarity(
            cls_adv_norm, cls_fused_norm, dim=-1, eps=_COS_EPS
        )

        return -similarity.mean()

    def compute_local_injection_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        A_adv: torch.Tensor,
        A_reallocated: torch.Tensor,
        target_indices: torch.Tensor
    ) -> torch.Tensor:
        """L_p (Eq.11): local injection loss over injection patches using attention-weighted features."""
        B = patch_embeds_adv.shape[0]
        if target_indices.dim() == 1:
            target_indices = target_indices.unsqueeze(0)
        if target_indices.shape[0] == 1 and B > 1:
            target_indices = target_indices.expand(B, -1)
        if target_indices.shape[0] != B:
            raise ValueError(
                f"target_indices batch size mismatch: got {target_indices.shape[0]}, expected {B}"
            )
        if target_indices.shape[1] == 0:
            raise ValueError("target_indices is empty; cannot compute injection loss.")
        injection_losses = []

        for b in range(B):
            weighted_adv = compute_attention_weighted_features_vector(
                A_adv[b:b+1],
                patch_embeds_adv[b:b+1]
            )

            ref_patches = self.reference_patch_embeds.to(patch_embeds_adv.dtype)
            weighted_ref = compute_attention_weighted_features_vector(
                A_reallocated[b:b+1],
                ref_patches
            )

            target_weighted_adv = weighted_adv[0, target_indices[b]]
            target_weighted_ref = weighted_ref[0, target_indices[b]]
            if target_weighted_adv.shape[0] == 0:
                raise ValueError("No injection patches selected for local injection loss.")

            patch_sim = _cosine_flat(target_weighted_adv, target_weighted_ref)

            injection_losses.append(-patch_sim)

        return torch.stack(injection_losses).mean()

    def compute_attention_fixation_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        patch_embeds_orig: torch.Tensor,
        A_adv: torch.Tensor,
        A_reallocated: torch.Tensor,
        target_indices: torch.Tensor
    ) -> torch.Tensor:
        """L_fix (Eq.12): preserve non-target regions using attention-weighted features."""
        B, num_patches = patch_embeds_adv.shape[:2]
        if target_indices.dim() == 1:
            target_indices = target_indices.unsqueeze(0)
        if target_indices.shape[0] == 1 and B > 1:
            target_indices = target_indices.expand(B, -1)
        if target_indices.shape[0] != B:
            raise ValueError(
                f"target_indices batch size mismatch: got {target_indices.shape[0]}, expected {B}"
            )
        if target_indices.shape[1] == 0:
            raise ValueError("target_indices is empty; cannot compute fixation loss.")
        fixation_losses = []

        for b in range(B):
            weighted_adv = compute_attention_weighted_features_vector(
                A_adv[b:b+1],
                patch_embeds_adv[b:b+1]
            )[0]

            weighted_orig = compute_attention_weighted_features_vector(
                A_reallocated[b:b+1],
                patch_embeds_orig[b:b+1]
            )[0]

            mask = torch.ones(num_patches, dtype=torch.bool, device=patch_embeds_adv.device)
            mask[target_indices[b]] = False

            non_target_weighted_adv = weighted_adv[mask]
            non_target_weighted_orig = weighted_orig[mask]
            if non_target_weighted_adv.shape[0] == 0:
                raise ValueError("All patches are marked as target; no non-target patches to preserve.")

            similarity = _cosine_flat(non_target_weighted_adv, non_target_weighted_orig)

            fixation_losses.append(-similarity)

        return torch.stack(fixation_losses).mean()

    def compute_total_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        cls_embed_adv: torch.Tensor,
        cls_embed_orig: torch.Tensor,
        patch_embeds_orig: torch.Tensor,
        A_adv: torch.Tensor,
        A_orig: torch.Tensor,
        target_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the Eq.8 total loss (w1*L_cls + w2*L_p + w3*L_fix) with attention reallocation."""
        mask = torch.ones_like(A_orig)  # [1, N]; bbox_to_target_indices gives >= 1 in-range index
        mask[:, target_indices] = 0
        A_ref = self.reference_attention.to(A_orig.dtype)

        A_reallocated = reallocate_attention_vector(
            A_orig,
            A_ref,
            mask,
            beta=self.beta
        )

        L_cls = self.compute_cls_loss(cls_embed_adv, cls_embed_orig)

        L_p = self.compute_local_injection_loss(
            patch_embeds_adv,
            A_adv,
            A_reallocated,
            target_indices
        )

        L_fix = self.compute_attention_fixation_loss(
            patch_embeds_adv,
            patch_embeds_orig,
            A_adv,
            A_reallocated,
            target_indices
        )

        total = (
            self.lambda_cls * L_cls +
            self.lambda_local * L_p +
            self.lambda_fix * L_fix
        )

        return total, {
            "total": total.detach(),
            "cls": L_cls.detach(),
            "reference_sim": L_p.detach(),
            "attention_fix": L_fix.detach(),
        }


class ADVEDMSemanticRemovalPaperExact:
    """ADVEDM-R semantic removal attack via text-guided masking (paper Equations 3-8)."""

    def __init__(self, target_text_embed: torch.Tensor, lambda_cls: float = 0.5, lambda_local: float = 2.0, lambda_fix: float = 0.2):
        """target_text_embed: [1, D] text embedding of the concept to remove."""
        self.target_text_embed = target_text_embed
        self.lambda_cls = lambda_cls
        self.lambda_local = lambda_local
        self.lambda_fix = lambda_fix

    def compute_cls_loss(
        self,
        cls_embed_adv: torch.Tensor
    ) -> torch.Tensor:
        """L_cls (Eq.5): positive similarity to the target text, minimized to remove semantics."""
        cls_norm = _safe_normalize(cls_embed_adv)
        text_norm = _safe_normalize(self.target_text_embed.to(cls_embed_adv.dtype))

        similarity = F.cosine_similarity(cls_norm, text_norm, dim=-1, eps=_COS_EPS)

        return similarity.mean()

    def compute_local_removal_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        patch_embeds_masked: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """L_p (Eq.6): maximize similarity to masked patches in the removal region."""
        removal_losses = []
        B = patch_embeds_adv.shape[0]

        for b in range(B):
            removal_indices = (mask[b] == 0).nonzero(as_tuple=True)[0]
            if removal_indices.numel() == 0:
                raise ValueError("No removal patches found; adjust k_ratio.")

            adv_removal = patch_embeds_adv[b, removal_indices]
            masked_removal = patch_embeds_masked[b, removal_indices]

            similarity = _cosine_mean(adv_removal, masked_removal)

            removal_losses.append(-similarity)

        return torch.stack(removal_losses).mean()

    def compute_attention_fixation_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        patch_embeds_orig: torch.Tensor,
        A_adv: torch.Tensor,
        A_orig: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """L_fix (Eq.7): preserve non-removal regions using clean attention weighting (no reallocation)."""
        fixation_losses = []
        B = patch_embeds_adv.shape[0]

        for b in range(B):
            weighted_adv = compute_attention_weighted_features_vector(
                A_adv[b:b+1], patch_embeds_adv[b:b+1]
            )[0]

            weighted_orig = compute_attention_weighted_features_vector(
                A_orig[b:b+1], patch_embeds_orig[b:b+1]
            )[0]

            preserve_indices = (mask[b] == 1).nonzero(as_tuple=True)[0]
            if preserve_indices.numel() == 0:
                raise ValueError("No preserve patches found; adjust k_ratio.")

            preserve_adv = weighted_adv[preserve_indices]
            preserve_orig = weighted_orig[preserve_indices]

            similarity = _cosine_flat(preserve_adv, preserve_orig)

            fixation_losses.append(-similarity)

        return torch.stack(fixation_losses).mean()

    def compute_total_loss(
        self,
        cls_embed_adv: torch.Tensor,
        patch_embeds_adv: torch.Tensor,
        patch_embeds_orig: torch.Tensor,
        patch_embeds_masked: torch.Tensor,
        A_adv: torch.Tensor,
        A_orig: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the Eq.8 total loss (w1*L_cls + w2*L_p + w3*L_fix)."""
        L_cls = self.compute_cls_loss(cls_embed_adv)

        L_p = self.compute_local_removal_loss(
            patch_embeds_adv, patch_embeds_masked, mask
        )

        L_fix = self.compute_attention_fixation_loss(
            patch_embeds_adv, patch_embeds_orig, A_adv, A_orig, mask
        )

        total_loss = (
            self.lambda_cls * L_cls +
            self.lambda_local * L_p +
            self.lambda_fix * L_fix
        )

        return total_loss, {
            'total': total_loss.detach(),
            'cls': L_cls.detach(),
            'local': L_p.detach(),
            'fix': L_fix.detach()
        }
