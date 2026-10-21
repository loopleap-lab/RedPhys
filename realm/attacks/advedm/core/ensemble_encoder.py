"""Multi-CLIP surrogate ensemble with differentiable feature extraction for black-box transfer attacks."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from ...surrogates import UNIFIED_SURROGATES

# CLIP normalization (OpenAI CLIP and most open_clip)
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# SigLIP uses different normalization (0.5, 0.5, 0.5)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

_COS_EPS = 1e-6


@functools.lru_cache(maxsize=None)
def _open_clip_tokenizer(name: str):
    import open_clip
    return open_clip.get_tokenizer(name)


@functools.lru_cache(maxsize=None)
def _norm(mean: tuple, std: tuple, device, dtype) -> tuple:
    """(mean, std) as [1, 3, 1, 1] tensors."""
    return (torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1),
            torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1))


@dataclass
class CLIPModelSpec:
    """Specification for a single CLIP surrogate model."""
    name: str
    source: str
    pretrained: str = ""
    image_size: int = 224
    norm_mean: Tuple[float, ...] = CLIP_MEAN
    norm_std: Tuple[float, ...] = CLIP_STD


PAPER_ENSEMBLE: List[CLIPModelSpec] = [
    CLIPModelSpec(
        name="ViT-L/14",
        source="openai_clip",
    ),
    CLIPModelSpec(
        name="ViT-B-32",
        source="open_clip",
        pretrained="laion2b_s34b_b79k",
    ),
    # ConvNeXt excluded: incompatible stem; ADVEDM needs ViT-style patch tokens.
    CLIPModelSpec(
        name="ViT-bigG-14",
        source="open_clip",
        pretrained="laion2b_s39b_b160k",
    ),
    CLIPModelSpec(
        name="ViT-SO400M-14-SigLIP",
        source="open_clip",
        pretrained="webli",
        norm_mean=SIGLIP_MEAN,
        norm_std=SIGLIP_STD,
    ),
]


UNIFIED_ENSEMBLE: List[CLIPModelSpec] = [
    CLIPModelSpec(name=m["clip"], source="openai_clip") if m["clip"]
    else CLIPModelSpec(name=m["open_clip"][0], source="open_clip", pretrained=m["open_clip"][1])
    for m in UNIFIED_SURROGATES
]

@dataclass
class SurrogateModel:
    """A loaded surrogate model with architecture metadata."""
    spec: CLIPModelSpec
    vision_encoder: torch.nn.Module
    image_size: int
    patch_size: int
    grid_size: int
    num_patches: int
    device: torch.device
    is_openai_clip: bool
    has_cls: bool = True
    _full_model: object = None


def _openai_style(sm: "SurrogateModel") -> bool:
    """OpenAI-CLIP-style tower (conv1 stem, resblocks)."""
    return sm.is_openai_clip or hasattr(getattr(sm.vision_encoder, "trunk", sm.vision_encoder), "conv1")


class EnsembleEncoder:
    """Load and manage an ensemble of CLIP surrogates for black-box transfer."""

    def __init__(
        self,
        specs: List[CLIPModelSpec] = None,
        device: str = "cuda",
    ):
        self.specs = specs or PAPER_ENSEMBLE
        self.device = torch.device(device)
        self.surrogates: List[SurrogateModel] = []

    def load_all(self) -> None:
        """Load all surrogate models."""
        for spec in self.specs:
            sm = self._load_one(spec)
            self.surrogates.append(sm)
            print(f"  Loaded {spec.name}: {sm.image_size}px, {sm.grid_size}x{sm.grid_size}={sm.num_patches} patches")
        print(f"Ensemble: {len(self.surrogates)} surrogates loaded on {self.device}")

    def _load_one(self, spec: CLIPModelSpec) -> SurrogateModel:
        """Load a single surrogate model."""
        if spec.source == "openai_clip":
            return self._load_openai_clip(spec)
        elif spec.source == "open_clip":
            return self._load_open_clip(spec)
        else:
            raise ValueError(f"Unknown source: {spec.source}")

    def _load_openai_clip(self, spec: CLIPModelSpec) -> SurrogateModel:
        """Load an OpenAI CLIP model (ViT-L/14, ViT-B/32, etc.)."""
        import clip

        model, _ = clip.load(spec.name, device=self.device)
        model = model.float().eval()
        for p in model.parameters():
            p.requires_grad_(False)

        vis = model.visual
        patch_size = int(vis.conv1.kernel_size[0])
        grid_size = spec.image_size // patch_size
        num_patches = grid_size * grid_size

        return SurrogateModel(
            spec=spec,
            vision_encoder=vis,
            image_size=spec.image_size,
            patch_size=patch_size,
            grid_size=grid_size,
            num_patches=num_patches,
            device=self.device,
            is_openai_clip=True,
            has_cls=True,
            _full_model=model,
        )

    def _load_open_clip(self, spec: CLIPModelSpec) -> SurrogateModel:
        """Load an open_clip model (ViT-G-14, SigLIP-384, etc.)."""
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            spec.name, pretrained=spec.pretrained, device=self.device,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        vis = model.visual
        # open_clip stores trunk as vis.trunk for timm-based models.
        trunk = getattr(vis, "trunk", vis)

        if hasattr(trunk, "patch_embed"):
            pe = trunk.patch_embed
            if hasattr(pe, "proj"):
                patch_size = int(pe.proj.kernel_size[0])
            elif hasattr(pe, "patch_size"):
                ps = pe.patch_size
                patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
            else:
                raise ValueError(f"Cannot detect patch_size for {spec.name}")
        elif hasattr(trunk, "conv1"):
            patch_size = int(trunk.conv1.kernel_size[0])
        else:
            raise ValueError(f"Cannot detect patch_embed for {spec.name}")

        grid_size = spec.image_size // patch_size
        num_patches = grid_size * grid_size
        _has_cls = hasattr(trunk, "class_embedding") or getattr(trunk, "cls_token", None) is not None

        return SurrogateModel(
            spec=spec,
            vision_encoder=vis,
            image_size=spec.image_size,
            patch_size=patch_size,
            grid_size=grid_size,
            num_patches=num_patches,
            device=self.device,
            is_openai_clip=False,
            has_cls=_has_cls,
            _full_model=model,
        )

    @staticmethod
    def resize(image: torch.Tensor, surrogate: SurrogateModel) -> torch.Tensor:
        """Squash to the surrogate's size."""
        sz = surrogate.image_size
        if image.shape[2] == sz and image.shape[3] == sz:
            return image
        return F.interpolate(image, size=(sz, sz), mode="bilinear", align_corners=False, antialias=max(image.shape[2:]) > sz)

    def resize_and_normalize(
        self,
        image: torch.Tensor,
        surrogate: SurrogateModel,
    ) -> torch.Tensor:
        """Resize and CLIP-normalize (differentiable)."""
        x = self.resize(image, surrogate)

        mean, std = _norm(surrogate.spec.norm_mean, surrogate.spec.norm_std, image.device, image.dtype)
        return (x - mean) / std

    def extract_features(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract normalized patch and CLS embeddings from a normalized image."""
        if _openai_style(surrogate):
            return self._extract_features_openai(surrogate, image_normalized)
        return self._extract_features_open_clip(surrogate, image_normalized)

    def _extract_features_openai(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """OpenAI CLIP: conv1 → class_embedding → transformer → ln_post."""
        from .attention_utils import openai_tokens
        vis = surrogate.vision_encoder
        x = openai_tokens(vis, image_normalized)
        seq_first = not getattr(vis.transformer, "batch_first", False)
        if seq_first:
            x = x.permute(1, 0, 2)
        x = vis.transformer(x)
        if seq_first:
            x = x.permute(1, 0, 2)
        x = vis.ln_post(x)
        x = x.float()

        cls_emb = F.normalize(x[:, 0, :], dim=-1, eps=_COS_EPS)
        patches = F.normalize(x[:, 1:, :], dim=-1, eps=_COS_EPS)
        return patches, cls_emb

    def _extract_features_open_clip(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """open_clip / timm ViT: patch_embed -> blocks -> norm."""
        vis = surrogate.vision_encoder
        trunk = getattr(vis, "trunk", vis)

        x = trunk.patch_embed(image_normalized)

        if hasattr(trunk, "cls_token") and trunk.cls_token is not None:
            cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            has_cls = True
        else:
            has_cls = False

        if hasattr(trunk, "pos_embed") and trunk.pos_embed is not None:
            x = x + trunk.pos_embed
        elif hasattr(trunk, "positional_embedding") and trunk.positional_embedding is not None:
            x = x + trunk.positional_embedding

        if hasattr(trunk, "patch_drop"):
            x = trunk.patch_drop(x)
        if hasattr(trunk, "norm_pre"):
            x = trunk.norm_pre(x)

        if hasattr(trunk, "blocks"):
            for blk in trunk.blocks:
                x = blk(x)
        elif hasattr(trunk, "resblocks"):
            seq_first = not getattr(trunk, "batch_first", False)
            if seq_first:
                x = x.permute(1, 0, 2)
            for blk in trunk.resblocks:
                x = blk(x)
            if seq_first:
                x = x.permute(1, 0, 2)

        if hasattr(trunk, "norm"):
            x = trunk.norm(x)
        elif hasattr(trunk, "ln_post"):
            x = trunk.ln_post(x)

        x = x.float()

        if has_cls:
            cls_emb = F.normalize(x[:, 0, :], dim=-1, eps=_COS_EPS)
            patches = F.normalize(x[:, 1:, :], dim=-1, eps=_COS_EPS)
        else:
            pooled = vis.head(trunk.forward_head(x)) if hasattr(trunk, "forward_head") else x.mean(dim=1)
            cls_emb = F.normalize(pooled.float(), dim=-1, eps=_COS_EPS)
            patches = F.normalize(x, dim=-1, eps=_COS_EPS)

        return patches, cls_emb

    def extract_features_and_attention(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Features and detached attention; one forward for OpenAI-CLIP towers."""
        vis = surrogate.vision_encoder
        if not _openai_style(surrogate):
            patches, cls_emb = self.extract_features(surrogate, image_normalized)
            with torch.no_grad():
                return patches, cls_emb, self.extract_attention(surrogate, image_normalized)
        from .attention_utils import attention_weights
        blocks, inputs = vis.transformer.resblocks, []
        hooks = [b.register_forward_pre_hook(lambda _m, args: inputs.append(args[0].detach())) for b in blocks]
        try:
            patches, cls_emb = self._extract_features_openai(surrogate, image_normalized)
        finally:
            for h in hooks:
                h.remove()
        to_seq_first = (lambda x: x.transpose(0, 1)) if getattr(vis.transformer, "batch_first", False) else (lambda x: x)
        with torch.no_grad():
            attn = torch.stack([attention_weights(b, to_seq_first(x))[0].mean(dim=1)[:, 0, 1:] for b, x in zip(blocks, inputs)])
        return patches, cls_emb, attn.mean(dim=0)

    def extract_attention(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Extract CLS->patch attention as a [B, N] vector."""
        if _openai_style(surrogate):
            from .attention_utils import extract_cls_to_patch_attention
            return extract_cls_to_patch_attention(surrogate.vision_encoder, image_normalized)
        return self._extract_attention_open_clip(surrogate, image_normalized)

    def _extract_attention_open_clip(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Extract CLS->patch attention for open_clip models via manual QKV decomposition."""
        vis = surrogate.vision_encoder
        trunk = getattr(vis, "trunk", vis)

        x = trunk.patch_embed(image_normalized)

        has_cls = hasattr(trunk, "cls_token") and trunk.cls_token is not None
        if has_cls:
            cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        if hasattr(trunk, "pos_embed") and trunk.pos_embed is not None:
            x = x + trunk.pos_embed
        elif hasattr(trunk, "positional_embedding") and trunk.positional_embedding is not None:
            x = x + trunk.positional_embedding

        if hasattr(trunk, "patch_drop"):
            x = trunk.patch_drop(x)
        if hasattr(trunk, "norm_pre"):
            x = trunk.norm_pre(x)

        if not has_cls:
            B = x.shape[0]
            N = surrogate.num_patches
            return torch.ones(B, N, device=x.device, dtype=x.dtype) / N

        all_cls_attn = []
        if hasattr(trunk, "blocks"):
            for blk in trunk.blocks:
                attn_mod = blk.attn
                x_norm = blk.norm1(x)

                B_size, seq_len, dim = x_norm.shape
                if hasattr(attn_mod, "qkv"):
                    qkv = attn_mod.qkv(x_norm)
                    num_heads = attn_mod.num_heads
                    head_dim = dim // num_heads
                    qkv = qkv.reshape(B_size, seq_len, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    q, k = qkv[0], qkv[1]
                else:
                    x = blk(x)
                    continue

                scale = head_dim ** -0.5
                attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_avg = attn_weights.mean(dim=1)
                cls_to_patches = attn_avg[:, 0, 1:]
                all_cls_attn.append(cls_to_patches)

                x = blk(x)

        if all_cls_attn:
            return torch.stack(all_cls_attn, dim=0).mean(dim=0)
        else:
            B = image_normalized.shape[0]
            N = surrogate.num_patches
            return torch.ones(B, N, device=image_normalized.device) / N

    def encode_text(
        self,
        surrogate: SurrogateModel,
        texts: List[str],
    ) -> torch.Tensor:
        """Encode text using surrogate's text encoder. Returns [N, D] normalized."""
        if surrogate.is_openai_clip:
            import clip
            tokens = clip.tokenize(texts)
        else:
            tokens = _open_clip_tokenizer(surrogate.spec.name)(texts)
        with torch.no_grad():
            return F.normalize(surrogate._full_model.encode_text(tokens.to(surrogate.device)).float(), dim=-1, eps=_COS_EPS)

    def project_patches(
        self,
        surrogate: SurrogateModel,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        """Project patch features to the model's output embedding space and L2-normalize."""
        vis = surrogate.vision_encoder
        proj = getattr(vis, "proj", None)
        if proj is not None:
            projected = patches @ proj.to(patches.dtype)
        else:
            projected = patches
        return F.normalize(projected, dim=-1, eps=_COS_EPS)

    def clip_bbox_from_attention(
        self,
        image_tensor: torch.Tensor,
        region_pixels: int = 100,
        source_size: int = 224,
    ) -> Tuple[int, int, int, int]:
        """Find a low-attention background region for semantic injection, returning an (x, y, w, h) bbox."""
        sm = next((s for s in self.surrogates if s.has_cls), None)
        if sm is None:
            raise RuntimeError("clip_bbox_from_attention: no surrogate with a CLS token (e.g. ViT-L/14) in the ensemble")
        with torch.no_grad():
            A = self.extract_attention(sm, self.resize_and_normalize(image_tensor, sm))
        A_map = A.squeeze(0).cpu().float().reshape(sm.grid_size, sm.grid_size)
        k = max(1, min(round(region_pixels / sm.patch_size), sm.grid_size))
        sums = A_map.unfold(0, k, 1).unfold(1, k, 1).sum((-1, -2))  # every k x k window; argmin: the first minimum
        best_r, best_c = divmod(int(sums.argmin()), sm.grid_size - k + 1)
        scale = source_size / sm.image_size
        w = h = max(1, int(k * sm.patch_size * scale))
        return int(best_c * sm.patch_size * scale), int(best_r * sm.patch_size * scale), w, h

    @staticmethod
    def bbox_to_target_indices(
        bbox_pixels: Tuple[int, int, int, int],
        surrogate: SurrogateModel,
        source_image_size: int,
    ) -> torch.Tensor:
        """Convert a pixel bbox (x, y, w, h) to patch indices for a given surrogate."""
        x, y, w, h = bbox_pixels
        gs = surrogate.grid_size
        ps = surrogate.patch_size
        sz = surrogate.image_size

        scale = sz / source_image_size
        x_s = int(x * scale)
        y_s = int(y * scale)
        w_s = max(1, int(w * scale))
        h_s = max(1, int(h * scale))

        x_s = max(0, min(x_s, sz - 1))
        y_s = max(0, min(y_s, sz - 1))
        w_s = min(w_s, sz - x_s)
        h_s = min(h_s, sz - y_s)

        px_start = x_s // ps
        py_start = y_s // ps
        px_end = min((x_s + w_s + ps - 1) // ps, gs)
        py_end = min((y_s + h_s + ps - 1) // ps, gs)

        return (torch.arange(py_start, py_end)[:, None] * gs + torch.arange(px_start, px_end)).flatten()
