"""ADVEDM-A/R semantic addition/removal attacks (SSA-CWA + CLIP ensemble)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from ..base_attack import AttackConfig, AttackResult, BaseAttack, pixel_linf, to_pil, to_tensor
from ...data import Sample


@dataclass
class ADVEDMConfig(AttackConfig):
    epsilon: float = 16.0
    num_iters: int = 30
    inner_step_size: float = 250.0
    ksi: float = 3.2
    mu: float = 1.0
    ssa_N: int = 20
    ssa_sigma: float = 16.0
    ssa_rho: float = 0.5

    lambda_cls: float = 0.8
    lambda_local: float = 2.0
    lambda_fix: float = 0.3
    cls_fusion_alpha: float = 0.5
    attention_beta: float = 0.4


    reference_image_path: Optional[str] = None
    target_images_dir: Optional[str] = None
    annotations_file: Optional[str] = None
    region_size: int = 100
    image_size: int = 224  # in the image squashed to image_size


@dataclass
class ADVEDMRConfig(AttackConfig):
    epsilon: float = 16.0
    num_iters: int = 30
    inner_step_size: float = 250.0
    ksi: float = 3.2
    mu: float = 1.0
    ssa_N: int = 20
    ssa_sigma: float = 16.0
    ssa_rho: float = 0.5

    lambda_cls: float = 0.5
    lambda_local: float = 2.0
    lambda_fix: float = 0.2
    k_ratio: float = 0.2


    target_text: Optional[str] = None


class ADVEDMAttack(BaseAttack):
    def __init__(self, config: ADVEDMConfig):
        super().__init__(config)
        self.config: ADVEDMConfig = config

        self._ensemble = None
        self._attack_objs = None
        self._current_ref: Optional[str] = None
        self._annotations: Optional[Dict[str, Any]] = None

    def _resolve_reference(self, sample_id: str) -> str:
        if self.config.target_images_dir:
            d = Path(self.config.target_images_dir)
            for ext in (".jpg", ".png", ".jpeg"):
                p = d / f"{sample_id}{ext}"
                if p.exists():
                    return str(p)
        if self.config.reference_image_path:
            return self.config.reference_image_path
        raise ValueError(
            f"No reference image found for sample '{sample_id}'. "
            "Set reference_image_path or target_images_dir."
        )

    def _initialize(self, reference_path: str) -> None:
        if self._ensemble is not None and self._current_ref == reference_path:
            return

        from .core.ensemble_encoder import EnsembleEncoder, PAPER_ENSEMBLE, UNIFIED_ENSEMBLE

        if self._ensemble is None:
            print("Loading CLIP ensemble for ADVEDM-A...")
            self._ensemble = EnsembleEncoder(
                specs=UNIFIED_ENSEMBLE if self.config.surrogates == "unified" else PAPER_ENSEMBLE,
                device=self.config.device,
            )
            self._ensemble.load_all()

        from .core.advedm_attack import ADVEDMSemanticAdditionPaperExact

        self._attack_objs = []
        for sm in self._ensemble.surrogates:
            atk = ADVEDMSemanticAdditionPaperExact(
                vision_encoder=sm.vision_encoder,
                reference_image_path=reference_path,
                device=self.config.device,
                lambda_cls=self.config.lambda_cls,
                lambda_local=self.config.lambda_local,
                lambda_fix=self.config.lambda_fix,
                alpha=self.config.cls_fusion_alpha,
                beta=self.config.attention_beta,
                norm_mean=sm.spec.norm_mean,
                norm_std=sm.spec.norm_std,
                image_size=sm.image_size,
            )
            self._attack_objs.append(atk)
        self._current_ref = reference_path
        self._ref_tensor = to_tensor(Image.open(reference_path)).to(self.config.device)

    def _load_annotations(self) -> Optional[Dict[str, Any]]:
        if self._annotations is not None:
            return self._annotations
        if self.config.annotations_file is None:
            return None
        with open(self.config.annotations_file, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Annotation file must contain a JSON object map, got {type(loaded).__name__}"
            )
        self._annotations = loaded
        return self._annotations

    def _get_bbox(
        self,
        sample: Sample,
        image_tensor: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """Return injection bbox from annotations file, else CLIP attention auto-detection."""
        image_file = str(sample.metadata.get("image_file", ""))
        annotations = self._load_annotations()
        if annotations is not None and image_file:
            entry = annotations.get(image_file)
            if entry is None:
                image_name = Path(image_file).name
                for key, value in annotations.items():
                    if Path(key).name == image_name:
                        entry = value
                        break
            if entry is not None:
                bbox_value = entry.get("bbox") if isinstance(entry, dict) else entry
                if bbox_value is not None:
                    return tuple(int(v) for v in bbox_value)

        bbox = self._ensemble.clip_bbox_from_attention(
            image_tensor,
            region_pixels=self.config.region_size,
            source_size=self.config.image_size,
        )
        print(f"  [ADVEDM-A] Auto bbox (CLIP attention): {bbox}")
        return bbox

    def generate(
        self,
        sample: Sample,
        **kwargs,
    ) -> AttackResult:
        ref_path = self._resolve_reference(sample.id)
        self._initialize(ref_path)

        original_pil = sample.images[0] if sample.images else None
        if original_pil is None:
            raise ValueError(f"Sample '{sample.id}' has no images.")

        image_tensor = to_tensor(original_pil).to(self.config.device)
        bbox = self._get_bbox(sample, image_tensor)
        source_image_size = self.config.image_size

        ensemble = self._ensemble
        attack_objs = self._attack_objs

        clean_cache = []
        for sm, atk in zip(ensemble.surrogates, attack_objs):
            with torch.no_grad():
                img_norm = ensemble.resize_and_normalize(image_tensor, sm)
                patches_orig, cls_orig = ensemble.extract_features(sm, img_norm)
                A_orig = ensemble.extract_attention(sm, img_norm)
                target_idx = ensemble.bbox_to_target_indices(
                    bbox, sm, source_image_size
                ).to(self.config.device)
                # Eq. 10-11: R fills the injection region
                r, c = target_idx // sm.grid_size, target_idx % sm.grid_size
                y0, y1, x0, x1 = (int(v) * sm.patch_size for v in (r.min(), r.max() + 1, c.min(), c.max() + 1))
                comp = ensemble.resize(image_tensor, sm).clone()
                comp[..., y0:y1, x0:x1] = F.interpolate(self._ref_tensor, size=(y1 - y0, x1 - x0), mode="bilinear",
                                                        align_corners=False, antialias=True)
                comp = ensemble.resize_and_normalize(comp, sm)
                atk.reference_patch_embeds = ensemble.extract_features(sm, comp)[0]
                atk.reference_attention = ensemble.extract_attention(sm, comp)
            clean_cache.append({
                "sm": sm, "atk": atk,
                "patches_orig": patches_orig,
                "cls_orig": cls_orig,
                "A_orig": A_orig,
                "target_indices": target_idx,
            })

        def make_loss_fn(cache_entry, surrogate_idx):
            sm = cache_entry["sm"]
            atk = cache_entry["atk"]
            patches_orig = cache_entry["patches_orig"]
            cls_orig = cache_entry["cls_orig"]
            A_orig = cache_entry["A_orig"]
            target_idx = cache_entry["target_indices"]
            _n_calls = [0]

            def loss_fn(x_aug: torch.Tensor) -> torch.Tensor:
                img_norm = ensemble.resize_and_normalize(x_aug, sm)
                patches_adv, cls_adv, A_adv = ensemble.extract_features_and_attention(sm, img_norm)

                loss, components = atk.compute_total_loss(
                    patch_embeds_adv=patches_adv,
                    cls_embed_adv=cls_adv,
                    cls_embed_orig=cls_orig,
                    patch_embeds_orig=patches_orig,
                    A_adv=A_adv,
                    A_orig=A_orig,
                    target_indices=target_idx,
                )
                _n_calls[0] += 1
                if surrogate_idx == 0 and _n_calls[0] % self.config.ssa_N == 1:
                    outer_iter = (_n_calls[0] - 1) // self.config.ssa_N
                    print(f"  [ADVEDM-A sur0 iter={outer_iter:2d}] loss={loss.item():.4f}  "
                          f"cls={components['cls']:.4f}  ref={components['reference_sim']:.4f}  "
                          f"fix={components['attention_fix']:.4f}", flush=True)
                return -loss  # SSA ascends, so negate the minimization loss

            return loss_fn

        loss_fns = [make_loss_fn(c, i) for i, c in enumerate(clean_cache)]

        from .core.ssa_cwa import ssa_cwa_attack

        adv_tensor = ssa_cwa_attack(
            x_orig=image_tensor,
            loss_fns=loss_fns,
            num_iters=self.config.num_iters,
            epsilon=self.config.epsilon / 255.0,
            inner_step_size=self.config.inner_step_size,
            ksi=self.config.ksi / 255.0,
            mu=self.config.mu,
            N=self.config.ssa_N,
            sigma=self.config.ssa_sigma / 255.0,
            rho=self.config.ssa_rho,
            verbose=True,
        )

        adversarial_pil = to_pil(adv_tensor)

        return AttackResult(
            adversarial_sample=adversarial_pil,
            perturbation_norm=pixel_linf(adversarial_pil, to_pil(image_tensor)),
            metadata={
                "reference_image": str(Path(ref_path).resolve()),
                "constraint": "linf",
                "num_iterations": self.config.num_iters,
                "epsilon": self.config.epsilon,
                "num_surrogates": len(self._ensemble.surrogates),
                "attack_type": "advedm_a",
            },
        )


class ADVEDMRAttack(BaseAttack):
    def __init__(self, config: ADVEDMRConfig):
        super().__init__(config)
        self.config: ADVEDMRConfig = config

        self._ensemble = None
        self._attack_objs = None
        self._default_text = config.target_text

    def _initialize(self) -> None:
        if self._ensemble is None:
            from .core.ensemble_encoder import EnsembleEncoder, PAPER_ENSEMBLE, UNIFIED_ENSEMBLE

            print("Loading CLIP ensemble for ADVEDM-R...")
            self._ensemble = EnsembleEncoder(
                specs=UNIFIED_ENSEMBLE if self.config.surrogates == "unified" else PAPER_ENSEMBLE,
                device=self.config.device,
            )
            self._ensemble.load_all()

        if self._attack_objs is None:
            target_text = (self.config.target_text or "").strip()
            if not target_text:
                raise ValueError(
                    "ADVEDMRConfig.target_text is required for ADVEDM-R."
                )

            from .core.advedm_attack import ADVEDMSemanticRemovalPaperExact

            self._attack_objs = []
            for sm in self._ensemble.surrogates:
                text_enc = _SurrogateTextEncoder(self._ensemble, sm)
                atk = ADVEDMSemanticRemovalPaperExact(
                    vision_encoder=sm.vision_encoder,
                    text_encoder=text_enc,
                    target_text=target_text,
                    lambda_cls=self.config.lambda_cls,
                    lambda_local=self.config.lambda_local,
                    lambda_fix=self.config.lambda_fix,
                    k_ratio=self.config.k_ratio,
                    device=self.config.device,
                )
                self._attack_objs.append(atk)
            print(f"  Created {len(self._attack_objs)} per-surrogate ADVEDM-R attack objects.")

    def generate(
        self,
        sample: Sample,
        **kwargs,
    ) -> AttackResult:
        # ADVEDM-R removes the SOURCE object's semantics, so use attack_source_text
        text = (sample.metadata or {}).get("attack_source_text") or self._default_text
        if text != self.config.target_text:
            self.config.target_text = text
            self._attack_objs = None

        self._initialize()

        original_pil = sample.images[0] if sample.images else None
        if original_pil is None:
            raise ValueError(f"Sample '{sample.id}' has no images.")

        image_tensor = to_tensor(original_pil).to(self.config.device)

        ensemble = self._ensemble
        attack_objs = self._attack_objs

        from .core.mask_utils import (
            compute_text_patch_similarity,
            construct_top_k_mask,
            create_masked_image,
        )

        clean_cache = []
        for sm, atk in zip(ensemble.surrogates, attack_objs):
            with torch.no_grad():
                img_norm = ensemble.resize_and_normalize(image_tensor, sm)
                patches_orig, cls_orig = ensemble.extract_features(sm, img_norm)
                A_orig = ensemble.extract_attention(sm, img_norm)

                if sm.has_cls or not clean_cache:
                    patches_proj = ensemble.project_patches(sm, patches_orig)
                    S = compute_text_patch_similarity(patches_proj, atk.target_text_embed)
                    mask = construct_top_k_mask(S, k_ratio=self.config.k_ratio)
                else:
                    # SigLIP: reuse the first mask
                    m0, g0 = clean_cache[0]["mask"], clean_cache[0]["sm"].grid_size
                    mask = F.interpolate(m0.view(1, 1, g0, g0), size=(sm.grid_size,) * 2, mode="nearest").view(1, -1)

                masked_img = create_masked_image(
                    ensemble.resize(image_tensor, sm), mask,
                    patch_size=sm.patch_size, image_size=sm.image_size,
                    grid_size=sm.grid_size,
                )
                masked_norm = ensemble.resize_and_normalize(masked_img, sm)
                patches_masked, _ = ensemble.extract_features(sm, masked_norm)

            clean_cache.append({
                "sm": sm, "atk": atk,
                "patches_orig": patches_orig,
                "patches_masked": patches_masked,
                "A_orig": A_orig,
                "mask": mask,
            })

        def make_loss_fn(cache_entry, surrogate_idx):
            sm = cache_entry["sm"]
            atk = cache_entry["atk"]
            patches_orig = cache_entry["patches_orig"]
            patches_masked = cache_entry["patches_masked"]
            A_orig = cache_entry["A_orig"]
            mask = cache_entry["mask"]
            _n_calls = [0]

            def loss_fn(x_aug: torch.Tensor) -> torch.Tensor:
                img_norm = ensemble.resize_and_normalize(x_aug, sm)
                patches_adv, cls_adv, A_adv = ensemble.extract_features_and_attention(sm, img_norm)

                cls_adv_proj = ensemble.project_patches(sm, cls_adv.unsqueeze(1)).squeeze(1)

                loss, components = atk.compute_total_loss(
                    cls_embed_adv=cls_adv_proj,
                    patch_embeds_adv=patches_adv,
                    patch_embeds_orig=patches_orig,
                    patch_embeds_masked=patches_masked,
                    A_adv=A_adv,
                    A_orig=A_orig,
                    mask=mask,
                )
                _n_calls[0] += 1
                if surrogate_idx == 0 and _n_calls[0] % self.config.ssa_N == 1:
                    outer_iter = (_n_calls[0] - 1) // self.config.ssa_N
                    print(f"  [ADVEDM-R sur0 iter={outer_iter:2d}] loss={loss.item():.4f}  "
                          f"cls={components['cls']:.4f}  local={components['local']:.4f}  "
                          f"fix={components['fix']:.4f}", flush=True)
                return -loss  # SSA ascends, so negate the minimization loss

            return loss_fn

        loss_fns = [make_loss_fn(c, i) for i, c in enumerate(clean_cache)]

        from .core.ssa_cwa import ssa_cwa_attack

        adv_tensor = ssa_cwa_attack(
            x_orig=image_tensor,
            loss_fns=loss_fns,
            num_iters=self.config.num_iters,
            epsilon=self.config.epsilon / 255.0,
            inner_step_size=self.config.inner_step_size,
            ksi=self.config.ksi / 255.0,
            mu=self.config.mu,
            N=self.config.ssa_N,
            sigma=self.config.ssa_sigma / 255.0,
            rho=self.config.ssa_rho,
            verbose=True,
        )

        adversarial_pil = to_pil(adv_tensor)

        return AttackResult(
            adversarial_sample=adversarial_pil,
            perturbation_norm=pixel_linf(adversarial_pil, to_pil(image_tensor)),
            metadata={
                "removed_text": self.config.target_text,
                "constraint": "linf",
                "num_iterations": self.config.num_iters,
                "epsilon": self.config.epsilon,
                "num_surrogates": len(self._ensemble.surrogates),
                "attack_type": "advedm_r",
            },
        )


class _SurrogateTextEncoder:
    """encode_text routed to one surrogate of the ensemble."""

    def __init__(self, ensemble, surrogate):
        self._ensemble = ensemble
        self._surrogate = surrogate

    def encode_text(self, texts: list) -> torch.Tensor:
        return self._ensemble.encode_text(self._surrogate, texts)
