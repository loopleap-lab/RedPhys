"""AdvEDM-A/R: semantic addition/removal (SSA-CWA, CLIP ensemble)."""

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from .. import trace
from ..base_attack import AttackConfig, AttackResult, BaseAttack, find_by_stem, pixel_linf, to_pil, to_tensor
from ...data import Sample, load_rgb, relpath


@dataclass
class _SSAConfig(AttackConfig):  # SSA-CWA, shared by AdvEDM-A and -R
    epsilon: float = 16.0
    max_iterations: int = 30
    inner_step_size: float = 250.0
    ksi: float = 3.2
    mu: float = 1.0
    ssa_N: int = 20
    ssa_sigma: float = 16.0
    ssa_rho: float = 0.5


def _run_ssa(c: _SSAConfig, x: torch.Tensor, loss_fns: list) -> torch.Tensor:
    from .core.ssa_cwa import ssa_cwa_attack
    return ssa_cwa_attack(x_orig=x, loss_fns=loss_fns, num_iters=c.max_iterations, epsilon=c.epsilon / 255.0, inner_step_size=c.inner_step_size,
                          ksi=c.ksi / 255.0, mu=c.mu, N=c.ssa_N, sigma=c.ssa_sigma / 255.0, rho=c.ssa_rho, verbose=True)


@dataclass
class ADVEDMConfig(_SSAConfig):
    lambda_cls: float = 0.8
    lambda_local: float = 2.0
    lambda_fix: float = 0.3
    cls_fusion_alpha: float = 0.5
    attention_beta: float = 0.4
    target_images_dir: Optional[str] = None
    region_size: int = 100
    image_size: int = 224  # frame of the bbox coordinates


@dataclass
class ADVEDMRConfig(_SSAConfig):
    needs_source_text: ClassVar[bool] = True  # R removes the source object
    lambda_cls: float = 0.5
    lambda_local: float = 2.0
    lambda_fix: float = 0.2
    k_ratio: float = 0.2
    source_text: Optional[str] = None  # the object to remove; default: the labels' source text


class _ADVEDMBase(BaseAttack):
    tag = ""

    def __init__(self, config):
        super().__init__(config)
        self._ensemble = self._attack_objs = None

    def _load_ensemble(self):
        if self._ensemble is None:
            from .core.ensemble_encoder import EnsembleEncoder, PAPER_ENSEMBLE, UNIFIED_ENSEMBLE
            print(f"Loading CLIP ensemble for {self.tag}...")
            self._ensemble = EnsembleEncoder(UNIFIED_ENSEMBLE if self.config.surrogates == "unified" else PAPER_ENSEMBLE, self.config.device)
            self._ensemble.load_all()
        return self._ensemble

    def _attack(self, sample: Sample, x: torch.Tensor, cache: list, loss_of, log, **meta) -> AttackResult:
        """SSA over loss_of(c, x_aug) -> (loss, components) per surrogate; logs surrogate 0 once per outer step."""
        n = [0]

        def make(c, i):
            def fn(x_aug):
                loss, comps = loss_of(c, x_aug)
                if i == 0:
                    n[0] += 1
                    if (n[0] - 1) % self.config.ssa_N == 0:  # once per outer step: surrogate 0
                        trace.log((n[0] - 1) // self.config.ssa_N, loss=loss, **{k: v for k, v in comps.items() if k != "total" and not isinstance(v, str)})
                        print(f"  [{self.tag} sur0 iter={(n[0] - 1) // self.config.ssa_N:2d}] loss={loss.item():.4f}  {log(comps)}", flush=True)
                return -loss  # SSA ascends
            return fn

        adv = to_pil(_run_ssa(self.config, x, [make(c, i) for i, c in enumerate(cache)]))
        return AttackResult(adversarial_sample=adv, perturbation_norm=pixel_linf(adv, sample.images[0]),
                            metadata={**meta, "num_surrogates": len(self._ensemble.surrogates)})


class ADVEDMAttack(_ADVEDMBase):
    tag = "ADVEDM-A"
    config: ADVEDMConfig

    def __init__(self, config: ADVEDMConfig):
        super().__init__(config)
        self._ref = self._ref_tensor = None

    def _set_reference(self, sample_id: str) -> str:
        p = find_by_stem(self.config.target_images_dir, sample_id, required=True)
        ens, c = self._load_ensemble(), self.config
        if str(p) != self._ref:
            from .core.advedm_attack import ADVEDMSemanticAdditionPaperExact
            self._attack_objs = [ADVEDMSemanticAdditionPaperExact(
                vision_encoder=sm.vision_encoder, reference_image_path=str(p), device=c.device, lambda_cls=c.lambda_cls,
                lambda_local=c.lambda_local, lambda_fix=c.lambda_fix, alpha=c.cls_fusion_alpha, beta=c.attention_beta,
                norm_mean=sm.spec.norm_mean, norm_std=sm.spec.norm_std, image_size=sm.image_size) for sm in ens.surrogates]
            self._ref, self._ref_tensor = str(p), to_tensor(load_rgb(p)).to(c.device)
        return self._ref

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        ref_path = self._set_reference(sample.id)
        ens, c = self._ensemble, self.config
        x = to_tensor(sample.images[0]).to(c.device)
        bbox = ens.clip_bbox_from_attention(x, region_pixels=c.region_size, source_size=c.image_size)

        cache = []
        for sm, atk in zip(ens.surrogates, self._attack_objs):
            with torch.no_grad():
                img_norm = ens.resize_and_normalize(x, sm)
                patches_orig, cls_orig = ens.extract_features(sm, img_norm)
                A_orig = ens.extract_attention(sm, img_norm)
                target_idx = ens.bbox_to_target_indices(bbox, sm, c.image_size).to(c.device)
                # Eq. 10-11: R fills the injection region
                r, col = target_idx // sm.grid_size, target_idx % sm.grid_size
                y0, y1, x0, x1 = (int(v) * sm.patch_size for v in (r.min(), r.max() + 1, col.min(), col.max() + 1))
                comp = ens.resize(x, sm).clone()
                comp[..., y0:y1, x0:x1] = F.interpolate(self._ref_tensor, size=(y1 - y0, x1 - x0), mode="bilinear",
                                                        align_corners=False, antialias=True)
                comp = ens.resize_and_normalize(comp, sm)
                atk.reference_patch_embeds = ens.extract_features(sm, comp)[0]
                atk.reference_attention = ens.extract_attention(sm, comp)
            cache.append({"sm": sm, "atk": atk, "patches_orig": patches_orig, "cls_orig": cls_orig, "A_orig": A_orig,
                          "target_indices": target_idx})

        def loss_of(e, x_aug):
            patches_adv, cls_adv, A_adv = ens.extract_features_and_attention(e["sm"], ens.resize_and_normalize(x_aug, e["sm"]))
            return e["atk"].compute_total_loss(
                patch_embeds_adv=patches_adv, cls_embed_adv=cls_adv, cls_embed_orig=e["cls_orig"], patch_embeds_orig=e["patches_orig"],
                A_adv=A_adv, A_orig=e["A_orig"], target_indices=e["target_indices"])

        return self._attack(sample, x, cache, loss_of,
                            lambda d: f"cls={d['cls']:.4f}  ref={d['reference_sim']:.4f}  fix={d['attention_fix']:.4f}",
                            reference_image=relpath(ref_path))


class ADVEDMRAttack(_ADVEDMBase):
    tag = "ADVEDM-R"
    config: ADVEDMRConfig

    def __init__(self, config: ADVEDMRConfig):
        super().__init__(config)
        self._text = None

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        text = self.config.source_text or sample.metadata.get("attack_source_text")
        ens, c = self._load_ensemble(), self.config
        if self._attack_objs is None or text != self._text:
            if not (stripped := (text or "").strip()):
                raise ValueError("ADVEDM-R needs the object to remove: labels= (source text) or source_text in the YAML")
            from .core.advedm_attack import ADVEDMSemanticRemovalPaperExact
            self._attack_objs = [ADVEDMSemanticRemovalPaperExact(target_text_embed=ens.encode_text(sm, [stripped]), lambda_cls=c.lambda_cls,
                                                                 lambda_local=c.lambda_local, lambda_fix=c.lambda_fix) for sm in ens.surrogates]
            self._text = text
            print(f"  Created {len(self._attack_objs)} per-surrogate ADVEDM-R attack objects.")

        from .core.mask_utils import compute_text_patch_similarity, construct_top_k_mask, create_masked_image
        x = to_tensor(sample.images[0]).to(c.device)
        cache = []
        for sm, atk in zip(ens.surrogates, self._attack_objs):
            with torch.no_grad():
                img_norm = ens.resize_and_normalize(x, sm)
                patches_orig, _ = ens.extract_features(sm, img_norm)
                A_orig = ens.extract_attention(sm, img_norm)
                if sm.has_cls or not cache:
                    S = compute_text_patch_similarity(ens.project_patches(sm, patches_orig), atk.target_text_embed)
                    mask = construct_top_k_mask(S, k_ratio=c.k_ratio)
                else:  # SigLIP: reuse the first mask
                    m0, g0 = cache[0]["mask"], cache[0]["sm"].grid_size
                    mask = F.interpolate(m0.view(1, 1, g0, g0), size=(sm.grid_size,) * 2, mode="nearest").view(1, -1)
                masked = create_masked_image(ens.resize(x, sm), mask, patch_size=sm.patch_size, grid_size=sm.grid_size)
                patches_masked, _ = ens.extract_features(sm, ens.resize_and_normalize(masked, sm))
            cache.append({"sm": sm, "atk": atk, "patches_orig": patches_orig, "patches_masked": patches_masked, "A_orig": A_orig,
                          "mask": mask})

        def loss_of(e, x_aug):
            patches_adv, cls_adv, A_adv = ens.extract_features_and_attention(e["sm"], ens.resize_and_normalize(x_aug, e["sm"]))
            cls_adv_proj = ens.project_patches(e["sm"], cls_adv.unsqueeze(1)).squeeze(1)
            return e["atk"].compute_total_loss(
                cls_embed_adv=cls_adv_proj, patch_embeds_adv=patches_adv, patch_embeds_orig=e["patches_orig"],
                patch_embeds_masked=e["patches_masked"], A_adv=A_adv, A_orig=e["A_orig"], mask=e["mask"])

        return self._attack(sample, x, cache, loss_of,
                            lambda d: f"cls={d['cls']:.4f}  local={d['local']:.4f}  fix={d['fix']:.4f}", removed_text=text)
