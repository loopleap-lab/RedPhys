"""PA-Attack: prototype- and attention-guided attack on CLIP ViT."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..base_attack import AttackConfig, AttackResult, BaseAttack, pixel_linf, resize_center_crop, to_pil, to_tensor
from ..surrogates import UNIFIED_SURROGATES
from ...data import PROJECT_ROOT, Sample
from .core.loss import ComputeLossWrapper
from .core.pgd import pgd_veattack

_PROTO_DIR = Path(__file__).parent / "prototypes"
_COCO = PROJECT_ROOT / "dataset" / "coco" / "val2014"


@dataclass
class PAAttackConfig(AttackConfig):
    epsilon: float = 16.0
    max_iterations: int = 100
    alpha: float = 1.0  # step size, pixel units
    momentum: float = 0.9
    attn_layer: Optional[int] = None  # None: the middle block (12 for ViT-L-14)
    attn_temp: float = 20.0
    phase1_iters: int = 50
    prototype_path: Optional[str] = None
    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    input_res: int = 224


class PAAttack(BaseAttack):
    """Untargeted: CLIP ViT attention + OOD prototypes."""

    def __init__(self, config: PAAttackConfig):
        super().__init__(config)
        self._members = None  # [(ClipVisionModel, prototype tokens, attention layer)]

    def _initialize_models(self):
        if self._members is not None:
            return
        cfg = self.config
        if cfg.surrogates == "unified":
            specs = [(*m["open_clip"], None, None) for m in UNIFIED_SURROGATES]
        else:
            specs = [(cfg.clip_model_name, cfg.clip_pretrained, cfg.attn_layer, cfg.prototype_path)]
        members = []  # set after all load: no partial ensemble
        for arch, tag, layer, proto_path in specs:
            vis = self._load_vision(arch, tag)
            layer = len(vis.model.transformer.resblocks) // 2 if layer is None else layer
            if proto_path is None:
                proto_path = _PROTO_DIR / f"coco3000_pca1024_k20_{arch}_{tag}_v3.pt"  # v3: QuickGELU, pre-ln_post tokens
                if not proto_path.exists():
                    self._build_prototypes(vis, proto_path)
            protos = torch.load(proto_path, map_location=cfg.device, weights_only=True)
            members.append((vis, protos, layer))
        self._members = members

    def _load_vision(self, arch, tag):
        import open_clip
        from .core.clip_vision import ClipVisionModel

        model, _, preprocess = open_clip.create_model_and_transforms(  # OpenAI weights need QuickGELU
            arch, pretrained=tag, force_quick_gelu=open_clip.get_pretrained_cfg(arch, tag).get("quick_gelu", False))
        model.visual.final_ln_after_pool = True  # pre-ln_post tokens, as open_clip 2.19
        resize, normalize = preprocess.transforms[0], preprocess.transforms[-1]
        squash = transforms.Resize(model.visual.image_size, interpolation=resize.interpolation, antialias=True)
        return ClipVisionModel(model.visual, transforms.Compose([squash, normalize])).to(self.config.device).eval().requires_grad_(False)

    @torch.no_grad()
    def _build_prototypes(self, vis, path, n_images=3000, n_proto=20, pca_dim=1024):
        """As prepare_prototype.py: COCO tokens -> PCA -> KMeans."""
        import random
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        files = sorted(_COCO.glob("*.jpg"))
        if len(files) < n_images:
            raise FileNotFoundError(f"PA-Attack needs {n_images} COCO val2014 images in {_COCO}; run `bash install.sh weights paattack`")
        files = random.Random(self.config.seed).sample(files, n_images) if len(files) > n_images else files
        tokens = torch.cat([
            vis(vision=torch.cat([to_tensor(resize_center_crop(Image.open(f).convert("RGB"), self.config.input_res)).to(self.config.device) for f in files[i:i + 16]]),
                output_normalize=False, tokens=True)[1].float().cpu()
            for i in range(0, len(files), 16)
        ])
        n, t, d = tokens.shape
        x = PCA(min(pca_dim, n - 1), random_state=0, svd_solver="randomized").fit_transform(tokens.reshape(n, -1).numpy())
        ids = torch.from_numpy(KMeans(n_clusters=n_proto, random_state=0, n_init="auto").fit_predict(x))
        protos = torch.stack([F.normalize(tokens[ids == c], dim=-1).mean(0) if (ids == c).any() else torch.zeros(t, d)
                              for c in range(n_proto)])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(protos, path)

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()
        cfg = self.config

        img_tensor = to_tensor(sample.images[0]).to(cfg.device)
        eps = cfg.epsilon / 255.0

        def attention_pass(x):
            outs = []
            with torch.no_grad():
                for vis, _, layer in self._members:
                    emb, tok, attn = vis(vision=x, output_normalize=False, tokens=True, attention=True)
                    outs.append((emb, tok, F.softmax(cfg.attn_temp * attn[layer][:, :, 0, 1:].mean(dim=1), dim=-1)))
            return outs

        # target: the least similar prototype
        clean_pass = attention_pass(img_tensor)
        clean, min_sims, min_idxs = [], [], []
        for (_, protos, _), (emb_orig, tokens_orig, _) in zip(self._members, clean_pass):  # no-grad tensors
            token_sims = torch.einsum('bld,nld->bnl', F.normalize(tokens_orig, dim=-1), F.normalize(protos, dim=-1))
            min_sim, min_idx = torch.min(token_sims.mean(dim=-1), dim=1)
            clean.append((emb_orig, tokens_orig, protos[min_idx]))
            min_sims.append(min_sim.item())
            min_idxs.append(min_idx.item())

        def forward(x, output_normalize, tokens):
            return tuple(zip(*[vis(vision=x, output_normalize=output_normalize, tokens=tokens) for vis, _, _ in self._members]))

        def ensemble_loss(masks):
            fns = [ComputeLossWrapper(e, t, p, m, reduction="none") for (e, t, p), m in zip(clean, masks)]
            return lambda embs, toks: sum(f(e, t) for f, e, t in zip(fns, embs, toks))

        common = dict(forward=forward, data_clean=img_tensor, norm="linf", eps=eps, stepsize=cfg.alpha / 255.0,
                      output_normalize=False, mode='max', momentum=cfg.momentum)
        # Phase 1: PGD with clean attention masks
        phase1_out = pgd_veattack(loss_fn=ensemble_loss([m for _, _, m in clean_pass]), iterations=cfg.phase1_iters,
                                  perturbation=torch.zeros_like(img_tensor).uniform_(-eps, eps).requires_grad_(True), **common)
        # Phase 2: masks from the phase-1 output, same eps-ball
        phase2_out = pgd_veattack(loss_fn=ensemble_loss([m for _, _, m in attention_pass(phase1_out)]), iterations=cfg.max_iterations,
                                  perturbation=(phase1_out - img_tensor).detach().requires_grad_(True), **common)

        adv_pil = to_pil(phase2_out)
        return AttackResult(
            adversarial_sample=adv_pil,
            perturbation_norm=pixel_linf(adv_pil, sample.images[0]),
            metadata={
                "min_prototype_similarity": min_sims if len(min_sims) > 1 else min_sims[0],
                "prototype_index": min_idxs if len(min_idxs) > 1 else min_idxs[0],
            },
        )
