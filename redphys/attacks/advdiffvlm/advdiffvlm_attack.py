"""AdvDiffVLM: latent diffusion with AEGE."""

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
from PIL import Image

from ..base_attack import BaseAttack, AttackConfig, AttackResult, pixel_linf, resize_center_crop, to_pil, to_tensor
from ..surrogates import UNIFIED_SURROGATES
from ...data import Sample, load_rgb, weight

_RES = 256  # the cin256 LDM's image size
_ASSETS = Path(__file__).parent / "assets"


def _asset(rel: str) -> str:
    return str(weight(_ASSETS / rel, "advdiffvlm"))


@dataclass
class AdvDiffVLMConfig(AttackConfig):
    ddim_steps: int = 200
    ddim_eta: float = 0.0
    guidance_scale: float = 5.0

    gradient_scale: float = 35
    refinement_iterations: int = 10
    t_star: float = 0.2  # guided fraction of the reverse process
    grad_clip: float = 0.0025  # delta
    aege_temp: float = 2.0  # tau
    hole_size: int = 8  # k, latent px
    mask_floor: float = 0.045
    cam_clip: Tuple[float, float] = (0.3, 0.7)  # hole sampling range

    clip_models: List[str] = field(default_factory=lambda: ["RN50", "RN101", "ViT-B/16", "ViT-B/32"])

    gradcam_layer: str = "layer4"
    auto_generate_masks: bool = True
    gradcam_masks_dir: Optional[str] = None

    target_images_dir: Optional[str] = None
    class_label: Optional[int] = None


class AdvDiffVLMAttack(BaseAttack):
    config: AdvDiffVLMConfig

    def __init__(self, config: AdvDiffVLMConfig):
        super().__init__(config)
        self._ldm_model = None
        self._ddim_sampler = None
        self._clip_models = []
        self._clip_preprocess = None
        self._gradcam_generator = None
        self._target_paths_cache = None

    def _initialize_models(self):
        if self._ldm_model is not None:
            return

        print("Loading AdvDiffVLM models (LDM + CLIP ensemble)...")

        _attack_dir = str(Path(__file__).parent)
        if _attack_dir not in sys.path:
            sys.path.insert(0, _attack_dir)

        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim_main import DDIMSampler
        import clip

        config_yaml = _asset("configs/latent-diffusion_cin256-v2.yaml")
        checkpoint = _asset("checkpoints/ldm_cin256-v2.ckpt")

        print(f"Loading LDM checkpoint: {checkpoint}")
        config = OmegaConf.load(config_yaml)
        pl_sd = torch.load(checkpoint, map_location="cpu")  # one GPU copy, after .to()
        model = instantiate_from_config(config.model)
        model.load_state_dict(pl_sd["state_dict"], strict=False)  # 1001 classes: the last is the null class for CFG
        model.to(self.config.device)
        model.eval()
        model.requires_grad_(False)
        self._ldm_model = model

        clip_names = self.config.clip_models
        if self.config.surrogates == "unified":
            clip_names = [m["clip"] or "open_clip:" + ":".join(m["open_clip"]) for m in UNIFIED_SURROGATES]
        print(f"Loading CLIP ensemble: {clip_names}")
        self._clip_names = clip_names
        for clip_name in clip_names:
            if clip_name.startswith("open_clip:"):
                import open_clip
                _, arch, tag = clip_name.split(":")
                clip_model = open_clip.create_model(arch, pretrained=tag, device=self.config.device)
            else:
                clip_model, _ = clip.load(clip_name, device=self.config.device)  # ~/.cache/clip
            clip_model.eval()
            clip_model.requires_grad_(False)
            self._clip_models.append(clip_model)

        v = self._clip_models[0].visual
        self._clip_res = res = v.input_resolution if hasattr(v, "input_resolution") else v.image_size[0]  # clip, open_clip
        # original final_preprocess
        resize = transforms.Compose([
            transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(res),
        ])
        self._clip_preprocess = transforms.Compose([
            resize,
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        self._ddim_sampler = DDIMSampler(model, models=self._clip_models, preprocess=self._clip_preprocess)

        print("AdvDiffVLM models loaded successfully")

    def _initialize_gradcam(self):
        if self._gradcam_generator is not None:
            return

        from .gradcam import GradCAMGenerator
        self._gradcam_generator = GradCAMGenerator(layer_name=self.config.gradcam_layer, device=self.config.device)

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        clean_image = self._prepare_image(sample.images[0])

        target_image = self._load_target_image(sample, kwargs.get("sample_idx"))

        class_label = self.config.class_label
        if class_label is None:
            class_label = self._infer_class_label(sample.images[0])

        gradcam_mask = self._load_or_generate_gradcam_mask(sample, class_label)

        target_features = self._extract_clip_features(target_image)

        with torch.no_grad():
            with self._ldm_model.ema_scope():
                encoder_posterior = self._ldm_model.encode_first_stage(clean_image)
                z = self._ldm_model.get_first_stage_encoding(encoder_posterior)

        xc = torch.tensor([class_label], device=self.config.device)

        with self._ldm_model.ema_scope():
            c = self._ldm_model.get_learned_conditioning({self._ldm_model.cond_stage_key: xc})
            null = torch.tensor([self._ldm_model.cond_stage_model.embedding.num_embeddings - 1], device=self.config.device)
            uc = self._ldm_model.get_learned_conditioning({self._ldm_model.cond_stage_key: null})

        latent = z
        for refinement_iter in range(self.config.refinement_iterations):
            # no graph across refinements
            latent = latent.detach()

            latent, _ = self._ddim_sampler.sample(
                S=self.config.ddim_steps,
                conditioning=c,
                x_T=latent,
                batch_size=1,
                shape=list(latent.shape[1:]),
                verbose=False,
                unconditional_guidance_scale=self.config.guidance_scale,
                unconditional_conditioning=uc,
                eta=self.config.ddim_eta,
                tgt_image_features_list=target_features,
                cam=gradcam_mask,
                K=1,
                seed=torch.initial_seed() % 2**32,  # the per-sample seed of run.py (seed_all); numpy needs 32 bits
                s=self.config.gradient_scale,
                guide={k: getattr(self.config, k) for k in ("t_star", "grad_clip", "aege_temp", "hole_size", "mask_floor", "cam_clip")},
            )
            if (refinement_iter + 1) % 5 == 0:
                print(f"  Refinement {refinement_iter + 1}/{self.config.refinement_iterations}")

        with torch.no_grad():
            adv_image = self._ldm_model.decode_first_stage(latent.detach())
            w, h = sample.images[0].size  # to the source size
            adv_image = transforms.functional.resize(torch.clamp((adv_image + 1.0) / 2.0, 0, 1), [h, w],
                                                     interpolation=transforms.InterpolationMode.BICUBIC, antialias=True).cpu()
        del latent
        torch.cuda.empty_cache()

        adv_pil = to_pil(adv_image)

        return AttackResult(
            adversarial_sample=adv_pil,
            perturbation_norm=pixel_linf(adv_pil, sample.images[0]),
            metadata={
                "clip_models": self._clip_names,
                "class_label": class_label,
            }
        )

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Squash to [1, 3, 256, 256] in [-1, 1]."""
        return (to_tensor(image.resize((_RES, _RES), Image.BICUBIC)) * 2 - 1).to(self.config.device)

    def _load_target_image(self, sample: Sample, sample_idx: Optional[int]) -> Image.Image:
        """Target image by ImageFolder order, as the official pairing."""
        if self.config.target_images_dir is None:
            raise ValueError("target_images_dir must be specified")
        if sample_idx is None:
            raise ValueError("sample_idx is required to match the ImageFolder target pairing")

        if self._target_paths_cache is None:
            target_dir = Path(self.config.target_images_dir)
            if not target_dir.exists():
                raise FileNotFoundError(f"Target images dir not found: {target_dir}")
            dataset = tv_datasets.ImageFolder(str(target_dir))
            self._target_paths_cache = [p for (p, _) in dataset.samples]

        if sample_idx < 0 or sample_idx >= len(self._target_paths_cache):
            raise IndexError(
                f"sample_idx {sample_idx} out of range for target dataset "
                f"(size={len(self._target_paths_cache)})"
            )

        target_path = self._target_paths_cache[sample_idx]
        return load_rgb(target_path)

    def _load_or_generate_gradcam_mask(self, sample: Sample, class_label: int) -> torch.Tensor:
        """GradCAM mask [64, 64], PNG-cached as upstream."""
        if self.config.gradcam_masks_dir is None:
            raise ValueError("gradcam_masks_dir must be specified")

        masks_dir = Path(self.config.gradcam_masks_dir)
        masks_dir.mkdir(parents=True, exist_ok=True)
        layer = "" if self.config.gradcam_layer == "layer4" else f"_{self.config.gradcam_layer}"  # layer4: the upstream name
        mask_file = masks_dir / f"{sample.id}_c{class_label}{layer}.png"
        if mask_file.exists():
            return self._read_mask(mask_file)

        if self.config.auto_generate_masks:
            print(f"Auto-generating GradCAM mask {mask_file.name}...")
            self._initialize_gradcam()

            mask = self._gradcam_generator.generate(sample.images[0], class_label)

            Image.fromarray((mask.clamp(0, 1).cpu().numpy() * 255).astype("uint8")).save(mask_file)  # PNG round trip, as upstream
            print(f"  Cached mask to {mask_file}")
            return self._read_mask(mask_file)

        raise FileNotFoundError(f"GradCAM mask not found and auto_generate_masks=False: {mask_file}")

    def _read_mask(self, mask_file: Path) -> torch.Tensor:
        import cv2
        cam = cv2.imread(str(mask_file), 0)
        if cam is None:
            raise FileNotFoundError(f"Failed to read mask image: {mask_file}")
        return torch.tensor(cv2.resize(cam / 255.0, (64, 64)), dtype=torch.float32)  # CPU: the DDIM sampler calls .numpy()

    def _infer_class_label(self, source_image: Image.Image) -> int:
        self._initialize_gradcam()
        with torch.no_grad():
            img_tensor = self._gradcam_generator.preprocess(source_image).unsqueeze(0).to(self.config.device)
            logits = self._gradcam_generator.model(img_tensor)
            label = int(logits.argmax(dim=1).item())
        return label

    def _extract_clip_features(self, image: Image.Image) -> List[torch.Tensor]:
        x = self._clip_preprocess(to_tensor(resize_center_crop(image, self._clip_res)).to(self.config.device))
        with torch.no_grad():
            return [(f := m.encode_image(x)) / f.norm(dim=1, keepdim=True) for m in self._clip_models]
