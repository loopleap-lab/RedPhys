"""AdvDiffVLM: Latent Diffusion adversarial example attack with AEGE."""

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
from PIL import Image

from ..base_attack import BaseAttack, AttackConfig, AttackResult, pixel_linf, to_pil
from ..surrogates import UNIFIED_SURROGATES
from ...data import Sample


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

    gradcam_resolution: int = 64
    gradcam_layer: str = "layer4"
    auto_generate_masks: bool = True
    gradcam_masks_dir: Optional[str] = None

    target_images_dir: Optional[str] = None
    class_label: Optional[int] = None

    image_resolution: int = 256


class AdvDiffVLMAttack(BaseAttack):
    def __init__(self, config: AdvDiffVLMConfig):
        super().__init__(config)
        self.config: AdvDiffVLMConfig = config

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

        from .config import get_checkpoint_path, get_config_yaml_path, get_clip_cache_dir

        config_yaml = get_config_yaml_path()
        checkpoint = get_checkpoint_path()
        clip_cache = get_clip_cache_dir()

        print(f"Loading LDM checkpoint: {checkpoint}")
        config = OmegaConf.load(config_yaml)
        pl_sd = torch.load(checkpoint, map_location=self.config.device)
        sd = pl_sd["state_dict"]
        # Pad 1000 -> 1001 classes (null class for CFG)
        emb_key = "cond_stage_model.embedding.weight"
        if emb_key in sd and sd[emb_key].shape[0] == 1000:
            sd[emb_key] = torch.cat([sd[emb_key], torch.zeros(1, sd[emb_key].shape[1], device=sd[emb_key].device)], dim=0)
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
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
                clip_model, _ = clip.load(
                    clip_name,
                    device=self.config.device,
                    download_root=clip_cache
                )
            clip_model.eval()
            clip_model.requires_grad_(False)
            self._clip_models.append(clip_model)

        clip_input_res = self._clip_models[0].visual.input_resolution
        # original final_preprocess
        self._clip_resize = transforms.Compose([
            transforms.Resize(clip_input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(clip_input_res),
        ])
        self._clip_preprocess = transforms.Compose([
            self._clip_resize,
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
        self._gradcam_generator = GradCAMGenerator(
            layer_name=self.config.gradcam_layer,
            resolution=self.config.gradcam_resolution,
            device=self.config.device
        )

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        clean_image = self._prepare_image(sample.images[0])

        sample_idx = kwargs.get("sample_idx", None)
        target_image = self._load_target_image(sample, sample_idx)

        class_label = self.config.class_label
        if class_label is None:
            class_label = self._infer_class_label(sample.images[0])

        gradcam_mask = self._load_or_generate_gradcam_mask(sample, class_label)
        # Legacy DDIM sampler does .numpy() internally, so mask must be on CPU
        gradcam_mask = gradcam_mask.cpu()

        target_features = self._extract_clip_features(target_image)

        with torch.no_grad():
            with self._ldm_model.ema_scope():
                encoder_posterior = self._ldm_model.encode_first_stage(clean_image)
                z = self._ldm_model.get_first_stage_encoding(encoder_posterior).detach()

        xc = torch.tensor([class_label], device=self.config.device)

        with self._ldm_model.ema_scope():
            c = self._ldm_model.get_learned_conditioning({self._ldm_model.cond_stage_key: xc})
            uc = self._ldm_model.get_learned_conditioning(
                {self._ldm_model.cond_stage_key: torch.tensor([self._ldm_model.cond_stage_model.embedding.num_embeddings - 1]).to(self.config.device)}
            )

        print(f"Running AdvDiffVLM with {self.config.refinement_iterations} refinement iterations...")
        latent = z
        for refinement_iter in range(self.config.refinement_iterations):
            # Detach: no graph across refinement iterations
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
                label=xc,
                tgt_image_features_list=target_features,
                org_image_features_list=None,
                cam=gradcam_mask,
                K=1,
                seed=self.config.seed,
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
                "ddim_steps": self.config.ddim_steps,
                "refinement_iterations": self.config.refinement_iterations,
                "gradient_scale": self.config.gradient_scale,
                "clip_models": self._clip_names,
                "class_label": class_label,
            }
        )

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Squash to [1, 3, R, R] in [-1, 1]."""
        r = self.config.image_resolution
        transform = transforms.Compose([
            transforms.Resize((r, r), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: (img * 2.0 - 1.0)),
        ])
        tensor = transform(image).unsqueeze(0).to(self.config.device)
        return tensor

    def _load_target_image(self, sample: Sample, sample_idx: Optional[int]) -> Image.Image:
        """Load target image by ImageFolder order (legacy pairing)."""
        if self.config.target_images_dir is None:
            raise ValueError("target_images_dir must be specified")
        if sample_idx is None:
            raise ValueError("sample_idx is required to match legacy target pairing")

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
        return Image.open(target_path).convert("RGB")

    def _load_or_generate_gradcam_mask(self, sample: Sample, class_label: int) -> torch.Tensor:
        """GradCAM mask [64, 64], cached as PNG like the legacy pipeline."""
        if self.config.gradcam_masks_dir is None:
            raise ValueError("gradcam_masks_dir must be specified")

        masks_dir = Path(self.config.gradcam_masks_dir)
        masks_dir.mkdir(parents=True, exist_ok=True)
        mask_id = f"{sample.id}_c{class_label}"

        mask_file = masks_dir / f"{mask_id}.png"

        if mask_file.exists():
            import cv2
            cam = cv2.imread(str(mask_file), 0)
            if cam is None:
                raise FileNotFoundError(f"Failed to read mask image: {mask_file}")
            cam = cam / 255.0
            cam = cv2.resize(cam, (64, 64))
            return torch.tensor(cam, dtype=torch.float32, device=self.config.device)

        if self.config.auto_generate_masks:
            print(f"Auto-generating GradCAM mask for {mask_id}...")
            self._initialize_gradcam()

            mask = self._gradcam_generator.generate(sample.images[0], class_label)

            # PNG round trip, as legacy
            cam_np = (mask.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(cam_np).save(mask_file)
            print(f"  Cached mask to {mask_file}")

            import cv2
            cam = cv2.imread(str(mask_file), 0)
            cam = cam / 255.0
            cam = cv2.resize(cam, (64, 64))
            return torch.tensor(cam, dtype=torch.float32, device=self.config.device)

        raise FileNotFoundError(f"GradCAM mask not found and auto_generate_masks=False: {mask_file}")

    def _infer_class_label(self, source_image: Image.Image) -> int:
        self._initialize_gradcam()
        with torch.no_grad():
            img_tensor = self._gradcam_generator.preprocess(source_image).unsqueeze(0).to(self.config.device)
            logits = self._gradcam_generator.model(img_tensor)
            label = int(logits.argmax(dim=1).item())
        print(f"Auto-detected class_label={label} from source image")
        return label

    def _extract_clip_features(self, image: Image.Image) -> List[torch.Tensor]:
        input_res = self._clip_models[0].visual.input_resolution

        transform = transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 1.0)),
        ])
        image_tgt = transform(image).unsqueeze(0).to(self.config.device)

        with torch.no_grad():
            tgt_image_features_list = []
            image_tgt = self._clip_preprocess(image_tgt)
            for clip_model in self._clip_models:
                tgt_image_features = clip_model.encode_image(image_tgt)
                tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)
                tgt_image_features_list.append(tgt_image_features)

        return tgt_image_features_list
