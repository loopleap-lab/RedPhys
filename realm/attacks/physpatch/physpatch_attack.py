"""PhysPatch: physically realizable patch attack."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..base_attack import BaseAttack, AttackConfig, AttackResult, LLMClientMixin, load_surrogates, open_by_stem, pixel_linf, to_pil, to_tensor
from ...data import Sample


@dataclass
class PhysPatchConfig(AttackConfig):
    epsilon: float = 16.0
    max_iterations: int = 300
    alpha: float = 1.0

    attack_method: str = "pgd"  # "pgd" or "mifgsm"
    decay: float = 1.0  # MI-FGSM momentum decay

    backbone: List[str] = field(default_factory=lambda: ["B16", "B32", "Laion"])
    K: int = 10  # SVD rank

    coords_file: Optional[str] = None
    llm_server_url: Optional[str] = None  # OpenAI-compatible server; None: OpenAI
    llm_api_key_env: str = "OPENAI_API_KEY"
    coords_model: str = "gpt-5-mini-2025-08-07"
    coords_prompt: str = "general"  # LLM region prompt: "general" or "driving" (upstream's)
    source_images_dir: Optional[str] = None
    source_images: Optional[List[str]] = None

    target_images_dir: Optional[str] = None

    crop_scale_min: float = 0.5
    crop_scale_max: float = 0.9
    crop_ratio: Tuple[float, float] = (0.5, 2.0)  # x image W/H
    tau0: float = 0.6  # mask threshold start
    tau_step: float = 0.02  # mask threshold growth per step
    sigma: float = 0.2  # initial potential field
    field_lr: float = 0.15  # potential field step
    patch_area: float = 0.01  # final patch area / image area
    full_weight: float = 0.1  # full-image loss weight

    def __post_init__(self):
        super().__post_init__()
        if self.attack_method not in ("pgd", "mifgsm"):
            raise ValueError(f"attack_method must be 'pgd' or 'mifgsm', got {self.attack_method!r}")
        if self.coords_prompt not in ("general", "driving"):
            raise ValueError(f"coords_prompt must be 'general' or 'driving', got {self.coords_prompt!r}")
        if self.coords_file is None and self.source_images_dir:  # <dataset>/coordinates/*.txt, else .../full.txt (generated)
            from .coordinate_generator import dataset_root
            coords = dataset_root(self.source_images_dir) / "coordinates"
            self.coords_file = str(next(iter(sorted(coords.glob("*.txt"))), coords / "full.txt"))


class PhysPatchAttack(LLMClientMixin, BaseAttack):
    config: PhysPatchConfig

    def __init__(self, config: PhysPatchConfig):
        super().__init__(config)
        self._ensemble_extractor = None
        self._ensemble_loss = None
        self._coords = None
        self._coords_error = None  # not retried per sample

    def _initialize_models(self):
        if self._ensemble_extractor is not None:
            return

        print("Loading CLIP ensemble models...")

        from .surrogates import (
            ClipB16FeatureExtractor,
            ClipB32FeatureExtractor,
            ClipLaionFeatureExtractor,
            EnsembleFeatureExtractor_ours,
            EnsembleFeatureLoss_ours_auto,
        )

        extractors = {"B16": ClipB16FeatureExtractor, "B32": ClipB32FeatureExtractor, "Laion": ClipLaionFeatureExtractor}
        models = load_surrogates(extractors, self.config.backbone, self.config.device)

        self._ensemble_extractor = EnsembleFeatureExtractor_ours(models, k=self.config.K)
        self._ensemble_loss = EnsembleFeatureLoss_ours_auto(models, k=self.config.K)

        print(f"✓ Ensemble ready with K={self.config.K} SVD components\n")

    def _load_coordinates(self) -> "dict | torch.Tensor":
        """Patch coordinates from file, generated if missing."""
        if self._coords is not None:
            return self._coords
        if self._coords_error is not None:
            raise self._coords_error

        if self.config.coords_file is None:  # set by __post_init__ when source_images_dir is
            raise ValueError("PhysPatch requires coords_file or source_images_dir")

        if self.config.source_images_dir:
            try:
                from .coordinate_generator import ensure_coordinates

                ensure_coordinates(
                    clean_images_dir=self.config.source_images_dir,
                    coords_file=self.config.coords_file,
                    llm=self.llm,  # called only if coordinates are missing
                    device=self.config.device,
                    model=self.config.coords_model,
                    prompt_style=self.config.coords_prompt,
                    names=set(self.config.source_images) if self.config.source_images else None,
                )
                print()

            except Exception as e:
                self._coords_error = RuntimeError(
                    f"Coordinate generation failed: {e}\n"
                    f"Please ensure:\n"
                    f"  1. {self.config.llm_api_key_env} is set\n"
                    f"  2. SoM dependencies are installed\n"
                    f"  3. SAM checkpoint exists at assets/checkpoints/\n"
                    f"Or provide a valid coords_file path."
                )
                raise self._coords_error from e

        # keyed "stem, x, y" or legacy positional "x, y" lines
        coords_dict = {}
        coords_list = []
        with open(self.config.coords_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 3:
                    stem = parts[0].strip()
                    x, y = float(parts[1]), float(parts[2])
                    coords_dict[stem] = [x, y]
                elif len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    coords_list.append([x, y])

        if coords_dict:
            self._coords = coords_dict
            print(f"Loaded {len(coords_dict)} keyed coordinates from {self.config.coords_file}")
        else:
            self._coords = torch.tensor(coords_list, device=self.config.device)
            print(f"Loaded {len(coords_list)} positional coordinates from {self.config.coords_file}")
        return self._coords

    def _prepare_image(self, image: Image.Image, target_size: tuple = None) -> torch.Tensor:
        """PIL -> [1, 3, H, W] in [0, 255], resized (bicubic) to target_size (H, W) if given."""
        if target_size is not None:
            image = image.resize(target_size[::-1], Image.BICUBIC)
        return to_tensor(image, 255).to(self.config.device)

    def _load_target_image(self, sample_id: str, size: tuple) -> torch.Tensor:
        """Target image by exact stem."""
        return self._prepare_image(open_by_stem(self.config.target_images_dir, sample_id), size)

    def _create_crops(self, coords: torch.Tensor, image_size: tuple):
        from .core.utils import RandomPointConstrainedCrop

        size = list(image_size)

        a = image_size[1] / image_size[0]  # W/H-scaled
        source_crop = RandomPointConstrainedCrop(
            size=size,
            scale=[self.config.crop_scale_min, self.config.crop_scale_max],
            ratio=(self.config.crop_ratio[0] * a, self.config.crop_ratio[1] * a),
            norm_coord=coords[0]
        )

        target_crop = transforms.RandomResizedCrop(
            size=size,
            scale=[self.config.crop_scale_min, self.config.crop_scale_max]
        )

        return source_crop, target_crop

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        coords = self._load_coordinates()
        if isinstance(coords, dict):
            sample_stem = str(sample.id)
            if sample_stem not in coords:
                raise KeyError(
                    f"No coordinate for sample '{sample_stem}'. "
                    f"Available: {list(coords.keys())[:5]}..."
                )
            xy = coords[sample_stem]
            coord = torch.tensor([xy], device=self.config.device)
        else:
            sample_idx = kwargs["sample_idx"]  # run.py: the position in the source dir
            if sample_idx < 0 or sample_idx >= len(coords):
                raise IndexError(
                    f"Coordinate index out of range: sample_idx={sample_idx}, coords_len={len(coords)}"
                )
            coord = coords[sample_idx:sample_idx+1]

        source_pil = sample.images[0]
        size = (source_pil.height, source_pil.width)
        clean_image = self._prepare_image(source_pil)
        target_tensor = self._load_target_image(sample.id, size)

        source_crop, target_crop = self._create_crops(coord, size)

        from .core.attacks import pgd, mifgsm

        attack_fn = pgd if self.config.attack_method == "pgd" else mifgsm

        self._ensemble_loss.previous_loss_list = []  # fresh dynamic weights per sample

        adv_image = attack_fn(
            image_tensor=clean_image,
            tgt_tensor=target_tensor,
            ensemble_extractor=self._ensemble_extractor,
            ensemble_loss=self._ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            center=coord,
            num_iters=self.config.max_iterations,
            epsilon=self.config.epsilon,  # pixel units, like image_tensor
            alpha=self.config.alpha,
            decay=self.config.decay,
            device=self.config.device,
            seed=torch.initial_seed() % 2**32,  # the per-sample seed of run.py (seed_all); numpy needs 32 bits
            **{k: getattr(self.config, k) for k in ("tau0", "tau_step", "sigma", "field_lr", "patch_area", "full_weight")},
        )

        adv_pil = to_pil(adv_image)

        return AttackResult(
            adversarial_sample=adv_pil,
            perturbation_norm=pixel_linf(adv_pil, source_pil),
            metadata={"coord": coord.cpu().tolist()},
        )
