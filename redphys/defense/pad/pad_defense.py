"""PAD: SAM-based removal of regions flagged by fused MI/CD heatmaps."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ...data import weight
from ..base_defense import BaseDefense, DefenseConfig, DefenseResult

ASSETS = Path(__file__).parent / "assets"
SAM_CHECKPOINTS = {"vit_b": "sam_vit_b_01ec64.pth", "vit_l": "sam_vit_l_0b3195.pth", "vit_h": "sam_vit_h_4b8939.pth"}


@dataclass
class PADDefenseConfig(DefenseConfig):
    precision: str = "fp32"  # SAM masks go through numpy: no bf16
    iou_threshold: float = 0.5  # intersection / mask area (IoA)
    overlap_threshold: float = 0.1  # vs already-removed regions
    ratio_mi: float = 0.5
    kernel_param: int = 80
    thresh_param: int = 80
    sam_model_type: str = "vit_l"
    sam_checkpoint: Optional[str] = None  # default: assets/models/<official checkpoint>


class PADDefense(BaseDefense):
    def __init__(self, config: PADDefenseConfig):
        super().__init__(config)
        self._sam_mask_generator = None

    def _initialize_models(self):
        if self._sam_mask_generator is not None:
            return

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        ckpt = self.config.sam_checkpoint or ASSETS / "models" / SAM_CHECKPOINTS[self.config.sam_model_type]
        weight(ckpt, "pad")
        print(f"Loading SAM model from {ckpt}")
        sam = sam_model_registry[self.config.sam_model_type](checkpoint=str(ckpt))
        sam.to(device=self.config.device)
        self._sam_mask_generator = SamAutomaticMaskGenerator(sam)
        print("SAM model loaded successfully")

    def clean(self, image_path: str, **_kwargs) -> DefenseResult:
        from .core.fuse_filter import fuse_heatmap, heatmap_filter
        self._initialize_models()

        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # raw orientation, as the heatmaps
        ori_height, ori_width = bgr.shape[:2]

        with warnings.catch_warnings():  # ~65k sklearn warnings per image
            warnings.simplefilter("ignore", UserWarning)
            fuse_img = fuse_heatmap(bgr, ori_height, ori_width, self.config.ratio_mi)

        threshold = np.percentile(fuse_img, self.config.thresh_param)
        gray = heatmap_filter(fuse_img, threshold, ori_height, ori_width, self.config.kernel_param) > 0

        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks = self._sam_mask_generator.generate(image)

        result_mask = np.zeros(image.shape[:2], bool)
        regions_removed = 0

        for m in masks:
            mask_k = m["segmentation"]
            mask_area = np.count_nonzero(mask_k)
            if mask_area == 0:
                continue
            ioa = np.count_nonzero(mask_k & gray) / mask_area
            ioa_prev = np.count_nonzero(mask_k & result_mask) / mask_area
            if ioa > self.config.iou_threshold and ioa_prev < self.config.overlap_threshold:
                image[mask_k] = 0
                result_mask |= mask_k
                regions_removed += 1

        return DefenseResult(cleaned_sample=Image.fromarray(image), metadata={
            "detection_confidence": float(gray.mean()),  # candidate area
            "regions_removed": regions_removed,
            "removed_fraction": float(result_mask.mean()),
            "total_sam_masks": len(masks),
        })
