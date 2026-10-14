"""PAD: SAM-based removal of regions flagged by fused MI/CD heatmaps."""

import warnings
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2
from PIL import Image

from ..base_defense import BaseDefense, DefenseConfig, DefenseResult


@dataclass
class PADDefenseConfig(DefenseConfig):
    precision: str = "fp32"  # SAM's mask generator converts tensors to numpy, which has no bf16
    iou_threshold: float = 0.5  # intersection / mask area (IoA)
    overlap_threshold: float = 0.1  # vs already-removed regions
    ratio_mi: float = 0.5
    kernel_param: int = 80
    thresh_param: int = 80
    sam_model_type: str = "vit_l"
    sam_checkpoint: Optional[str] = None
    device: str = "cuda"


class PADDefense(BaseDefense):
    def __init__(self, config: PADDefenseConfig):
        super().__init__(config)
        self.config: PADDefenseConfig = config
        self._sam_mask_generator = None
        self._pad_modules = None

    def _initialize_models(self):
        if self._sam_mask_generator is not None:
            return

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        from .core import fuse_filter as fuse_filter_module

        self._pad_modules = {
            'fuse_heatmap': fuse_filter_module.fuse_heatmap,
            'heatmap_filter': fuse_filter_module.heatmap_filter,
        }

        if self.config.sam_checkpoint is None:
            from .config import get_sam_checkpoint_path
            self.config.sam_checkpoint = get_sam_checkpoint_path(self.config.sam_model_type)

        print(f"Loading SAM model from {self.config.sam_checkpoint}")
        sam = sam_model_registry[self.config.sam_model_type](
            checkpoint=self.config.sam_checkpoint
        )
        sam.to(device=self.config.device)
        self._sam_mask_generator = SamAutomaticMaskGenerator(sam)
        print("SAM model loaded successfully")

    def clean(self, image_path: str, **_kwargs) -> DefenseResult:
        self._initialize_models()

        ori_img = Image.open(image_path).convert('RGB')
        ori_width, ori_height = ori_img.size

        fuse_heatmap = self._pad_modules['fuse_heatmap']
        with warnings.catch_warnings():  # ~65k sklearn warnings per image
            warnings.simplefilter("ignore", UserWarning)
            _mi_img, _cd_img, fuse_img = fuse_heatmap(image_path, ori_height, ori_width, self.config.ratio_mi)

        heatmap_filter = self._pad_modules['heatmap_filter']
        threshold = np.percentile(fuse_img, self.config.thresh_param)
        _h_t, _h_t_o, _h_t_o_c, h_t_o_c_o = heatmap_filter(
            fuse_img, threshold, ori_height, ori_width, self.config.kernel_param
        )

        gray = h_t_o_c_o > 0

        rgb_color = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # as PIL's size above
        image = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2RGB)
        masks = self._sam_mask_generator.generate(image.astype(np.uint8))

        print(f"Generated {len(masks)} SAM masks")

        result_mask = np.zeros(image.shape[:2], bool)
        regions_removed = 0

        for k in range(len(masks)):
            mask_k = masks[k].get('segmentation')
            mask_area = np.count_nonzero(mask_k)
            if mask_area == 0:
                continue
            iou = np.count_nonzero(mask_k & gray) / mask_area
            iou_prev = np.count_nonzero(mask_k & result_mask) / mask_area

            if iou > self.config.iou_threshold and iou_prev < self.config.overlap_threshold:
                rgb_color[mask_k] = 0
                result_mask |= mask_k
                regions_removed += 1

        cleaned_pil = Image.fromarray(cv2.cvtColor(rgb_color, cv2.COLOR_BGR2RGB))

        detection_confidence = float(np.count_nonzero(gray)) / (ori_height * ori_width)  # candidate area

        return DefenseResult(
            cleaned_sample=cleaned_pil,
            original_image_path=image_path,
            detection_confidence=detection_confidence,
            regions_removed=regions_removed,
            metadata={
                "removed_fraction": float(result_mask.mean()),
                "iou_threshold": self.config.iou_threshold,
                "ratio_mi": self.config.ratio_mi,
                "kernel_param": self.config.kernel_param,
                "thresh_param": self.config.thresh_param,
                "sam_model_type": self.config.sam_model_type,
                "total_sam_masks": len(masks),
                "image_size": f"{ori_width}x{ori_height}",
            }
        )
