# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms, box_area

from typing import Any, Dict, List, Optional, Tuple

from segment_anything.modeling import Sam
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SeemAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.9,
        stability_score_thresh: float = 0.5,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """Using a SAM model, generates masks for the entire image."""

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils

        if min_mask_region_area > 0:
            import cv2

        self.predictor = model
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

        self.dilation = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.dilation.weight.data.fill_(1.0)
        self.dilation.cuda()

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generates masks for the given image."""

        mask_data = self._generate_masks(image)

        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[-2:]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        if len(crop_boxes) > 1:
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        x0, y0, x1, y1 = crop_box
        cropped_im = image
        cropped_im_size = cropped_im.shape[-2:]

        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx]

        data = MaskData()
        self.enc_features=None

        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(cropped_im, points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data

        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros(len(data["boxes"])),
            iou_threshold=self.box_nms_thresh,
        )

        data.filter(keep_by_nms)

        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        images,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        data = {"image": images, "height": orig_h, "width": orig_w}
        points = torch.tensor(points,dtype=torch.float).to(images.device)
        
        abs_points = (points * torch.tensor(orig_size)[None,:].to(points.device)).long()
        abs_masks = torch.zeros((len(points), orig_h, orig_w), dtype=torch.bool).to(device=points.device)
        abs_masks[torch.arange(0, abs_points.size(0))[:,None], abs_points[:,0:1], abs_points[:,1:2]] = True
        abs_masks = self.dilation(abs_masks[:,None].float())[:,0] > 0
        data['spatial_query'] = {'rand_shape': abs_masks[:,None]}

        batch_inputs = [data]
        if self.enc_features is None:
            masks, iou_preds, mask_features, transformer_encoder_features, multi_scale_features = self.predictor.model.evaluate_demo(batch_inputs, None, None, return_features=True)
            self.enc_features = (mask_features, transformer_encoder_features, multi_scale_features)
        else:
            masks, iou_preds = self.predictor.model.evaluate_demo(batch_inputs, self.enc_features[0], self.enc_features[1], self.enc_features[2])

        data = MaskData(
            masks=masks,
            iou_preds=iou_preds,
            points=points,
        )
        del masks
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        data["stability_score"] = calculate_stability_score(
            data["masks"], 0.0, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        data["masks"] = data["masks"] > 0.0
        data["boxes"] = batched_mask_to_box(data["masks"])

        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """Remove small disconnected regions and holes in masks."""
        if len(mask_data["rles"]) == 0:
            return mask_data

        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            scores.append(float(unchanged))

        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),
            iou_threshold=nms_thresh,
        )

        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]
        mask_data.filter(keep_by_nms)

        return mask_data