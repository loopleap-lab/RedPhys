# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from task_adapter.utils.visualizer import Visualizer
from typing import Tuple
from PIL import Image
from detectron2.data import MetadataCatalog
from kornia.contrib import distance_transform
import matplotlib.pyplot as plt
import cv2
import io
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

from segment_anything import SamAutomaticMaskGenerator
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

def sam_interactive_mask(mask_generator, points, in_points, in_labels, mask_input):
    masks, iou_preds, _ = mask_generator.predictor.predict_torch(
            in_points,
            in_labels,
            mask_input=mask_input,
            multimask_output=True,
            return_logits=True,
    )
    nm,_,h,w = masks.shape

    data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
    )
    del masks

    data["stability_score"] = calculate_stability_score(
            data["masks"], mask_generator.predictor.model.mask_threshold, mask_generator.stability_score_offset
    )

    masks = data["masks"].reshape(nm, -1, h, w)
    scores = (data['iou_preds'] + data['stability_score']).reshape(nm, -1)

    index = torch.stack([torch.arange(nm).cuda(), scores.argmax(dim=1)]).tolist()
    return masks[index]

def inference_sam_m2m_interactive(model, image, spatial_masks, text_size, label_mode='1', alpha=0.1, anno_mode=['Mask']):
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)

    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    orig_size = images.shape[-2:]
    orig_h, orig_w = orig_size
    crop_box = [0,0,orig_w,orig_h]

    spatial_masks = spatial_masks[:, None].float().cuda()
    spatial_masks = F.interpolate(spatial_masks, size=(orig_h, orig_w), mode='bicubic', align_corners=False) > 0


    acc_points = []
    for i in range(len(spatial_masks)):
        points = spatial_masks[i:i+1].nonzero()[:,2:].flip(dims=[1]).cpu().numpy()
        rand_ids = np.random.choice(points.shape[0], size=40, replace=True)
        points = points[rand_ids]
        acc_points.append(points)
    _np = len(acc_points)
    points = np.concatenate(acc_points)

    mask_generator = SamAutomaticMaskGenerator(model)
    mask_generator.predictor.set_image(image_ori)
    im_size = image_ori.shape[:-1]

    transformed_points = mask_generator.predictor.transform.apply_coords(points, im_size)
    in_points = torch.as_tensor(transformed_points, device=mask_generator.predictor.device).reshape(_np,-1,2).transpose(0,1)
    in_labels = torch.ones((in_points.shape[0], _np), dtype=torch.int, device=mask_generator.predictor.device)

    masks = sam_interactive_mask(mask_generator, points, in_points.transpose(0,1), in_labels.transpose(0,1), None)

    masks = masks > 0.0
    iou_preds = torch.ones(masks.shape[0], dtype=torch.float32)
    points = torch.zeros((masks.shape[0], 2), dtype=torch.float32)

    mask_data = MaskData(
        masks=masks,
        iou_preds=iou_preds,
        points=points,
    )

    mask_data["stability_score"] = torch.ones(masks.shape[0], dtype=torch.float32)
    del masks

    mask_data["boxes"] = batched_mask_to_box(mask_data["masks"])
    mask_data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(mask_data["boxes"]))])

    mask_data["masks"] = uncrop_masks(mask_data["masks"], crop_box, orig_h, orig_w)
    mask_data["rles"] = mask_to_rle_pytorch(mask_data["masks"])
    del mask_data["masks"]
    mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

    outputs = []
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
        outputs.append(ann)

    from task_adapter.utils.visualizer import Visualizer
    visual = Visualizer(image_ori, metadata=metadata)
    sorted_anns = sorted(outputs, key=(lambda x: x['area']), reverse=True)
    label = 1

    mask_map = np.zeros(image_ori.shape, dtype=np.uint8)    
    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        mask_map[mask == 1] = label
        label += 1
    im = demo.get_image()    
    return im, sorted_anns


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """Remove small disconnected regions and holes in a mask."""
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # row 0 is the background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # keep largest region if all are below threshold
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))