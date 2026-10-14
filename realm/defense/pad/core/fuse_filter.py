import numpy as np
import cv2

from .heatmap_MI import img_heatmap_mi
from .heatmap_CD import img_heatmap_cd

def fuse_heatmap(impath, ori_height, ori_width, ratio_mi=0.5):  # ratio_cd = 1-ratio_mi
    h_mi = img_heatmap_mi(impath)
    h_cd, _qt = img_heatmap_cd(impath)
    h_cd = np.mean(h_cd, axis=0)

    h_mi = cv2.resize(h_mi, (ori_width, ori_height))
    h_cd = cv2.resize(h_cd, (ori_width, ori_height))

    h_mi, h_cd = _scale_255(h_mi), _scale_255(h_cd)
    h_fuse = (h_mi * ratio_mi + h_cd * (1 - ratio_mi)).astype(np.int64)  # float64, truncated like int()
    return h_mi.astype(np.uint8), h_cd.astype(np.uint8), h_fuse.astype(np.uint8)


def _scale_255(h):
    """Min-max scale to 0..255, truncated as in the original."""
    h_range = np.max(h) - np.min(h)
    return ((h - np.min(h)) * 255 / h_range).astype(np.int64) if h_range > 0 else np.zeros(h.shape, np.int64)

def heatmap_filter(heatmap, threshold, height, width, kernel_param=80):
    _thresh, h_t = cv2.threshold(heatmap, threshold, maxval=255, type=cv2.THRESH_TOZERO)

    base_kernel_size = int(min(height, width)/kernel_param)

    kernel = np.ones((base_kernel_size*2, base_kernel_size*2), np.uint8)
    h_t_o = cv2.morphologyEx(h_t, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel = np.ones((base_kernel_size, base_kernel_size), np.uint8)
    h_t_o_c = cv2.morphologyEx(h_t_o, cv2.MORPH_CLOSE, kernel, iterations=2)

    kernel = np.ones((base_kernel_size*3, base_kernel_size*3), np.uint8)
    h_t_o_c_o = cv2.morphologyEx(h_t_o_c, cv2.MORPH_OPEN, kernel, iterations=2)

    return h_t, h_t_o, h_t_o_c, h_t_o_c_o
