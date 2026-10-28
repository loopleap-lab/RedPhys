from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


def recompress_diff(imorig):
    """Per JPEG quality 51..100: the smoothed resave error, min-max scaled, at 1/4 size (no displacement search)."""
    smoothing_b = 17
    offset = (smoothing_b - 1) // 2
    imorig_f = imorig.astype(float)
    h = np.ones((smoothing_b, smoothing_b)) / smoothing_b**2

    def quality(q):
        jpeg = cv2.imencode(".jpg", imorig, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]  # in memory: same codec as imwrite/imread
        resave = cv2.imdecode(jpeg, cv2.IMREAD_COLOR).astype(float)
        comparison = cv2.filter2D(np.square(imorig_f - resave), -1, h)[offset:-offset, offset:-offset, :]
        delta = np.mean(comparison, axis=2)
        delta_range = np.max(delta) - np.min(delta)
        delta = (delta - np.min(delta)) / delta_range if delta_range > 0 else delta * 0
        return cv2.resize(delta.astype(np.float32), (delta.shape[1] // 4, delta.shape[0] // 4), interpolation=cv2.INTER_LINEAR)

    with ThreadPoolExecutor(max_workers=4) as pool:  # independent qualities; cv2/numpy release the GIL; map keeps order
        return list(pool.map(quality, range(51, 101)))


def img_heatmap_cd(bgr):
    """CD heatmaps of a BGR uint8 image, one per JPEG quality."""
    return recompress_diff(bgr)
