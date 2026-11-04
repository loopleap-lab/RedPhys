import numpy as np
import cv2

from sklearn.metrics.cluster import mutual_info_score


def _mi(a, b):
    """mutual_info_score(a, b) from the same contingency table, skipping sklearn's per-call label checks."""
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    n = ib.max() + 1
    return mutual_info_score(None, None, contingency=np.bincount(ia * n + ib, minlength=(ia.max() + 1) * n).reshape(-1, n))


def get_heatmap_mi(img_gr, size_x, size_y, winsize, strd):
    heatmap_mi = np.zeros_like(img_gr, dtype=np.float32)
    stride_x = int(strd)
    stride_y = int(strd)
    win_size_x = int(winsize)
    win_size_y = int(winsize)
    img_gr = cv2.copyMakeBorder(img_gr, int(win_size_y/2), int(win_size_y/2), int(win_size_x/2), int(win_size_x/2), cv2.BORDER_REFLECT)
    for x in range(0, size_x - win_size_x, stride_x):
        for y in range(0, size_y - win_size_y, stride_y):
            window_cur = img_gr[x:x+win_size_x, y:y+win_size_y].flatten()
            mi_sum = 0
            neighbor_count = 0
            if (x-win_size_x) >= 0:
                window_cur_left = img_gr[x-win_size_x:x, y:y+win_size_y].flatten()
                mi_sum += _mi(window_cur, window_cur_left)
                neighbor_count += 1
            if (y-win_size_y) >= 0:
                window_cur_up = img_gr[x:x+win_size_x, y-win_size_y:y].flatten()
                mi_sum += _mi(window_cur, window_cur_up)
                neighbor_count += 1
            if (x + win_size_x*2) < size_x:
                window_cur_right = img_gr[x+win_size_x:x+win_size_x*2, y:y+win_size_y].flatten()
                mi_sum += _mi(window_cur, window_cur_right)
                neighbor_count += 1
            if (y + win_size_y*2) < size_y:
                window_cur_down = img_gr[x:x+win_size_x, y+win_size_y:y+win_size_y*2].flatten()
                mi_sum += _mi(window_cur, window_cur_down)
                neighbor_count += 1

            heatmap_mi[x:x+win_size_x, y:y+win_size_y] = mi_sum / neighbor_count

    x_exc = heatmap_mi.shape[0] - size_x
    y_exc = heatmap_mi.shape[1] - size_y
    heatmap_mi = heatmap_mi[round(x_exc / 2):size_x+round(x_exc / 2)-1, round(y_exc / 2):size_y+round(y_exc / 2)-1]  # -1 as the original

    return heatmap_mi

def img_heatmap_mi(bgr):
    """MI heatmap of a BGR uint8 image."""
    greyIm = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    size_x, size_y = greyIm.shape

    sx = np.ceil(size_x/100) + np.mod(np.ceil(size_x/100), 2)
    sy = np.ceil(size_y/100) + np.mod(np.ceil(size_y/100), 2)

    s1 = max(sx, sy)
    s2 = max(s1, 8)  # window; stride half of it
    return get_heatmap_mi(greyIm, size_x, size_y, s2, s2 / 2)
