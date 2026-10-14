from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

def recompress_diff(imorig, checkDisplacements):
    minQ = 51
    maxQ = 100
    stepQ = 1

    if checkDisplacements == 1:
        maxDisp = 7
    else:
        maxDisp = 0

    smoothing_b = 17
    Offset = (smoothing_b - 1) // 2

    height, width, _ = imorig.shape

    imorig_f = imorig.astype(float)
    h = np.ones((smoothing_b, smoothing_b)) / smoothing_b**2

    def quality(ii):
        jpeg = cv2.imencode(".jpg", imorig, [int(cv2.IMWRITE_JPEG_QUALITY), ii])[1]  # in memory: same codec as imwrite/imread
        tmpResave = cv2.imdecode(jpeg, cv2.IMREAD_COLOR).astype(float)
        Deltas = []
        overallDelta = []

        for dispx in range(maxDisp + 1):
            for dispy in range(maxDisp + 1):
                DisplacementIndex = dispx * 8 + dispy + 1
                tmpResave_disp = tmpResave[dispx:, dispy:, :]
                imorig_disp = imorig_f[:height-dispx, :width-dispy, :]
                Comparison = np.square(imorig_disp - tmpResave_disp)
                Comparison = cv2.filter2D(Comparison, -1, h)

                Comparison = Comparison[Offset:-Offset, Offset:-Offset, :]
                Deltas.append(np.mean(Comparison, axis=2))
                overallDelta.append(np.mean(Deltas[DisplacementIndex - 1]))

        minOverallDelta, minInd = min(overallDelta), np.argmin(overallDelta)
        delta = Deltas[minInd]
        delta_range = np.max(delta) - np.min(delta)
        delta = (delta - np.min(delta)) / delta_range if delta_range > 0 else delta * 0
        return minOverallDelta, minInd, cv2.resize(delta.astype(np.float32), (delta.shape[1] // 4, delta.shape[0] // 4), interpolation=cv2.INTER_LINEAR)

    with ThreadPoolExecutor(max_workers=4) as pool:  # independent qualities; cv2/numpy release the GIL; map keeps order
        Output, mins, dispImages = map(list, zip(*pool.map(quality, range(minQ, maxQ + 1, stepQ))))

    OutputY = Output
    OutputX = list(range(minQ, maxQ + 1, stepQ))
    _xmax, _imax, _xmin, imin = cv2.minMaxLoc(np.array(OutputY))
    imin = sorted(imin)
    Qualities = [i * stepQ + minQ - 1 for i in imin]

    return OutputX, OutputY, dispImages, imin, Qualities, mins

def clean_up_image(filename):
    im = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    dots = filename.rfind('.')
    extension = filename[dots:]
    
    if extension.lower() == '.gif' and im.shape[2] < 3:
        im_gif, gif_map = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        im_gif = im_gif[:, :, 0]
        im = np.uint8(cv2.cvtColor(im_gif, cv2.COLOR_GRAY2RGB) * 255)

    if im.shape[2] < 3:
        im[:, :, 1] = im[:, :, 0]
        im[:, :, 2] = im[:, :, 0]

    if im.shape[2] > 3:
        im = im[:, :, 0:3]

    if im.dtype == np.uint16:
        im = np.uint8(np.floor(im / 256))

    return im

def img_heatmap_cd(impath):
    im = clean_up_image(impath)
    checkDisplacements = 0
    OutputX, _OutputY, dispImages, _imin, _Qualities, _mins = recompress_diff(im, checkDisplacements)
    return dispImages, OutputX
