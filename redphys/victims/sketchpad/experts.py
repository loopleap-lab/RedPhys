"""Visual Sketchpad's vision experts behind a unix socket: GroundingDINO, Depth-Anything, SoM with SAM ViT-H (upstream Semantic-SAM)."""

import base64
import io
import json
import math
import socketserver
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...data import weight

SOM = Path(__file__).parents[2] / "attacks" / "physpatch" / "assets" / "som"
SAM = SOM.parent / "checkpoints" / "sam_vit_h_4b8939.pth"
GDINO, DEPTH = "IDEA-Research/grounding-dino-tiny", "LiheYoung/depth-anything-large-hf"  # the upstream checkpoints, converted
PALETTE = ["#A351FB", "#FF4040", "#FFA1A0", "#FF7633", "#FFB633", "#D1D435", "#4CFB12", "#94CF1A", "#40DE8A", "#1B9640", "#00D6C1",
           "#2E9CAA", "#00C4FF", "#364797", "#6675FF", "#0019EF", "#863AFF", "#530087", "#CD3AFF", "#FF97FF", "#FF39C7"]  # supervision's


def png(arr_or_img) -> str:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def annotate(rgb: np.ndarray, boxes: np.ndarray, labels: list[str]) -> np.ndarray:
    """Boxes and top-left labels colored by index, as supervision's BoxAnnotator + LabelAnnotator (thickness 2, scale 0.5, padding 10)."""
    out = rgb.copy()
    for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes.astype(int), labels)):
        c = tuple(int(PALETTE[i % len(PALETTE)][k:k + 2], 16) for k in (1, 3, 5))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 20), (x1 + tw + 20, y1), c, -1)
        cv2.putText(out, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class Experts:

    def __init__(self, device: str = "cuda"):
        self.device, self.lock, self.models = device, threading.Lock(), {}

    def _model(self, name):
        if name not in self.models:
            if name == "detection":
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
                self.models[name] = AutoProcessor.from_pretrained(GDINO), AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to(self.device).eval()
            elif name == "depth":
                from transformers import AutoModelForDepthEstimation
                self.models[name] = AutoModelForDepthEstimation.from_pretrained(DEPTH).to(self.device).eval()
            else:
                from segment_anything import sam_model_registry
                sys.path.insert(0, str(SOM))
                from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
                self.models[name] = sam_model_registry["vit_h"](checkpoint=str(weight(SAM, "sketchpad"))).to(self.device).eval(), inference_sam_m2m_auto
        return self.models[name]

    @torch.no_grad()
    def detection(self, image, text, box_threshold=0.35, text_threshold=0.25):
        processor, model = self._model("detection")
        caption = text.lower().strip()
        caption = caption if caption.endswith(".") else caption + "."  # as groundingdino's predict
        inputs = processor(images=image, text=caption, return_tensors="pt").to(self.device)
        out = processor.post_process_grounded_object_detection(model(**inputs), inputs.input_ids, threshold=box_threshold,
                                                               text_threshold=text_threshold, target_sizes=[image.size[::-1]])[0]
        xyxy, (w, h) = out["boxes"].cpu().numpy(), image.size
        boxes = [[(a + c) / 2 / w, (b + d) / 2 / h, (c - a) / w, (d - b) / h] for a, b, c, d in xyxy.tolist()]  # normalized cxcywh
        return {"image": png(annotate(np.asarray(image), xyxy, [f"{p} {i}" for i, p in enumerate(out["text_labels"], 1)])), "boxes": boxes}

    @torch.no_grad()
    def depth(self, image):
        rgb = np.asarray(image)
        h, w = rgb.shape[:2]
        x = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) / 255.0  # as the upstream server (its input is RGB, so this swaps channels)
        s = max(518 / h, 518 / w)  # Resize(518, keep_aspect_ratio, lower_bound, ensure_multiple_of=14, cubic)
        m14 = lambda v: (round(v / 14) * 14) if round(v / 14) * 14 >= 518 else math.ceil(v / 14) * 14
        x = cv2.resize(x, (m14(s * w), m14(s * h)), interpolation=cv2.INTER_CUBIC)
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(self.device)
        d = F.interpolate(self._model("depth")(pixel_values=x).predicted_depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
        d = ((d - d.min()) / (d.max() - d.min()) * 255.0).cpu().numpy().astype(np.uint8)
        return {"image": png(cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)[:, :, ::-1])}

    @torch.no_grad()
    def som(self, image, granularity=1.8, alpha=0.1, anno_mode=("Mask", "Mark")):
        sam, infer = self._model("som")
        np.random.seed(0)  # mask colors
        im, anns = infer(sam, image, 640, label_mode="1", alpha=alpha, anno_mode=list(anno_mode))  # text_size 640 as upstream
        return {"image": png(im), "bboxes": [a["bbox"] for a in anns]}

    def __call__(self, req: dict) -> dict:
        image = Image.open(io.BytesIO(base64.b64decode(req.pop("image")))).convert("RGB")
        with self.lock:
            return getattr(self, req.pop("tool"))(image, **req)


def serve(path: str, device: str = "cuda") -> socketserver.ThreadingUnixStreamServer:
    """The experts on unix socket `path`, in a background thread."""
    experts = Experts(device)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                reply = experts(json.loads(self.rfile.readline()))
            except Exception as e:  # the agent sees it as a tool error
                reply = {"error": f"{type(e).__name__}: {e}"}
            self.wfile.write(json.dumps(reply).encode() + b"\n")

    server = socketserver.ThreadingUnixStreamServer(path, Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
