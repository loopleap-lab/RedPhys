"""SceneTAP (Cao et al., CVPR 2025): an LLM plans scene-coherent adversarial text, TextDiffuser-2 writes it into the scene."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFont
from torchvision import transforms

from ..base_attack import AttackResult, BaseAttack, LLMClientMixin, target_text
from ...data import Sample, image_data_url, weight
from ...judge import clean_llm_text, no_reasoning
from .config import ADJUST, ASSETS, PLAN, TARGET_LINE, SceneTAPConfig

SOM = Path(__file__).parents[1] / "physpatch" / "assets" / "som"  # SoM's SAM adapter and visualizer

PLAN_FIELDS = ("image_analysis", "correct_answer", "incorrect_answer", "adversarial_text", "text_position_number", "text_placement",
               "short_caption_with_adversarial_text")


def _schema(name: str, props: dict) -> dict:
    """Structured-output response_format, as the official pydantic models."""
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "properties": props, "required": list(props), "additionalProperties": False}}}


PLAN_FORMAT = _schema("PlanSom", {k: {"type": "integer" if k == "text_position_number" else "string"} for k in PLAN_FIELDS})
ADJUST_FORMAT = _schema("PlanSomAdjust", {"adjust_explanation": {"type": "string"},
                                          "adjust_plan": PLAN_FORMAT["json_schema"]["schema"]})


def largest_box(mask: np.ndarray) -> tuple:
    """Largest rectangle inside a mask that is wider than tall: (x, y, w, h), or zeros if there is none."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        return 0, 0, 0, 0
    m = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    heights, best, area = np.zeros(m.shape[1], int), (0, 0, 0, 0), 0
    for i, row in enumerate(m):  # histogram of column heights ending at row i
        heights = np.where(row, heights + 1, 0)
        h, n, stack = heights.tolist(), len(heights), []
        left, right = [0] * n, [n - 1] * n
        for j in range(n):
            while stack and h[stack[-1]] >= h[j]:
                right[stack.pop()] = j - 1
            left[j] = stack[-1] + 1 if stack else 0
            stack.append(j)
        for j in range(n):
            w = right[j] - left[j] + 1
            hh = min(h[j], w - 1)  # taller than wide: keep the widest part that is wider than tall
            if hh > 0 and w * hh > area:
                area, best = w * hh, (left[j], i - hh + 1, w, hh)
    return best[0] + int(cols[0]), best[1] + int(rows[0]), best[2], best[3]


def fit_text_box(text: str, left: float, top: float, right: float, bottom: float, font: ImageFont.FreeTypeFont, tol: float) -> tuple:
    """Shrink the box to the text's aspect ratio (centered) when it differs by more than tol."""
    w, h = right - left, bottom - top
    _, _, tw, th = font.getbbox(text)  # (right, bottom) from the origin, as the old FreeTypeFont.getsize
    text_ratio, box_ratio = th / tw, h / w
    if abs(box_ratio - text_ratio) <= tol:
        return int(left), int(top), int(right), int(bottom)
    new_w, new_h = (h / text_ratio, h) if text_ratio > box_ratio else (w, w * text_ratio)
    left, top = left + (w - new_w) / 2, top + (h - new_h) / 2
    return int(left), int(top), int(left + new_w), int(top + new_h)


class SceneTAPAttack(LLMClientMixin, BaseAttack):

    def __init__(self, config: SceneTAPConfig):
        super().__init__(config)
        self.llm()  # fails here without llm_model, a key or a server
        self.font = ImageFont.truetype(str(weight(ASSETS / config.font, "scenetap")), config.font_size)
        self._som = self._diffuser = None

    def _load(self):
        if self._som:
            return
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sys.path.insert(0, str(SOM))
        from detectron2.data import MetadataCatalog
        from task_adapter.utils.visualizer import Visualizer
        from .textdiffuser import TextDiffuser
        c = self.config
        sam = sam_model_registry["vit_h"](checkpoint=str(weight(c.sam_checkpoint or SOM.parent / "checkpoints" / "sam_vit_h_4b8939.pth", "scenetap"))).eval().to(c.device)
        self._diffuser = TextDiffuser(c.sd_model, c.td_model, c.device, c.td_revision)
        self._som = (SamAutomaticMaskGenerator(sam), Visualizer, MetadataCatalog.get("coco_2017_train_panoptic"))  # last: set once all loaded

    def _segment(self, image: Image.Image) -> tuple:
        """SoM (SAM, numbered marks): (marked image at the input size, at least the mark size; regions in mark order)."""
        generator, Visualizer, metadata = self._som
        c = self.config
        small = np.asarray(transforms.Resize(c.mark_size, interpolation=transforms.InterpolationMode.BICUBIC)(image))
        H, W = small.shape[:2]
        with torch.autocast(torch.device(c.device).type, dtype=torch.float16):  # as upstream
            regions = sorted(generator.generate(small), key=lambda a: a["area"], reverse=True)
        if not regions:
            raise RuntimeError("SAM found no regions")
        f = c.mark_filter

        def keep(a):  # not too small by area, bbox or inscribed rectangle
            if a["area"] < H * W / f / f or a["bbox"][2] < W / f or a["bbox"][3] < H / f:
                return False
            _, _, w, h = largest_box(a["segmentation"])
            return w >= W / f and h >= H / f

        regions = [a for a in regions if keep(a)] or regions  # none left: all
        visual = Visualizer(small, metadata=metadata)
        for number, a in enumerate(regions, 1):
            visual.draw_binary_mask_with_number(a["segmentation"], text=str(number), label_mode="1", alpha=c.mark_alpha,
                                                anno_mode=["Mask", "Mark"])
        marked = Image.fromarray(visual.output.get_image())
        return (marked if min(image.size) < c.mark_size else marked.resize(image.size)), regions

    def _ask(self, messages: list, response_format: dict) -> dict:
        c, client = self.config, self.llm()
        out = client.chat.completions.create(model=c.llm_model, messages=messages, temperature=c.temperature, top_p=c.top_p,
                                             max_tokens=c.max_tokens, response_format=response_format, extra_body=no_reasoning(client, c.llm_model))
        choice = out.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(f"LLM reply cut off at max_tokens={c.max_tokens}")
        return json.loads(clean_llm_text(choice.message.content))

    def _plan(self, image: Image.Image, marked: Image.Image, question: str, correct: str, target: str) -> tuple:
        """(first plan, revised plan): plan, then review the plan against the image and its map, in one conversation."""
        text = (f"Image 0 is the original image, image 1 is the corresponding segmentation map. Observe the image and the corresponding "
                f"segmentation map carefully. Question to attack: {question}. Correct answer: {correct}." + TARGET_LINE.format(target=target)
                + " Please provide a detailed, step-by-step plan for achieving this goal.")
        content = [part for i, img in enumerate((image, marked))
                   for part in ({"type": "text", "text": f"Image {i}"}, {"type": "image_url", "image_url": {"url": image_data_url(img, "JPEG"), "detail": "auto"}})]
        messages = [{"role": "system", "content": PLAN}, {"role": "user", "content": [*content, {"type": "text", "text": text}]}]
        first = self._ask(messages, PLAN_FORMAT)
        messages += [{"role": "assistant", "content": " ".join(f"{k}={first[k]!r}" for k in PLAN_FIELDS)},  # the pydantic str()
                     {"role": "user", "content": [{"type": "text", "text": ADJUST}]}]
        return first, self._ask(messages, ADJUST_FORMAT)["adjust_plan"]

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        c = self.config
        target, correct = target_text(c, sample), sample.metadata.get("attack_source_text")
        if not correct:
            raise ValueError(f"scenetap needs the correct answer for sample {sample.id} (labels= or targets=)")
        self._load()
        image = sample.images[0]
        marked, regions = self._segment(image)
        first, plan = self._plan(image, marked, sample.question, correct, target)

        number = int(plan["text_position_number"])
        mask = regions[number - 1 if 1 <= number <= len(regions) else 0]["segmentation"]  # out of range: the largest region
        x, y, w, h = largest_box(mask)
        if not w:
            raise RuntimeError(f"no rectangle wider than tall in region {number}")
        mh, mw = mask.shape
        text = plan["adversarial_text"]
        if not text.strip():
            raise RuntimeError("empty adversarial_text in the plan")
        box = fit_text_box(text, x / mw * image.width, y / mh * image.height, (x + w) / mw * image.width, (y + h) / mh * image.height,
                           self.font, c.aspect_tolerance)

        generator = torch.Generator().manual_seed(random.Random(f"{c.seed}-{sample.id}").getrandbits(63))  # per sample: resumes alike
        adv = self._diffuser.write(image, box, text, plan["short_caption_with_adversarial_text"], steps=c.steps, guidance=c.guidance,
                                   scale=c.region_scale, n=c.num_candidates, generator=generator)[0]
        return AttackResult(adversarial_sample=adv, perturbation_norm=None,  # not a perturbation
                            metadata={"adversarial_text": text, "box": box,
                                      "text_position_number": number, "plan": plan, "first_plan": first})
