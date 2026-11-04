"""Tool-calling PhyAgent: the PhyModel calls tools for up to max_steps rounds."""

import inspect
import json
import math
import re
from dataclasses import dataclass

from PIL import Image

from ..server import Victim, VictimConfig, png64, view

HINT = "\n\nImages are numbered from 0; an image a tool returns gets the next number. Use the tools to check details before you answer."
FINAL = "No more tool calls: give your final answer now, in the format the question asks for."  # models keep calling otherwise
SCHEMA = {int: {"type": "integer"}, float: {"type": "number"}, str: {"type": "string"}, Image.Image: {"type": "integer"},  # an image's number
          list: {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}  # bbox_2d
TOOLS, SPECS = {}, []


def tool(fn):  # typed signature + docstring -> spec; the first argument is the agent
    params = {n: SCHEMA[p.annotation] for n, p in list(inspect.signature(fn).parameters.items())[1:]}
    TOOLS[fn.__name__] = fn
    SPECS.append({"type": "function", "function": {"name": fn.__name__, "description": fn.__doc__,
                                                   "parameters": {"type": "object", "properties": params, "required": list(params)}}})
    return fn


@tool
def image_zoom_in_tool(agent, img_idx: Image.Image, bbox_2d: list, label: str):
    "Zoom in on a region of an image: bbox_2d [x1, y1, x2, y2] in relative coordinates 0-1000, label the object in it."
    w, h = img_idx.size
    x1, x2 = sorted(min(max(v, 0), 1000) * w / 1000 for v in bbox_2d[::2])
    y1, y2 = sorted(min(max(v, 0), 1000) * h / 1000 for v in bbox_2d[1::2])
    if min(x2 - x1, y2 - y1) < 32:  # as Qwen-Agent's: grown about the center to 32 px, kept inside
        r = 32 / max(min(x2 - x1, y2 - y1), 1e-6)
        hw, hh = min(w, (x2 - x1) * r) / 2, min(h, (y2 - y1) * r) / 2
        cx, cy = min(max((x1 + x2) / 2, hw), w - hw), min(max((y1 + y2) / 2, hh), h - hh)
        x1, x2, y1, y2 = cx - hw, cx + hw, cy - hh, cy + hh
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    return img_idx.crop((x1, y1, x2, y2)).resize(smart_resize(x2 - x1, y2 - y1), Image.BICUBIC)  # float box, as Qwen-Agent's


def smart_resize(w, h, factor=32, min_pixels=256 * 32 * 32, max_pixels=12845056):
    """Qwen-Agent's (image_zoom_in_qwen3vl.py): sides multiples of factor, area within [min_pixels, max_pixels]."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        b = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / b / factor) * factor, math.floor(w / b / factor) * factor
    elif hb * wb < min_pixels:
        b = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * b / factor) * factor, math.ceil(w * b / factor) * factor
    return wb, hb


@tool
def image_rotate_tool(agent, img_idx: Image.Image, angle: int):
    "Rotate an image counterclockwise by angle degrees."
    return img_idx.rotate(angle, expand=True)


@tool
def detect_objects(agent, img_idx: Image.Image, query: str):
    "Find objects in an image; returns a label and a bbox_2d (relative 0-1000) for each."
    return agent.ask(img_idx, f"Locate every {query}. One line per object: label, x1, y1, x2, y2 (relative 0-1000). Reply 'none found' if absent.")


@tool
def ocr(agent, img_idx: Image.Image):
    "Read the text, signs and numbers in an image."
    return agent.ask(img_idx, "Transcribe all text, signs and numbers you can see, and where. Reply 'none' if there is none.")


@tool
def calculator(agent, expression: str):
    "Evaluate an arithmetic expression (sqrt, sin, pow, ...)."
    if not re.fullmatch(r"[0-9a-z_+\-*/().,%\s]+", expression) or "__" in expression or "**" in expression:
        raise ValueError("arithmetic and math functions only; use pow(x, y)")
    return str(eval(expression, {"__builtins__": {}}, vars(math)))


@dataclass
class ToolsConfig(VictimConfig):
    max_steps: int = 4


class ToolsAgent(Victim):

    def ask(self, img, prompt):
        return self.chat([{"role": "user", "content": [view(img), {"type": "text", "text": prompt}]}], max_tokens=300).content

    def call(self, name, args, images):
        for n, p in list(inspect.signature(TOOLS[name]).parameters.items())[1:]:
            if p.annotation is Image.Image:
                args[n] = images[min(max(int(args[n]), 0), len(images) - 1)]  # 0-based, as Qwen-Agent
        out = TOOLS[name](self, **args)
        if isinstance(out, Image.Image):
            images.append(out)
            return f"image {len(images) - 1}"
        return out

    def run(self, messages, images, **params):
        if messages[0]["role"] == "system":
            messages[0]["content"] += HINT
        last = self.config.max_steps - 1
        for step in range(self.config.max_steps):
            msg = self.chat(messages, tools=SPECS if step < last else None, **params)
            if not msg.tool_calls and "<tool_call>" in (msg.content or ""):  # a call written as text (unparsed, or no tools offered)
                messages += [{"role": "assistant", "content": msg.content.split("<tool_call>")[0]}, {"role": "user", "content": FINAL}]
                return self.chat(messages, **params)
            if not msg.tool_calls:
                return msg
            messages.append(msg.model_dump(exclude_none=True))
            new = []
            for c in msg.tool_calls:
                n = len(images)
                try:
                    reply = self.call(c.function.name, json.loads(c.function.arguments), images)
                except Exception as e:  # the model sees the error and can retry
                    reply = f"error: {e}"
                new += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + png64(i)}} for i in images[n:]]  # as returned
                messages.append({"role": "tool", "tool_call_id": c.id, "content": reply})
            new += [{"type": "text", "text": FINAL}] if step == last - 1 else []
            if new:
                messages.append({"role": "user", "content": new})  # the chat API takes no images in tool messages
        return msg
