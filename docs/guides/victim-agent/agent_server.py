"""A tool-using VLM agent behind an OpenAI-compatible chat API (a RedPhys victim); evaluate with server_url="http://localhost:9000"."""
import base64
import inspect
import io
import json
import math
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai import OpenAI
from PIL import Image

client = OpenAI(base_url=os.environ.get("AGENT_BASE_URL", "https://openrouter.ai/api/v1"), api_key=os.environ["OPENROUTER_API_KEY"])
MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-4.1-mini")  # needs image input and tool calls
MAX_STEPS = 4
HINT = "\n\nImages are numbered from 1; an image a tool returns gets the next number. Use the tools to check details before you answer."
SCHEMA = {int: {"type": "integer"}, float: {"type": "number"}, str: {"type": "string"}, Image.Image: {"type": "integer"},  # an image's number
          list: {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}  # bbox_2d
TOOLS, SPECS = {}, []


def tool(fn):  # the typed signature and docstring are the spec, as in LangChain, smolagents and the Agents SDK
    params = {n: SCHEMA[p.annotation] for n, p in inspect.signature(fn).parameters.items()}
    TOOLS[fn.__name__] = fn
    SPECS.append({"type": "function", "function": {"name": fn.__name__, "description": fn.__doc__,
                                                   "parameters": {"type": "object", "properties": params, "required": list(params)}}})
    return fn


def chat(messages, **kwargs):
    return client.chat.completions.create(model=MODEL, messages=messages, **kwargs).choices[0].message


def view(img):
    img = img.resize([max(1, round(s * 768 / max(img.size))) for s in img.size], Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def ask(img, prompt):
    return chat([{"role": "user", "content": [view(img), {"type": "text", "text": prompt}]}], max_tokens=300).content


@tool
def image_zoom_in_tool(img_idx: Image.Image, bbox_2d: list):
    "Zoom into a region of an image. bbox_2d is [x1, y1, x2, y2], relative coordinates in 0-1000."
    x1, x2 = sorted(int(min(max(v, 0), 1000) * img_idx.width / 1000) for v in bbox_2d[::2])
    y1, y2 = sorted(int(min(max(v, 0), 1000) * img_idx.height / 1000) for v in bbox_2d[1::2])
    return img_idx.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))


@tool
def image_rotate_tool(img_idx: Image.Image, angle: int):
    "Rotate an image counterclockwise by angle degrees."
    return img_idx.rotate(angle, expand=True)


@tool
def detect_objects(img_idx: Image.Image, query: str):
    "Find objects in an image; returns a label and a bbox_2d (relative 0-1000) for each."
    return ask(img_idx, f"Locate every {query}. One line per object: label, x1, y1, x2, y2 (relative 0-1000). Reply 'none found' if absent.")


@tool
def ocr(img_idx: Image.Image):
    "Read the text, signs and numbers in an image."
    return ask(img_idx, "Transcribe all text, signs and numbers you can see, and where. Reply 'none' if there is none.")


@tool
def calculator(expression: str):
    "Evaluate an arithmetic expression (sqrt, sin, pow, ...)."
    if not re.fullmatch(r"[0-9a-z_+\-*/().,%\s]+", expression) or "__" in expression or "**" in expression:
        raise ValueError("arithmetic and math functions only; use pow(x, y)")
    return str(eval(expression, {"__builtins__": {}}, vars(math)))


def call(name, args, images):
    for n, p in inspect.signature(TOOLS[name]).parameters.items():
        if p.annotation is Image.Image:
            args[n] = images[min(max(int(args[n]), 1), len(images)) - 1]
    out = TOOLS[name](**args)
    if isinstance(out, Image.Image):
        images.append(out)
        return f"image {len(images)}"
    return out


def agent(messages, **params):
    content = messages[-1]["content"]
    images = [Image.open(io.BytesIO(base64.b64decode(p["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
              for p in content if isinstance(content, list) and p["type"] == "image_url"]
    messages = [dict(m) for m in messages]
    if not images:  # RedPhys's server check: a text-only "Hi"
        return chat(messages, **params).content
    if messages[0]["role"] == "system":
        messages[0]["content"] += HINT
    for step in range(MAX_STEPS):
        msg = chat(messages, tools=SPECS if step < MAX_STEPS - 1 else None, **params)
        if not msg.tool_calls:
            return msg.content
        messages.append(msg.model_dump(exclude_none=True))
        new = []
        for c in msg.tool_calls:
            n, args = len(images), json.loads(c.function.arguments)
            print(f"  {c.function.name} {args}", file=sys.stderr)
            try:
                reply = call(c.function.name, args, images)
            except Exception as e:  # the model sees the error and can retry
                reply = f"error: {e}"
            new += [view(i) for i in images[n:]]
            messages.append({"role": "tool", "tool_call_id": c.id, "content": reply})
        if new:
            messages.append({"role": "user", "content": new})  # the chat API takes no images in tool messages


class Chat(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            text, status = agent(req["messages"], **{k: req[k] for k in ("temperature", "max_tokens") if k in req}), 200
        except Exception as e:  # a failed query for the evaluation
            text, status = f"agent error: {e}", 500
        choice = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        body = json.dumps({"id": "0", "object": "chat.completion", "created": 0, "model": req.get("model", "agent"), "choices": [choice]}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("", int(os.environ.get("AGENT_PORT", 9000))), Chat).serve_forever()
