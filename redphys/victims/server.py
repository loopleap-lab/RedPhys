"""serve(name) -> url of a PhyAgent's OpenAI-compatible chat API, for redphys.evaluate(server_url=...)."""

import base64
import io
import json
import os
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai import OpenAI
from PIL import Image

from ..evaluation.metrics import VICTIM
from ..judge import no_reasoning
from ..registry import Registry

VictimRegistry = Registry("victim")


@dataclass
class VictimConfig:
    model: str = "openai/gpt-4.1-mini"  # the PhyModel
    server_url: str = "https://openrouter.ai/api"
    api_key_env: str = "OPENROUTER_API_KEY"


def png64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def view(img: Image.Image, size: int = 768) -> dict:
    """Image part, longer side `size`."""
    img = img.resize([max(1, round(s * size / max(img.size))) for s in img.size], Image.BICUBIC)
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png64(img)}}


def question(messages) -> str:
    """The question RedPhys asks, without its answer-format line (for agents with their own answer protocol)."""
    return next(p["text"] for p in messages[-1]["content"] if p["type"] == "text").replace(f"\n{VICTIM['mc_suffix']}", "")


def _mark(m: dict) -> dict:
    parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"] or ""}]
    k = next((i for i in reversed(range(len(parts))) if parts[i].get("type") == "text"), len(parts) - 1)
    return {**m, "content": [*parts[:k], {**parts[k], "cache_control": {"type": "ephemeral"}}, *parts[k + 1:]]} if parts else m


def cached(client, model: str, messages: list) -> tuple[list, dict]:
    """(messages, extra body) for prompt caching: OpenAI caches by itself; Gemini on OpenRouter only on Google AI Studio, with
    cache_control on the last message and the previous call's last one (read from cache now)."""
    if "openrouter.ai" not in str(client.base_url) or not model.startswith("google/"):
        return messages, {}
    a = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=0)
    marks = {len(messages) - 1, a - 1} if a else {len(messages) - 1}
    return [_mark(m) if i in marks else m for i, m in enumerate(messages)], {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": False}}


class Victim:
    """run(messages, images) -> the answer's message (.content, .cut: hit max_tokens); messages: RedPhys's (system, user)."""

    def __init__(self, config: VictimConfig):
        self.config = config
        self.client = OpenAI(base_url=f"{config.server_url}/v1", api_key=os.environ.get(config.api_key_env, "dummy"))

    def chat(self, messages, **kw):
        messages, extra = cached(self.client, self.config.model, messages)
        c = self.client.chat.completions.create(model=self.config.model, messages=messages,
                                                extra_body={**(no_reasoning(self.client, self.config.model) or {}), **extra}, **kw).choices[0]
        c.message.cut = c.finish_reason == "length"
        return c.message

    def run(self, messages: list, images: list, **params):
        raise NotImplementedError

    def __call__(self, messages: list, **params):
        content = messages[-1]["content"]
        images = [Image.open(io.BytesIO(base64.b64decode(p["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
                  for p in content if isinstance(content, list) and p["type"] == "image_url"]
        if not images:  # RedPhys's server check, "Hi"
            return self.chat(messages, **params)
        return self.run([dict(m) for m in messages], images, **params)


def serve(name: str, port: int = 9000, background: bool = True, config: str | None = None, **params) -> str:
    """Victim `name` on localhost:port; params over configs/victims/<name>.yaml; background: in a thread."""
    from ..configs import build_config

    victim = VictimRegistry.create(name, **build_config("victims", name, VictimRegistry.get_spec(name).config_class, config, params))

    class Chat(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                msg, status = victim(req["messages"], **{k: req[k] for k in ("temperature", "max_tokens") if k in req}), 200
                text, finish = msg.content or "", "length" if msg.cut else "stop"  # length: cut off, as for a PhyModel
            except Exception as e:  # a failed query for the evaluation
                text, finish, status = f"agent error: {e}", "stop", 500
            choice = {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}
            body = json.dumps({"id": "0", "object": "chat.completion", "created": 0, "model": req.get("model", name), "choices": [choice]}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Chat)
    if not background:
        server.serve_forever()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://localhost:{port}"
