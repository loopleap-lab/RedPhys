"""Visual Sketchpad (NeurIPS 2024): the PhyModel writes Python against vision tools in a Docker sandbox until TERMINATE."""

import base64
import io
import json
import os
import re
import select
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from ...agent import IMAGE, python_dirs, sandbox_image
from ..server import Victim, VictimConfig, png64, question
from .prompt import SYSTEM, exec_feedback, initial_prompt, parse, parsing_feedback

HERE = Path(__file__).resolve().parent
MOUNTS = dict.fromkeys([*python_dirs(sys.executable), HERE, *(Path(p).resolve() for p in sys.path if p and Path(p).is_dir())])


@dataclass
class SketchpadConfig(VictimConfig):
    max_replies: int = 10
    exec_timeout: int = 60  # seconds per code block, autogen's JupyterCodeExecutor default
    experts_device: str = "cuda"


class Sandbox:
    """One kernel container per question."""

    def __init__(self, sock: str, images: list):
        self.name = f"sketchpad-{uuid.uuid4().hex[:12]}"
        self.p = subprocess.Popen(
            ["docker", "run", "-i", "--rm", "--init", "--pull=never", "--network", "none", "--read-only", "--tmpfs", "/tmp:size=256m",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "128", "--memory", "4g",
             "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "MPLBACKEND=Agg", "-e", f"PYTHONPATH={HERE}",
             "-e", "SKETCHPAD_EXPERTS=/experts.sock", "--name", self.name,
             *(f"--volume={p}:{p}:ro" for p in MOUNTS), f"--volume={sock}:/experts.sock",
             "--entrypoint", sys.executable, IMAGE, str(HERE / "kernel.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.send({"images": [png64(i) for i in images]}, 60)

    def send(self, req: dict, timeout: int) -> dict:
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        if not select.select([self.p.stdout], [], [], timeout)[0]:
            self.close()
            raise TimeoutError(f"no reply in {timeout} s")
        return json.loads(self.p.stdout.readline())

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        self.p.kill()


class SketchpadAgent(Victim):

    def __init__(self, config: SketchpadConfig):
        super().__init__(config)
        from . import experts

        sandbox_image()
        self.sock = str(Path(tempfile.mkdtemp(prefix="sketchpad-")) / "experts.sock")
        self.experts = experts.serve(self.sock, config.experts_device)

    def observe(self, result: dict) -> list:
        """exec_feedback as a message, the displayed images where they were shown."""
        from PIL import Image

        text = exec_feedback(result["exit_code"], result["output"])
        imgs = [Image.open(io.BytesIO(base64.b64decode(b))) for b in result["images"]]
        parts = []
        for i, chunk in enumerate(re.split(r"<<IMG:(\d+)>>\n?", text)):
            if i % 2:
                parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + png64(imgs[int(chunk)])}})
            elif chunk:
                parts.append({"type": "text", "text": chunk})
        return parts

    def run(self, messages, images, **params):
        parts = messages[-1]["content"]
        query = re.sub(r"\n(?=A\. )", "\nSelect from the following choices.\n", question(messages), count=1)  # as its tasks: "(A) ..."
        query = re.sub(r"^([A-D])\. ", r"(\1) ", query, flags=re.M)
        head, tail = initial_prompt("\0", len(images)).split("\0")  # upstream: "<img src=...> question", the image before the question
        convo = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": [{"type": "text", "text": head}, *(p for p in parts if p["type"] == "image_url"),
                                              {"type": "text", "text": f" {query}{tail}"}]}]
        box = Sandbox(self.sock, images)
        try:
            for _ in range(self.config.max_replies):
                msg = self.chat(convo, **params)
                reply = msg.content = msg.content or ""
                convo.append({"role": "assistant", "content": reply})
                code = parse(reply)
                if not code["status"]:
                    if "TERMINATE" in reply:
                        break
                    convo.append({"role": "user", "content": parsing_feedback(code["message"], code["error_code"])})
                    continue
                try:
                    result = box.send({"code": code["content"]}, self.config.exec_timeout)
                except (TimeoutError, json.JSONDecodeError) as e:
                    box, result = Sandbox(self.sock, images), {"exit_code": 1, "output": f"{type(e).__name__}: {e}", "images": []}
                convo.append({"role": "user", "content": self.observe(result)})
        finally:
            box.close()
        msg.content = reply.replace("TERMINATE", "").strip()
        return msg
