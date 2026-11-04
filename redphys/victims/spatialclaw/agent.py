"""SpatialClaw (github.com/NVlabs/SpatialClaw, NVIDIA Source Code License-NC, installed separately), run unchanged in a Docker sandbox."""

import atexit
import contextlib
import http.client
import json
import os
import re
import socket
import socketserver
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from PIL import Image

from ...agent import python_dirs, sandbox_image
from ...data import PROJECT_ROOT
from ...judge import no_reasoning
from ..server import Victim, VictimConfig, cached, png64, question

RUNNER = Path(__file__).resolve().parent / "runner.py"
ROLES = ("main", "planning", "general", "vlm", "vlm_grounding", "reflection")
LLM_PORT = 18000  # the PhyModel inside the sandbox


@dataclass
class SpatialClawConfig(VictimConfig):
    root: str = "third_party/SpatialClaw"  # bash install.sh spatialclaw; relative to the RedPhys root
    python: str = "third_party/SpatialClaw/.venv-agent/bin/python"  # its environment
    gpu_server: str = "http://127.0.0.1:9700"  # its GPU tool server (logs/gpu_server.json)
    max_tokens: int = 4096  # per call, every role (its configs: 16K-131K, thinking on)
    timeout: int = 1800  # seconds per question


def _forward(path: str, host: str, port: int):
    """Unix socket `path` -> TCP host:port, in background threads."""
    def pipe(a, b):
        try:
            while data := a.recv(65536):
                b.sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):  # shutdown, not just close, wakes the other thread's recv
                with contextlib.suppress(OSError):
                    s.shutdown(socket.SHUT_RDWR)
                s.close()

    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(64)

    def loop():
        while True:
            c, _ = srv.accept()
            t = socket.create_connection((host, port))
            for x, y in ((c, t), (t, c)):
                threading.Thread(target=pipe, args=(x, y), daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class SpatialClawAgent(Victim):

    def __init__(self, config: SpatialClawConfig):
        super().__init__(config)
        root, python = (Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in (config.root, config.python))
        gpu = urlparse(config.gpu_server)
        registry = json.loads((root / "spatial_agent/logs/gpu_server.json").read_text())
        server = max((e for e in registry.values() if e.get("http_port") == gpu.port), key=lambda e: e.get("create_time", ""), default=None)  # newest: a killed server leaves its entry
        if server is None:
            raise RuntimeError(f"no SpatialClaw GPU server on port {gpu.port} in its gpu_server.json: start launch_gpu_server")
        work = Path(tempfile.mkdtemp(prefix="spatialclaw-"))
        for d in ("logs", "socks", "sessions"):
            (work / d).mkdir()
        (work / "logs/gpu_server.json").write_text(json.dumps({"redphys": {**server, "ip": "127.0.0.1"}}))
        (work / "model.json").write_text(json.dumps({"llm_model": config.model, "llm_base_url": f"http://127.0.0.1:{LLM_PORT}/v1",
                                                     "llm_api_key": "proxy", "roles": {r: {"max_tokens": config.max_tokens, "temperature": 0.0}
                                                                                       for r in ROLES}}))
        self._proxy(str(work / "socks/llm.sock"))
        _forward(str(work / "socks/gpu.sock"), gpu.hostname, gpu.port)
        self.name = f"spatialclaw-{uuid.uuid4().hex[:12]}"
        self.p = subprocess.Popen(
            ["docker", "run", "-i", "--rm", "--init", "--pull=never", "--network", "none", "--read-only", "--tmpfs", "/tmp:size=4g",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "4096", "--memory", "16g",
             "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "MPLBACKEND=Agg", "-e", "HF_HUB_OFFLINE=1",
             *(f"-e={v}_NUM_THREADS=4" for v in ("OMP", "OPENBLAS", "MKL")), "--name", self.name,
             *(f"--volume={p}:{p}:ro" for p in dict.fromkeys([*python_dirs(python), root])), f"--volume={work / 'logs'}:{root / 'spatial_agent/logs'}",
             f"--volume={work}:/work:ro", f"--volume={work / 'socks'}:/socks", f"--volume={work / 'sessions'}:/tmp/spatialclaw",
             f"--volume={RUNNER}:/runner.py:ro", "-w", str(root),
             "--entrypoint", str(python), sandbox_image(), "/runner.py", str(root), "/work/model.json",
             f"{LLM_PORT}=/socks/llm.sock", f"{gpu.port}=/socks/gpu.sock"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(work / "stderr.log", "w"), text=True)
        atexit.register(subprocess.run, ["docker", "rm", "-f", self.name], capture_output=True)
        self.sessions = work / "sessions"  # SpatialClaw's per-question traces
        self.replies, self.cv, self.lock = {}, threading.Condition(), threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _proxy(self, path: str):
        """The PhyModel API on unix socket `path`: thinking off, caching on, the key added."""
        agent = self

        class Proxy(BaseHTTPRequestHandler):
            def relay(self, method):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))) or None
                if body and self.path.endswith("/chat/completions"):
                    req = json.loads(body)
                    req["messages"], extra = cached(agent.client, agent.config.model, req["messages"])
                    body = json.dumps({**req, **(no_reasoning(agent.client, agent.config.model) or {}), **extra}).encode()
                req = urllib.request.Request(agent.config.server_url + self.path, body, method=method,
                                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {agent.client.api_key}"})
                try:
                    with urllib.request.urlopen(req, timeout=agent.config.timeout) as r:
                        status, data = r.status, r.read()
                except urllib.error.HTTPError as e:
                    status, data = e.code, e.read()
                except (OSError, http.client.HTTPException) as e:  # also a truncated body
                    status, data = 502, json.dumps({"error": {"message": str(e)}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                self.relay("POST")

            def do_GET(self):
                self.relay("GET")

            def log_message(self, *args):
                pass

        threading.Thread(target=_UnixHTTPServer(path, Proxy).serve_forever, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            r = json.loads(line)
            with self.cv:
                self.replies[r["id"]] = r
                self.cv.notify_all()

    @staticmethod
    def _shrink(session: Path):
        """Session images as JPEG (PNGs take about 1 MB each)."""
        names = []
        for p in session.rglob("*.png"):
            Image.open(p).convert("RGB").save(p.with_suffix(".jpg"), quality=85)
            p.unlink()
            names.append(p.stem)
        trace = session / "trace.jsonl"
        if names and trace.exists():
            trace.write_text(re.sub(rf"\b({'|'.join(map(re.escape, names))})\.png", r"\1.jpg", trace.read_text()))

    def run(self, messages, images, **params):
        rid = uuid.uuid4().hex
        with self.lock:
            self.p.stdin.write(json.dumps({"id": rid, "instruction": question(messages), "images": [png64(i) for i in images]}) + "\n")
            self.p.stdin.flush()
        with self.cv:
            if not self.cv.wait_for(lambda: rid in self.replies or self.p.poll() is not None, self.config.timeout):
                raise TimeoutError(f"no SpatialClaw answer in {self.config.timeout} s")
        r = self.replies.pop(rid, None) or {"error": f"SpatialClaw exited ({self.p.poll()})"}
        self._shrink(self.sessions / f"session-{rid}")
        if "error" in r:
            raise RuntimeError(r["error"])
        return SimpleNamespace(content=r["answer"], cut=False)
