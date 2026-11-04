"""SpatialClaw's side, in its env: JSON lines in, answers out. python runner.py <SpatialClaw dir> <model json> [port=socket ...]"""

import asyncio
import base64
import contextlib
import io
import json
import os
import socket
import sys
import threading

out = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)  # SpatialClaw prints; keep stdout for replies
root, model_json = sys.argv[1:3]
sys.path.insert(0, root)


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


def forward(srv, path: str):
    while True:
        c, _ = srv.accept()
        u = socket.socket(socket.AF_UNIX)
        u.connect(path)
        for x, y in ((c, u), (u, c)):
            threading.Thread(target=pipe, args=(x, y), daemon=True).start()


for spec in sys.argv[3:]:  # listening before SpatialClaw probes its servers
    port, path = spec.split("=", 1)
    threading.Thread(target=forward, args=(socket.create_server(("127.0.0.1", int(port))), path), daemon=True).start()

from PIL import Image  # noqa: E402
from spatial_agent.config import SpatialAgentConfig, set_config  # noqa: E402
from spatial_agent.workflow import SpatialAgentWorkflow  # noqa: E402

config = SpatialAgentConfig()
config.update_from_model_json(model_json)
config.work_dir = "/tmp/spatialclaw"
config.generate_report = False
os.makedirs(config.work_dir, exist_ok=True)
set_config(config)
workflow = SpatialAgentWorkflow(config)
lock = threading.Lock()


async def one(req: dict):
    try:
        images = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in req["images"]]
        r = await workflow.arun(instruction=req["instruction"], images=images, session_id=str(req["id"]), defer_report=True)
        reply = {"answer": (r.get("final_answer") or {}).get("text", ""), "termination": r.get("termination_reason"), "steps": r.get("step_count")}
    except Exception as e:  # the victim reports it as a failed query
        reply = {"error": f"{type(e).__name__}: {e}"}
    with lock:
        out.write(json.dumps({"id": req["id"], **reply}) + "\n")


async def main():
    loop, tasks = asyncio.get_running_loop(), set()
    while line := await loop.run_in_executor(None, sys.stdin.readline):
        t = asyncio.create_task(one(json.loads(line)))
        tasks.add(t)
        t.add_done_callback(tasks.discard)
    await asyncio.gather(*tasks)


asyncio.run(main())
