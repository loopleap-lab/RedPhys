"""Qwen-Image-2.1 behind the OpenAI images API (generations, edits): python server.py [port]."""
import asyncio, base64, hashlib, io, os, subprocess, sys, threading, time
from pathlib import Path

import numpy as np, safetensors.torch, torch, uvicorn
from diffusers import QwenImage21Pipeline
from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import snapshot_download
from PIL import Image

TE, DIT = map(torch.device, os.environ.get("TARGET_GENERATOR_DEVICES", "cuda:0,cuda:1").split(","))  # text encoder, transformer + VAE


class Pipe(QwenImage21Pipeline):
    _execution_device = DIT

    def _get_qwen_prompt_embeds(self, p, i, d):
        return tuple(t if t is None else t.to(DIT) for t in super()._get_qwen_prompt_embeds(p, i, TE))


REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"  # read by install.sh (no refs/main)
snap = Path(snapshot_download("Qwen/Qwen-Image-2.1", revision=REVISION, local_files_only=True))
pipe = Pipe.from_pretrained(snap, torch_dtype=torch.bfloat16)
pipe.text_encoder.to(TE), pipe.transformer.to(DIT), pipe.vae.to(DIT)
with torch.inference_mode():
    for f in snap.rglob("*.safetensors"):  # re-read from disk, checked against the blob's sha256 name
        for _ in range(3):  # large reads are sometimes corrupt: retry
            data = subprocess.run(["dd", f"if={f}", "iflag=direct", "bs=16M"], capture_output=True).stdout
            if hashlib.sha256(data).hexdigest() == Path(os.readlink(f)).name:
                break
        else:
            raise RuntimeError(f"{f}: sha256 mismatch")
        sd = getattr(pipe, f.relative_to(snap).parts[0]).state_dict()
        for k, v in safetensors.torch.load(data).items():
            sd[k].copy_(v)
    a = np.asarray(pipe("A ceramic teapot on a wooden table, soft morning light, a sign reading Qwen Image 2.1",
                        generator=torch.Generator(DIT).manual_seed(1)).images[0], float)
    assert abs(a.mean() - 152.7) < 0.5 and abs(a.std() - 78.59) < 0.5, f"self-test failed: mean {a.mean():.2f}, std {a.std():.2f}"

app, lock = FastAPI(), threading.Lock()


def run(f, images):
    w, h = map(int, f["size"].split("x")) if f.get("size") else (None, None)
    seed = f.get("seed")
    with lock, torch.inference_mode():
        pipe.vae.enable_tiling() if images else pipe.vae.disable_tiling()
        out = pipe(f["prompt"], images or None, f.get("negative_prompt"), float(f.get("true_cfg_scale", 1)), h, w,
                   int(f.get("num_inference_steps", 40)), num_images_per_prompt=int(f.get("n", 1)),
                   generator=None if seed is None else torch.Generator(DIT).manual_seed(int(seed))).images
        torch.cuda.empty_cache()
    if any(np.asarray(i).max() == 0 for i in out):  # black image: exit, server.sh restarts
        threading.Timer(0.5, os._exit, (1,)).start()
        raise HTTPException(503, "generation failed, restarting")
    data = []
    for img in out:
        img.save(buf := io.BytesIO(), "PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
    return {"created": int(time.time()), "data": data}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/v1/images/{op}")
async def images(op: str, r: Request):
    if op == "edits":
        form = await r.form()
        return await asyncio.to_thread(run, dict(form), [Image.open(x.file) for x in form.getlist("image")[:4]])
    return await asyncio.to_thread(run, await r.json(), None)


uvicorn.run(app, host="0.0.0.0", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8091)
