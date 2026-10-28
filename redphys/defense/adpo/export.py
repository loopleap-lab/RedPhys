"""Trained encoder -> visual.safetensors -> servable checkpoint."""

import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .models import trained


def save_visual(model, vlm, train_projector: bool, path) -> None:
    sd = {vlm.prefixes[n] + k: v.detach().to(torch.bfloat16).cpu().contiguous()
          for n, m in trained(model, vlm, train_projector).items() for k, v in m.state_dict().items()}
    save_file(sd, f"{path}.tmp")
    os.replace(f"{path}.tmp", path)  # complete or absent


def merge(base: str, visual_path, out_dir) -> Path:
    """Copy of base with the trained tensors replaced; built beside out_dir and renamed, so out_dir is complete or absent."""
    from huggingface_hub import snapshot_download
    src = Path(base) if Path(base).is_dir() else Path(snapshot_download(base))
    sd, seen = load_file(str(visual_path)), set()
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.suffix != ".safetensors":
            if f.is_file():
                shutil.copy(f, tmp / f.name)
            continue
        shard = load_file(str(f))
        hit = sd.keys() & shard.keys()
        if bad := [k for k in hit if shard[k].shape != sd[k].shape]:
            raise ValueError(f"{visual_path}: shapes differ from {f.name}, e.g. {bad[:3]}")
        shard.update({k: sd[k].to(shard[k].dtype) for k in hit})
        save_file(shard, str(tmp / f.name), metadata={"format": "pt"})
        seen |= hit
    missing = sorted(sd.keys() - seen)
    if not sd or missing:
        raise ValueError(f"{visual_path}: {len(missing)} of {len(sd)} tensors not in {base}, e.g. {missing[:3]}")
    shutil.rmtree(out_dir, ignore_errors=True)  # an incomplete one
    tmp.rename(out_dir)
    return out_dir
