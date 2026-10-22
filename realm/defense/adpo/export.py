"""Trained encoder -> visual.safetensors -> servable checkpoint."""

import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .models import trained


def save_visual(model, vlm, train_projector: bool, path) -> None:
    sd = {vlm.prefixes[n] + k: v.detach().to(torch.bfloat16).cpu().contiguous()
          for n, m in trained(model, vlm, train_projector).items() for k, v in m.state_dict().items()}
    save_file(sd, str(path))


def merge(base: str, visual_path, out_dir) -> Path:
    """Copy of base with the trained tensors replaced."""
    from huggingface_hub import snapshot_download
    src = Path(base) if Path(base).is_dir() else Path(snapshot_download(base))
    sd, seen = load_file(str(visual_path)), set()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.suffix != ".safetensors":
            if f.is_file():
                shutil.copy(f, out_dir / f.name)
            continue
        shard = load_file(str(f))
        hit = sd.keys() & shard.keys()
        if bad := [k for k in hit if shard[k].shape != sd[k].shape]:
            raise ValueError(f"{visual_path}: shapes differ from {f.name}, e.g. {bad[:3]}")
        shard.update({k: sd[k].to(shard[k].dtype) for k in hit})
        save_file(shard, str(out_dir / f.name), metadata={"format": "pt"})
        seen |= hit
    missing = sorted(sd.keys() - seen)
    if not sd or missing:
        raise ValueError(f"{visual_path}: {len(missing)} of {len(sd)} tensors not in {base}, e.g. {missing[:3]}")
    return out_dir
