"""Loading, encoding and log-probs shared by all families (families/: encoder and preprocessing)."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from PIL import Image

from .families import FAMILIES


def load(model_id: str, device, dtype=torch.bfloat16):
    """(model or None, family); device "auto" splits the model over the GPUs."""
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    kind = AutoConfig.from_pretrained(model_id).model_type
    if kind not in FAMILIES:
        raise ValueError(f"AdPO supports {sorted(FAMILIES)} (InternVL: -HF repos), not {kind!r} ({model_id})")
    vlm = FAMILIES[kind](AutoProcessor.from_pretrained(model_id))
    if device is None:
        return None, vlm
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, **({"device_map": "auto"} if device == "auto" else {}))
    return model if device == "auto" else model.to(device), vlm


def prompt_text(processor, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    think = {"enable_thinking": False} if "enable_thinking" in str(processor.chat_template) else {}
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **think)


def encode(vlm, image: Image.Image, prompt: str, response=None, end: bool = True) -> dict:
    """Labels mark only the response (+ end of turn unless cut)."""
    proc, kw = vlm.processor, {"return_tensors": "pt", "add_special_tokens": False, **vlm.encode_kwargs}
    text = prompt_text(proc, prompt)
    enc = proc(text=[text if response is None else text + response + (vlm.end if end else "")], images=[image], **kw)
    if response is not None:
        n = proc(text=[text], images=[image], **kw)["input_ids"].shape[1]
        enc["labels"] = enc["input_ids"].clone()
        enc["labels"][:, :n] = -100
    return dict(enc)


def collate(encs: list[dict], pad_id: int) -> dict:
    n = max(e["input_ids"].shape[1] for e in encs)
    pad = {"input_ids": pad_id, "attention_mask": 0, "labels": -100, "mm_token_type_ids": 0}
    return {k: torch.cat([F.pad(e[k], (0, n - e[k].shape[1]), value=pad[k]) if k in pad else e[k] for e in encs])
            for k in encs[0]}


def token_logps(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token label log-probs and mask; logits only from the first answer token."""
    labels = batch["labels"]
    has = labels != -100
    first = int(torch.where(has.any(1), has.int().argmax(1), labels.shape[1]).min().clamp(min=1))
    logits = model(**{k: v for k, v in batch.items() if k != "labels"}, use_cache=False,
                   logits_to_keep=labels.shape[1] - first + 1).logits.float()
    target = labels[:, first:]
    mask = target != -100
    logps = torch.gather(logits[:, :-1].log_softmax(-1), 2, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return logps * mask, mask


@torch.no_grad()
def answer(model, vlm, enc: dict, max_new_tokens: int) -> tuple[str, bool]:
    """Greedy (text, ended by itself)."""
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)[0, enc["input_ids"].shape[1]:]
    tok = vlm.processor.tokenizer
    end = len(out) < max_new_tokens or out[-1].item() == tok.convert_tokens_to_ids(vlm.end)
    return tok.decode(out, skip_special_tokens=True).strip(), end


def unwrap(model):
    return getattr(model, "module", model)  # DDP


def trained(model, vlm, train_projector: bool) -> dict:
    names = [vlm.encoder] + ([vlm.projector] if train_projector and vlm.projector else [])
    return {n: getattr(unwrap(model).model, n) for n in names}


@contextmanager
def swapped(model, modules: dict):
    inner = unwrap(model).model
    old = {n: getattr(inner, n) for n in modules}
    for n, m in modules.items():
        setattr(inner, n, m)
    try:
        yield
    finally:
        for n, m in old.items():
            setattr(inner, n, m)
