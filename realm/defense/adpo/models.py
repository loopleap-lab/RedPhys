"""Supported VLMs: per-family encoder and differentiable preprocessing; shared encoding and log-probs."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from PIL import Image

from ...data import load_rgb


def resized(x: torch.Tensor, size, ip) -> torch.Tensor:
    """The processor's uint8 resize forward, the float resize's gradient."""
    from torchvision.transforms.v2.functional import resize
    from transformers.image_utils import pil_torch_interpolation_mapping
    kw = dict(interpolation=pil_torch_interpolation_mapping[ip.resample], antialias=True)
    f = resize(x, list(size), **kw).clamp(0, 1)
    return f + (resize((x.detach() * 255).round().to(torch.uint8), list(size), **kw) / 255 - f).detach()


class Family:
    encoder: str
    projector = None  # module beside the encoder
    projector_params = ()  # or inside it
    prefixes: dict  # module -> checkpoint prefix
    encode_kwargs = {}
    end = "<|im_end|>"  # end of turn

    def __init__(self, processor):
        self.processor = processor

    def sized(self, path, max_pixels=None) -> Image.Image:
        return load_rgb(path)

    def normalize(self, x):
        ip = self.processor.image_processor
        if not ip.do_normalize:
            return x
        return (x - x.new_tensor(ip.image_mean).view(3, 1, 1)) / x.new_tensor(ip.image_std).view(3, 1, 1)


class Qwen(Family):
    """Qwen3-VL, Cosmos-Reason2, Qwen3.5."""
    encoder, projector_params = "visual", ("merger.", "deepstack_merger_list.")
    prefixes = {"visual": "model.visual."}
    encode_kwargs = {"do_resize": False}

    def sized(self, path, max_pixels=None) -> Image.Image:
        """Resized once, as the processor would."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        ip, img = self.processor.image_processor, load_rgb(path)
        h, w = smart_resize(img.height, img.width, factor=ip.patch_size * ip.merge_size, min_pixels=ip.size["shortest_edge"],
                            max_pixels=max_pixels or ip.size["longest_edge"])
        return img if (w, h) == img.size else img.resize((w, h), Image.BICUBIC)

    def pixel_inputs(self, x: torch.Tensor) -> dict:
        ip = self.processor.image_processor
        p, m, t = ip.patch_size, ip.merge_size, ip.temporal_patch_size
        x = self.normalize(x).unsqueeze(0).expand(t, -1, -1, -1)
        c, h, w = x.shape[1:]
        gh, gw = h // p, w // p
        x = x.reshape(1, t, c, gh // m, m, p, gw // m, m, p).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return {"pixel_values": x.reshape(gh * gw, -1), "image_grid_thw": torch.tensor([[1, gh, gw]], device=x.device)}

    def features(self, encoder, inputs: dict) -> torch.Tensor:
        """phi: the input of the merger (projector)."""
        out = []
        hook = encoder.merger.register_forward_hook(lambda m, i, o: out.append(i[0]))
        try:
            encoder(inputs["pixel_values"].to(encoder.dtype), grid_thw=inputs["image_grid_thw"])
        finally:
            hook.remove()
        return out[0].float()


class InternVL(Family):
    """InternVL3.5 (-HF repos)."""
    encoder, projector = "vision_tower", "multi_modal_projector"
    prefixes = {"vision_tower": "vision_tower.", "multi_modal_projector": "multi_modal_projector."}

    def pixel_inputs(self, x: torch.Tensor) -> dict:
        from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
        ip = self.processor.image_processor
        th, tw = ip.size["height"], ip.size["width"]
        cols, rows = get_optimal_tiled_canvas(x.shape[-2:], (th, tw), ip.min_patches, ip.max_patches)
        canvas = resized(x, (th * rows, tw * cols), ip)
        tiles = [canvas[:, r * th:(r + 1) * th, c * tw:(c + 1) * tw] for r in range(rows) for c in range(cols)]
        tiles += [resized(x, (th, tw), ip)] if len(tiles) > 1 else []
        return {"pixel_values": self.normalize(torch.stack(tiles))}

    def features(self, encoder, inputs: dict) -> torch.Tensor:
        return encoder(pixel_values=inputs["pixel_values"].to(encoder.dtype)).last_hidden_state.float()


class Gemma4(Family):
    """Gemma 4."""
    encoder, projector = "vision_tower", "embed_vision"
    prefixes = {"vision_tower": "model.vision_tower.", "embed_vision": "model.embed_vision."}
    end = "<turn|>"

    def pixel_inputs(self, x: torch.Tensor) -> dict:
        from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
        ip = self.processor.image_processor
        p, k, n = ip.patch_size, ip.pooling_kernel_size, ip.max_soft_tokens * ip.pooling_kernel_size ** 2
        h, w = get_aspect_ratio_preserving_size(*x.shape[-2:], patch_size=p, max_patches=n, pooling_kernel_size=k)
        x = self.normalize(resized(x, (h, w), ip) if (h, w) != tuple(x.shape[-2:]) else x)
        gh, gw = h // p, w // p
        patches = x.reshape(3, gh, p, gw, p).permute(1, 3, 2, 4, 0).reshape(gh * gw, -1)
        pos = torch.stack(torch.meshgrid(torch.arange(gw), torch.arange(gh), indexing="xy"), -1).reshape(-1, 2).to(x.device)
        return {"pixel_values": F.pad(patches, (0, 0, 0, n - gh * gw)).unsqueeze(0),
                "image_position_ids": F.pad(pos, (0, 0, 0, n - gh * gw), value=-1).unsqueeze(0)}

    def features(self, encoder, inputs: dict) -> torch.Tensor:
        return encoder(pixel_values=inputs["pixel_values"].to(encoder.dtype),
                       pixel_position_ids=inputs["image_position_ids"]).last_hidden_state.float()


FAMILIES = {"qwen3_vl": Qwen, "qwen3_5": Qwen, "internvl": InternVL, "gemma4": Gemma4}


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
