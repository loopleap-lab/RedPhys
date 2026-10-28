"""Qwen3-VL, Cosmos-Reason2, Qwen3.5."""

import torch
from PIL import Image

from ....data import load_rgb
from .base import Family


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
        x = x.reshape(1, t, c, gh // m, m, p, gw // m, m, p).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)  # the processor's patch order
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
