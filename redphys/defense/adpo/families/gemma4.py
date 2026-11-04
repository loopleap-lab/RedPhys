"""Gemma 4."""

import torch
import torch.nn.functional as F

from .base import Family, resized


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
