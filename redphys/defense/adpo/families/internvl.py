"""InternVL3.5 (-HF repos)."""

import torch

from .base import Family, resized


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
