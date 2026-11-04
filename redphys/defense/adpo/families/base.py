"""A model family: its vision encoder and the differentiable preprocessing of its processor."""

import torch
from PIL import Image

from ....data import load_rgb


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
