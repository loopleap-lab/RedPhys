"""Supported VLMs by transformers `model_type`."""

from .gemma4 import Gemma4
from .internvl import InternVL
from .qwen import Qwen

FAMILIES = {"qwen3_vl": Qwen, "qwen3_5": Qwen, "internvl": InternVL, "gemma4": Gemma4}
