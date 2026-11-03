"""Surrogates: `unified` (below; the default) or `default` (each attack's own)."""

UNIFIED_SURROGATES = (
    {"hf": "openai/clip-vit-base-patch16", "clip": "ViT-B/16", "open_clip": ("ViT-B-16", "openai")},
    {"hf": "openai/clip-vit-base-patch32", "clip": "ViT-B/32", "open_clip": ("ViT-B-32", "openai")},
    {"hf": "laion/CLIP-ViT-g-14-laion2B-s12B-b42K", "clip": None, "open_clip": ("ViT-g-14", "laion2b_s12b_b42k")},
)

SURROGATE_MODES = ("default", "unified")
# No swappable surrogates.
NO_SURROGATES = {"anyattack", "figstep", "promptinject", "imagemix", "leadingq", "scenetap", "corruption"}
