"""AdvDiffVLM attack configuration."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


def resolve_cli_kwargs(attack_name, args, context):
    return {
        "target_images_dir": context["target_dir"],
        "gradcam_masks_dir": str(Path(context["output_dir"]) / "_gradcam_masks"),
    }


def get_checkpoint_path():
    checkpoint = BASE_DIR / "assets" / "checkpoints" / "ldm_cin256-v2.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return str(checkpoint)


def get_config_yaml_path():
    config_yaml = BASE_DIR / "assets" / "configs" / "latent-diffusion_cin256-v2.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(f"Config not found: {config_yaml}")
    return str(config_yaml)


def get_clip_cache_dir():
    """clip.load's default dir, shared with the other attacks' CLIP weights."""
    return os.path.expanduser("~/.cache/clip")
