"""Chain of Attack (CoA) configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

def resolve_cli_kwargs(attack_name, args, context):
    return {"clean_images_dir": context["source_dir"], "target_images_dir": context["target_dir"]}
