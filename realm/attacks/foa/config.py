from pathlib import Path

BASE_DIR = Path(__file__).parent


def resolve_cli_kwargs(attack_name, args, context):
    return {"target_images_dir": context["target_dir"]}
