"""PhysPatch attack configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent


def resolve_cli_kwargs(attack_name, args, context):
    return {
        "target_images_dir": context["target_dir"],
        "source_images_dir": context["source_dir"],
        "source_images": context["source_images"],
        "coords_file": _find_coords_file(context["source_dir"]),
    }


def _find_coords_file(source_dir: str) -> str:
    dataset_root = Path(source_dir).parent
    coords_dir = dataset_root / "coordinates"
    if coords_dir.is_dir():
        txts = sorted(coords_dir.glob("*.txt"))
        if txts:
            return str(txts[0])
    return str(dataset_root / "coordinates" / "full.txt")
