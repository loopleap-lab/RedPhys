"""PAD asset paths."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

SAM_CHECKPOINTS = {"vit_b": "sam_vit_b_01ec64.pth", "vit_l": "sam_vit_l_0b3195.pth", "vit_h": "sam_vit_h_4b8939.pth"}


def get_sam_checkpoint_path(model_type: str = "vit_l"):
    """Official SAM checkpoint for model_type under PAD assets."""
    sam_checkpoint = BASE_DIR / "assets" / "models" / SAM_CHECKPOINTS[model_type]

    if not sam_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {sam_checkpoint}. "
            f"Place the checkpoint under pad/assets/models or set sam_checkpoint in the defense YAML."
        )

    return str(sam_checkpoint)
