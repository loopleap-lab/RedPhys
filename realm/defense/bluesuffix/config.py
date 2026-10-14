"""BlueSuffix asset paths."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

def get_diffusion_checkpoint_path():
    checkpoint = BASE_DIR / "assets" / "256x256_diffusion_uncond.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"BlueSuffix diffusion checkpoint not found at {checkpoint}")
    return str(checkpoint)


def get_suffix_generator_path():
    suffix_dir = BASE_DIR / "assets" / "suffix_generator"
    if not suffix_dir.exists():
        raise FileNotFoundError(f"Suffix generator not found at {suffix_dir}")
    return str(suffix_dir)
