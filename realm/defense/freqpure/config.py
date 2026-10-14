from pathlib import Path

BASE_DIR = Path(__file__).parent

def get_diffusion_checkpoint_path():
    checkpoint = BASE_DIR / "assets" / "pretrained" / "guided_diffusion" / "256x256_diffusion_uncond.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"FreqPure checkpoint not found at {checkpoint}")
    return str(checkpoint)
