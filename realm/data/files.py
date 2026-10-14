"""Paths, image listing and the manifest.json write shared by the scripts and evaluation."""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def relpath(path) -> str:
    """Relative to the project root; absolute for a path outside it."""
    p = Path(path).resolve()
    return str(p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p)


def resolve(path) -> Path:
    """A manifest path as absolute: relative ones are from the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def image_files(directory) -> list[Path]:
    return sorted(f for f in Path(directory).iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)


def save_manifest(out_dir: Path, attack: str, results: list, duration: float, **extra) -> None:
    """Atomic manifest.json write; extra fields come right after `attack`."""
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps({
        "attack": attack,
        **extra,
        "n_samples": len(results),
        "n_errors": sum("error" in r for r in results),
        "duration_s": round(duration, 1),
        "samples": results,
    }, indent=2))
    os.replace(tmp, out_dir / "manifest.json")
