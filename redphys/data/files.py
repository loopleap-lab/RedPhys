"""Paths, image listing and encoding, JSON and manifest.json writes."""

import base64
import io
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


def image_bytes(img, fmt: str = "PNG") -> bytes:
    """A PIL image encoded (PNG by default)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def image_data_url(img, fmt: str = "PNG") -> str:
    """A PIL image as a base64 data URL."""
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(image_bytes(img, fmt)).decode()}"


def file_data_url(path) -> str:
    """An image file's bytes as a base64 data URL (type from the suffix)."""
    kind = Path(path).suffix.lower().lstrip(".").replace("jpg", "jpeg")
    return f"data:image/{kind};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def image_files(directory) -> list[Path]:
    return sorted(f for f in Path(directory).iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)


def list_images(source, max_samples: int | None = None) -> list[Path]:
    """An image file, or a dir's images (first max_samples); outputs are named <stem>.png, so stems must be unique."""
    p = Path(source)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise ValueError(f"Source not found: {source}")
    files = image_files(p)
    if not files:
        raise ValueError(f"No images found in {source}")
    if len({f.stem for f in files}) < len(files):
        raise ValueError(f"Duplicate file stems in {source} (e.g. a.jpg and a.png): outputs are named <stem>.png")
    return files[:max_samples] if max_samples else files


def released_entry(r: dict, d: Path, attack: str) -> dict:
    """A released-dataset manifest sample (output_path, behavior_id, ...) in redphys.attack form; d: its dir."""
    return {"id": r["behavior_id"], "output": relpath(d / r["output_path"]), "source": relpath(d.parents[1] / "clean" / r["source_path"]),
            **({"question": r["question"]} if r.get("question") else {}), "label": {"source": r["correct_answer"], "target": r["attack_target"]},
            **({"metadata": {"attack": attack, **r["attack_metadata"]}} if "attack_metadata" in r else {})}


def manifest_samples(d: Path, data: dict) -> list[dict]:
    """A manifest's samples without errors, released ones in redphys.attack form."""
    return [released_entry(s, d, data["attack"]) if "output_path" in s else s for s in data.get("samples", []) if "error" not in s]


def input_stamp(path: Path) -> list:
    """[path, mtime_ns] of an input: resume redoes it when it changed."""
    return [relpath(path), path.stat().st_mtime_ns]


def write_json(path, obj, **kw) -> None:
    """Atomic JSON write: a crash never leaves half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, **kw))
    os.replace(tmp, path)


def save_manifest(out_dir: Path, name: str, results: list, duration: float, **extra) -> None:
    """manifest.json; name is the run's attack (defended: "<attack>+<defense>"), extra fields come right after it."""
    write_json(out_dir / "manifest.json", {
        "attack": name,
        **extra,
        "n_samples": len(results),
        "n_errors": sum("error" in r for r in results),
        "duration_s": round(duration, 1),
        "samples": results,
    })
