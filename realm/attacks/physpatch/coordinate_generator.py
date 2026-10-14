#!/usr/bin/env python3
"""PhysPatch patch placement: SoM -> SAM -> GPT-4V."""

import os
import json
import base64
import re
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from PIL import Image
from scipy.ndimage import center_of_mass


def validate_pipeline_data(
    clean_images_dir: str,
    coords_file: Optional[str] = None,
    som_dir: Optional[str] = None,
    sam_labels_dir: Optional[str] = None,
) -> dict:
    """Status dict of which pipeline outputs exist."""
    status = {
        "coordinates_exist": False,
        "som_exist": False,
        "labels_exist": False,
        "clean_images_exist": False,
        "num_images": 0,
        "missing_components": []
    }

    clean_path = Path(clean_images_dir)
    if clean_path.exists():
        image_files = [f for f in clean_path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        status["clean_images_exist"] = len(image_files) > 0
        status["num_images"] = len(image_files)
    else:
        status["missing_components"].append("clean_images")

    if coords_file and Path(coords_file).exists():
        with open(coords_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        status["coordinates_exist"] = len(lines) > 0
    else:
        status["missing_components"].append("coordinates")

    if som_dir and Path(som_dir).exists():
        status["som_exist"] = any(f.suffix.lower() in (".jpg", ".jpeg", ".png") for f in Path(som_dir).iterdir())
    else:
        status["missing_components"].append("som_images")

    if sam_labels_dir and Path(sam_labels_dir).exists():
        label_files = list(Path(sam_labels_dir).glob("*.txt"))
        status["labels_exist"] = len(label_files) > 0
    else:
        status["missing_components"].append("sam_labels")

    return status


def generate_som_labels(
    clean_images_dir: str,
    som_output_dir: str,
    labels_output_dir: str,
    sam_checkpoint: str,
    granularity: float = 2.6,
    alpha: float = 0.1,
    label_mode: str = "Number",
    device: str = "cuda",
    names: Optional[set] = None,
) -> None:
    from segment_anything import sam_model_registry
    from realm.data import load_rgb

    import sys
    legacy_som_path = Path(__file__).parent / "assets" / "som"
    sys.path.insert(0, str(legacy_som_path))

    try:
        from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
    except ImportError:
        raise ImportError(
            f"Cannot import SoM dependencies. Please ensure the legacy SoM code is available at: {legacy_som_path}"
        )

    os.makedirs(som_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    print("Loading SAM model...")
    sam_type = "vit_b" if "vit_b" in sam_checkpoint else "vit_h"
    model_sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    model_sam.eval().to(device)
    print(f"SAM model ready ({sam_type}).")

    clean_path = Path(clean_images_dir)
    image_files = sorted(f for f in clean_path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    image_files = [f for f in image_files if names is None or f.name in names]

    label_mode_char = 'a' if label_mode == 'Alphabet' else '1'
    text_size = 900
    anno_mode = ['Mask', 'Mark']

    print(f"Processing {len(image_files)} images...")
    for img_file in image_files:
        try:
            image = load_rgb(img_file)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):  # as upstream
                output, anns = inference_sam_m2m_auto(
                    model_sam, image, text_size, label_mode_char, alpha, anno_mode
                )

            output_path = Path(som_output_dir) / img_file.name
            Image.fromarray(output).save(output_path)

            labels = []
            H, W = anns[0]["segmentation"].shape
            for ann in anns:
                cy, cx = center_of_mass(ann["segmentation"])
                x_norm = (2 * cx / W) - 1
                y_norm = (2 * cy / H) - 1
                labels.append((x_norm, y_norm))

            label_filename = img_file.stem + '.txt'
            label_path = Path(labels_output_dir) / label_filename
            with open(label_path, 'w') as f:
                for x_norm, y_norm in labels:
                    f.write(f"{x_norm:.4f}, {y_norm:.4f}\n")

            print(f"  ✓ {img_file.name} ({len(labels)} segments)")

        except Exception as e:
            print(f"  ✗ {img_file.name}: {e}")

    print(f"SoM generation complete. Output: {som_output_dir}")


GPT_PROMPT_GENERAL = """You will receive two input images:
The first image is the original image;
The second image is the semantic segmentation map (SoM image) of the original image, where each identifiable region has a unique numeric label displayed at the center of the mask.

Your task: select the single best region for placing an adversarial patch overlay.

Selection criteria (in order of priority):
1. The region should belong to the most prominent, visually dominant object in the image.
2. The region must have a clearly visible numeric label in the segmentation image.
3. Prefer large, flat, unoccluded surfaces — the patch should be clearly visible.
4. Prefer regions with a frontal or side-on view (not heavily foreshortened).
5. If one object spans multiple labeled regions, choose the region with the largest area.
6. Avoid background regions (sky, ground, distant scenery). Prefer foreground objects.

Only select the single best region.
Return format: {"label": <number>} — output only this JSON object, nothing else."""

GPT_PROMPT_DRIVING = """You will receive two input images:
The first image is the original scene;
The second image is the semantic segmentation map (SOM image) of the original scene, where each identifiable region has a unique numeric label displayed at the center of the mask.

Please analyze the images according to the following requirements and output the region ID that best meets the criteria:

- First, identify areas in the original image that are suitable for pasting or painting artistic advertisements.
- Then, locate the corresponding region in the SOM image and find its label number.
- The selected region must have a clearly visible numeric label in the segmentation image.
- It is allowed to select the surface of vehicles as the painting area, especially those parked near the edge.
- The selected region should be prominent in the image, flat, and easy to paint on.
- Prioritize objects with a frontal or side-on view.
- The object must not be occluded — the painting surface should be clearly visible and unobstructed.
- If an entity (such as a vehicle) consists of multiple labeled regions, choose the region with the largest area as the representative center label.
- The selected object should occupy a relatively large area in the current camera view, ensuring it is clearly visible and impactful for advertisement placement.
- Prefer selecting large-looking buildings, vehicles, or other structures that appear visually dominant from the current camera perspective.

Only select the single best region.
Return format: {"label": <center label>} — output only this JSON object."""


def _parse_label(raw_output: str, image_stem: str, labels_dir: str) -> Tuple[float, float, int]:
    """GPT's {"label": k} -> region k's centre."""
    match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in response: {raw_output[:100]!r}")
    label_index = json.loads(match.group(0)).get("label")
    if not isinstance(label_index, int):
        raise ValueError(f"Non-integer label: {label_index}")
    txt_path = os.path.join(labels_dir, image_stem + ".txt")
    if not os.path.exists(txt_path):
        raise ValueError("Label file not found")
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    if not 1 <= label_index <= len(lines):
        raise ValueError(f"Label {label_index} out of bounds (max {len(lines)})")
    x_str, y_str = lines[label_index - 1].strip().split(',')
    return float(x_str), float(y_str), label_index


def generate_coordinates_with_gpt(
    clean_images_dir: str,
    som_dir: str,
    labels_dir: str,
    output_coords_file: str,
    api_key: str,
    api_base: Optional[str] = None,
    model: str = "gpt-5-mini-2025-08-07",
    prompt_style: str = "general",
    names: Optional[set] = None,
) -> List[Tuple[str, float, float]]:
    """GPT-4V picks patch coordinates: (stem, x, y) tuples."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    prompt = GPT_PROMPT_DRIVING if prompt_style == "driving" else GPT_PROMPT_GENERAL

    clean_images = sorted([
        os.path.join(clean_images_dir, f)
        for f in os.listdir(clean_images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and (names is None or f in names)
    ])

    # keyed by stem: skipped entries don't misalign
    coordinates = []
    print(f"Running GPT-4V coordinate selection on {len(clean_images)} images...")

    for clean_path in clean_images:
        image_filename = os.path.basename(clean_path)
        image_stem = os.path.splitext(image_filename)[0]
        som_path = os.path.join(som_dir, image_filename)  # SoM maps use the source file name
        label_path = os.path.join(labels_dir, image_stem + ".txt")
        try:
            base64_clean, base64_som = encode_image(clean_path), encode_image(som_path)
            with open(label_path) as f:
                n_labels = sum(1 for _ in f)
        except OSError as e:
            print(f"  ✗ {image_filename}: {e}")
            continue
        for attempt in range(1, 4):
            text = prompt if attempt == 1 else f"{prompt}\nValid labels are integers from 1 to {n_labels}."
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_clean}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_som}"}}
                        ]
                    }],
                    temperature=1,
                    max_completion_tokens=4096,
                )
                x, y, label_index = _parse_label(response.choices[0].message.content or "", image_stem, labels_dir)
                coordinates.append((image_stem, x, y))
                print(f"  ✓ {image_filename} → Label {label_index} → ({x:.4f}, {y:.4f})")
                break
            except Exception as e:
                print(f"  ✗ {image_filename} (attempt {attempt}/3): {e}")

    if not coordinates:
        raise RuntimeError(f"No coordinates generated for {clean_images_dir}")
    os.makedirs(os.path.dirname(output_coords_file), exist_ok=True)
    lines = {}  # merge with the existing file
    if os.path.exists(output_coords_file):
        lines = {l.split(",")[0].strip(): l.strip() for l in open(output_coords_file) if l.count(",") == 2}
    lines.update({stem: f"{stem}, {x:.4f}, {y:.4f}" for stem, x, y in coordinates})
    with open(output_coords_file, 'w') as f:
        f.write("".join(l + "\n" for l in lines.values()))

    print(f"Saved {len(coordinates)} coordinates to {output_coords_file}")
    return coordinates


def ensure_coordinates(
    clean_images_dir: str,
    coords_file: str,
    som_dir: Optional[str] = None,
    sam_labels_dir: Optional[str] = None,
    sam_checkpoint: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    device: str = "cuda",
    prompt_style: str = "general",
    model: str = "gpt-5-mini-2025-08-07",
    names: Optional[set] = None,
) -> str:
    """Generate missing coordinates (names; default: all)."""
    # som/ and sam_labels/ are siblings of the source dir
    dataset_root = Path(clean_images_dir).parent

    if som_dir is None:
        som_dir = str(dataset_root / "som")
    if sam_labels_dir is None:
        sam_labels_dir = str(dataset_root / "sam_labels")
    if sam_checkpoint is None:
        sam_checkpoint = str(Path(__file__).parent / "assets" / "checkpoints" / "sam_vit_h_4b8939.pth")

    status = validate_pipeline_data(
        clean_images_dir=clean_images_dir,
        coords_file=coords_file,
        som_dir=som_dir,
        sam_labels_dir=sam_labels_dir
    )

    print("=" * 70)
    print("PhysPatch Coordinate Pipeline Validation")
    print("=" * 70)
    print(f"Clean images:  {'✓' if status['clean_images_exist'] else '✗'} ({status['num_images']} images)")
    print(f"Coordinates:   {'✓' if status['coordinates_exist'] else '✗'}")
    print(f"SoM images:    {'✓' if status['som_exist'] else '✗'}")
    print(f"SAM labels:    {'✓' if status['labels_exist'] else '✗'}")
    print("=" * 70)

    lines = [l for l in open(coords_file) if l.strip()] if Path(coords_file).exists() else []
    have = {l.split(",")[0].strip() for l in lines if l.count(",") == 2}
    wanted = names or {p.name for p in Path(clean_images_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    todo = {n for n in wanted if Path(n).stem not in have}
    if not todo or (lines and not have):  # all present, or a legacy positional "x, y" file
        print("✓ Coordinates exist. Using existing file.")
        return coords_file

    print(f"✗ Coordinates missing for {len(todo)} image(s). Generating...")

    if not status["clean_images_exist"]:
        raise ValueError(f"Cannot generate coordinates: No clean images found in {clean_images_dir}")

    if openai_api_key is None:
        raise ValueError(
            "Cannot generate coordinates: OpenAI API key required. "
            "Please provide via --openai_api_key or set OPENAI_API_KEY environment variable."
        )

    no_som = {n for n in todo if not (Path(som_dir) / n).exists() or not (Path(sam_labels_dir) / f"{Path(n).stem}.txt").exists()}
    if no_som:
        print("\n[1/2] Generating SoM annotations and labels...")

        if not Path(sam_checkpoint).exists():
            raise ValueError(f"SAM checkpoint not found: {sam_checkpoint}")

        generate_som_labels(
            clean_images_dir=clean_images_dir,
            som_output_dir=som_dir,
            labels_output_dir=sam_labels_dir,
            sam_checkpoint=sam_checkpoint,
            device=device,
            names=no_som,
        )
        print("✓ SoM generation complete.\n")

    print("[2/2] Running GPT-4V coordinate selection...")
    generate_coordinates_with_gpt(
        clean_images_dir=clean_images_dir,
        som_dir=som_dir,
        labels_dir=sam_labels_dir,
        output_coords_file=coords_file,
        api_key=openai_api_key,
        api_base=openai_api_base,
        model=model,
        prompt_style=prompt_style,
        names=todo,
    )
    print("✓ Coordinate generation complete.\n")

    return coords_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PhysPatch coordinate generation pipeline")
    parser.add_argument('--clean_images', required=True, help='Clean images directory')
    parser.add_argument('--coords_file', required=True, help='Output coordinates file')
    parser.add_argument('--som_dir', help='SoM images directory (auto-inferred if not provided)')
    parser.add_argument('--labels_dir', help='SAM labels directory (auto-inferred if not provided)')
    parser.add_argument('--sam_checkpoint', help='SAM checkpoint path (auto-inferred if not provided)')
    parser.add_argument('--openai_api_key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--openai_api_base', help='Optional custom OpenAI API base URL')
    parser.add_argument('--prompt_style', default='general', choices=['general', 'driving'],
                        help='GPT prompt style: "general" for arbitrary images, "driving" for autonomous driving scenes')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')

    args = parser.parse_args()

    api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')

    ensure_coordinates(
        clean_images_dir=args.clean_images,
        coords_file=args.coords_file,
        som_dir=args.som_dir,
        sam_labels_dir=args.labels_dir,
        sam_checkpoint=args.sam_checkpoint,
        openai_api_key=api_key,
        openai_api_base=args.openai_api_base,
        device=args.device,
        prompt_style=args.prompt_style,
    )
