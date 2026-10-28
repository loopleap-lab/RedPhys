"""PhysPatch patch placement: SAM masks -> SoM marks -> a vision LLM picks the region."""

import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from PIL import Image
from scipy.ndimage import center_of_mass

from ...data import file_data_url, image_files, load_rgb


def generate_som_labels(
    clean_images_dir: str,
    som_output_dir: str,
    labels_output_dir: str,
    sam_checkpoint: str,
    alpha: float = 0.1,  # mask opacity
    device: str = "cuda",
    names: Optional[set] = None,
) -> None:
    from segment_anything import sam_model_registry

    import sys
    som_path = Path(__file__).parent / "assets" / "som"  # the official SoM code
    sys.path.insert(0, str(som_path))

    try:
        from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
    except ImportError:
        raise ImportError(
            f"Cannot import SoM dependencies. Please ensure the SoM code is available at: {som_path}"
        )

    os.makedirs(som_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    print("Loading SAM model...")
    sam_type = "vit_b" if "vit_b" in sam_checkpoint else "vit_h"
    model_sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    model_sam.eval().to(device)
    print(f"SAM model ready ({sam_type}).")

    images = [f for f in image_files(clean_images_dir) if names is None or f.name in names]

    text_size = 900
    anno_mode = ['Mask', 'Mark']

    print(f"Processing {len(images)} images...")
    for img_file in images:
        try:
            image = load_rgb(img_file)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):  # as upstream
                output, anns = inference_sam_m2m_auto(
                    model_sam, image, text_size, "1", alpha, anno_mode  # numeric labels
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


GPT_PROMPT = """You will receive two input images:
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
Return format: {"label": <center label>} — output only this JSON object."""  # upstream's (som_gpt.py)

PROMPTS = {"general": GPT_PROMPT, "driving": GPT_PROMPT_DRIVING}


def _parse_label(raw_output: str, centres: List[str]) -> Tuple[float, float, int]:
    """The LLM's {"label": k} -> region k's centre."""
    match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in response: {raw_output[:100]!r}")
    label_index = json.loads(match.group(0)).get("label")
    if not isinstance(label_index, int):
        raise ValueError(f"Non-integer label: {label_index}")
    if not 1 <= label_index <= len(centres):
        raise ValueError(f"Label {label_index} out of bounds (max {len(centres)})")
    x_str, y_str = centres[label_index - 1].strip().split(',')
    return float(x_str), float(y_str), label_index


def generate_coordinates_with_gpt(
    clean_images_dir: str,
    som_dir: str,
    labels_dir: str,
    output_coords_file: str,
    client,
    model: str = "gpt-5-mini-2025-08-07",
    prompt_style: str = "general",  # or "driving"
    names: Optional[set] = None,
) -> List[Tuple[str, float, float]]:
    """A vision LLM (an OpenAI client) picks patch coordinates: (stem, x, y) tuples."""
    prompt = PROMPTS[prompt_style]
    clean_images = [str(p) for p in image_files(clean_images_dir) if names is None or p.name in names]

    # keyed by stem: skipped entries don't misalign
    coordinates = []
    print(f"Running LLM coordinate selection on {len(clean_images)} images...")

    for clean_path in clean_images:
        image_filename = os.path.basename(clean_path)
        image_stem = os.path.splitext(image_filename)[0]
        som_path = os.path.join(som_dir, image_filename)  # SoM maps use the source file name
        label_path = os.path.join(labels_dir, image_stem + ".txt")
        try:
            url_clean, url_som = file_data_url(clean_path), file_data_url(som_path)
            with open(label_path) as f:
                centres = f.read().splitlines()
            n_labels = len(centres)
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
                            {"type": "image_url", "image_url": {"url": url_clean}},
                            {"type": "image_url", "image_url": {"url": url_som}}
                        ]
                    }],
                    temperature=1,
                    max_completion_tokens=4096,
                )
                x, y, label_index = _parse_label(response.choices[0].message.content or "", centres)
                coordinates.append((image_stem, x, y))
                print(f"  ✓ {image_filename} → Label {label_index} → ({x:.4f}, {y:.4f})")
                break
            except Exception as e:
                print(f"  ✗ {image_filename} (attempt {attempt}/3): {e}")

    if not coordinates:
        raise RuntimeError(f"No coordinates generated for {clean_images_dir}")
    os.makedirs(os.path.dirname(output_coords_file) or ".", exist_ok=True)
    lines = {}  # merge with the existing file
    if os.path.exists(output_coords_file):
        lines = {l.split(",")[0].strip(): l.strip() for l in open(output_coords_file) if l.count(",") == 2}
    lines.update({stem: f"{stem}, {x:.4f}, {y:.4f}" for stem, x, y in coordinates})
    with open(output_coords_file, 'w') as f:
        f.write("".join(l + "\n" for l in lines.values()))

    print(f"Saved {len(coordinates)} coordinates to {output_coords_file}")
    return coordinates


def dataset_root(images_dir) -> Path:
    """Where coordinates/, som/ and sam_labels/ go: the dataset dir for a source/ or clean/ subdir, else the images dir itself."""
    p = Path(images_dir)
    return p.parent if p.name in ("source", "clean") else p


def ensure_coordinates(
    clean_images_dir: str,
    coords_file: str,
    llm=None,
    som_dir: Optional[str] = None,
    sam_labels_dir: Optional[str] = None,
    sam_checkpoint: Optional[str] = None,
    device: str = "cuda",
    model: str = "gpt-5-mini-2025-08-07",
    prompt_style: str = "general",
    names: Optional[set] = None,
) -> str:
    """Generate missing coordinates (names; default: all); llm() gives the OpenAI client."""
    root = dataset_root(clean_images_dir)
    som_dir, sam_labels_dir = som_dir or str(root / "som"), sam_labels_dir or str(root / "sam_labels")
    sam_checkpoint = sam_checkpoint or str(Path(__file__).parent / "assets" / "checkpoints" / "sam_vit_h_4b8939.pth")

    lines = [l for l in open(coords_file) if l.strip()] if Path(coords_file).exists() else []
    have = {l.split(",")[0].strip() for l in lines if l.count(",") == 2}
    wanted = names or {p.name for p in image_files(clean_images_dir)}
    todo = {n for n in wanted if Path(n).stem not in have}
    if not todo or (lines and not have):  # all present, or a legacy positional "x, y" file
        print("✓ Coordinates exist. Using existing file.")
        return coords_file

    print(f"✗ Coordinates missing for {len(todo)} image(s). Generating...")

    if llm is None:
        raise ValueError("Cannot generate coordinates without an LLM client")
    client = llm()  # raises without a key or server

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

    print("[2/2] Running LLM coordinate selection...")
    generate_coordinates_with_gpt(
        clean_images_dir=clean_images_dir,
        som_dir=som_dir,
        labels_dir=sam_labels_dir,
        output_coords_file=coords_file,
        client=client,
        model=model,
        prompt_style=prompt_style,
        names=todo,
    )
    print("✓ Coordinate generation complete.\n")

    return coords_file

