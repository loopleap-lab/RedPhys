from .sample import DEFAULT_QUESTION, Sample, load_rgb
from .files import (PROJECT_ROOT, file_data_url, image_bytes, image_data_url, image_files, input_stamp, list_images, manifest_samples, relpath,
                    resolve, save_manifest, write_json)

_ROOT = PROJECT_ROOT / "dataset"
DATASETS = {  # --dataset name -> folders
    "nips2017": {"source": _ROOT / "nips2017" / "source", "target": _ROOT / "nips2017" / "target", "labels": _ROOT / "nips2017" / "labels.json"},
}

__all__ = ["Sample", "load_rgb", "DATASETS", "DEFAULT_QUESTION", "PROJECT_ROOT", "file_data_url", "image_bytes", "image_data_url", "image_files",
           "input_stamp", "list_images", "manifest_samples", "relpath", "resolve", "save_manifest", "write_json"]
