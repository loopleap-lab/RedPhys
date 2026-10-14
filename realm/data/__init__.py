"""Datasets."""

from pathlib import Path

from .base_dataset import Sample, BaseDataset, load_rgb
from .files import image_files, relpath, resolve, save_manifest
from .nips2017_dataset import DEFAULT_QUESTION, Nips2017Dataset

_ROOT = Path(__file__).resolve().parents[2] / "dataset"
DATASETS = {  # --dataset name -> class and folders
    "nips2017": {"class": Nips2017Dataset, "source": _ROOT / "nips2017" / "source",
                 "target": _ROOT / "nips2017" / "target", "labels": _ROOT / "nips2017" / "labels.json"},
}

__all__ = ["Sample", "BaseDataset", "load_rgb", "Nips2017Dataset", "DATASETS", "DEFAULT_QUESTION", "image_files", "relpath", "resolve", "save_manifest"]
