#!/usr/bin/env python3
"""Generate adversarial images with a registered attack."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from realm.attacks.run import ALL_ATTACKS, run
from realm.attacks.surrogates import SURROGATE_MODES
from realm.data import DATASETS


def parse_args():
    p = argparse.ArgumentParser(description="Generate adversarial images")
    p.add_argument("attack", choices=ALL_ATTACKS, help="Attack name")
    p.add_argument("--dataset", choices=sorted(DATASETS), help="Dataset (auto-sets --source/--target)")
    p.add_argument("--source", help="Source image file or directory")
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--max_samples", type=int, help="Max images to process")
    p.add_argument("--surrogates", choices=SURROGATE_MODES, default="default",
                   help="default: each attack's original surrogates; unified: shared CLIP ensemble (surrogates.py)")
    p.add_argument("--config", help="Attack hyperparameters YAML (default: configs/attacks/<attack>.yaml)")
    p.add_argument("--target", help="Target image file or directory")
    p.add_argument("--target_text", help="Target text (advedm_r, vattack, figstep, promptinject)")
    p.add_argument("--question", help="Question to inject into (promptinject)")
    p.add_argument("--labels_file", help="Per-sample labels JSON (auto-detected from --dataset)")
    p.set_defaults(params={})  # overrides: Python API only
    return p.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
