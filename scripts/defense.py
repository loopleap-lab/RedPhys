#!/usr/bin/env python3
"""Clean adversarial images with a defense."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from realm.defense import DefenseRegistry
from realm.defense.run import run


def parse_args():
    p = argparse.ArgumentParser(description="Clean adversarial images")
    p.add_argument("defense", choices=DefenseRegistry.names(), help="Defense name")
    p.add_argument("--input", required=True, nargs="+", help="Attack output dir, image dir or image (adpo: one or more attack dirs)")
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--max_samples", type=int, help="Max images to process")
    p.add_argument("--question", help="Question for samples whose manifest has none (text-side defenses)")
    p.add_argument("--config", help="Defense hyperparameters YAML (default: configs/defenses/<defense>.yaml)")
    p.set_defaults(params={})
    a = p.parse_args()
    a.input = a.input[0] if len(a.input) == 1 else a.input
    return a


if __name__ == "__main__":
    try:
        run(parse_args())
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
