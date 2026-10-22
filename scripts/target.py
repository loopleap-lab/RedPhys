#!/usr/bin/env python3
"""Scenario-specific attack targets (target answer + target image) with the scenario agent."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from realm.agent.scenario_agent.run import run
from realm.data import DATASETS, DEFAULT_QUESTION


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cases", help="JSON list of cases: id, image, question, answer, [options, domain, task, target]")
    src.add_argument("--dataset", choices=sorted(DATASETS), help="Open-ended cases from a dataset's images and labels")
    src.add_argument("--source", help="Image dir; answers from --labels_file ({stem: {source}})")
    p.add_argument("--labels_file", help="Labels JSON for --source")
    p.add_argument("--question", default=DEFAULT_QUESTION, help="Question for --dataset/--source cases")
    p.add_argument("--keep_targets", action="store_true", help="Use the labels' target texts instead of choosing new ones")
    p.add_argument("--output", "-o", required=True, help="Output dir: target/<id>.png, labels.json, targets.json")
    p.add_argument("--max_samples", type=int, help="Max cases")
    p.add_argument("--config", help="Agent YAML (default: configs/agents/scenario.yaml)")
    p.add_argument("--workers", type=int, default=8, help="Cases in parallel (API-bound)")
    p.add_argument("--params", type=json.loads, default={}, help='Agent settings as JSON, e.g. {"max_rounds": 3}')
    return p.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
