#!/usr/bin/env python3
"""Evaluate adversarial images against a VLM (accuracy, ASR)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from realm.data import DATASETS
from realm.evaluation import evaluate
from realm.judge import DEFAULT_JUDGE_MODEL, DEFAULT_PROMPT, JUDGES


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate adversarial images against a VLM")
    p.add_argument("--attack_dirs", nargs="+", required=True, help="Attack or defense output dirs")
    p.add_argument("--dataset", choices=sorted(DATASETS), help="Dataset (its labels)")
    p.add_argument("--question", help="Question for the VLM where the manifests have none")

    tgt = p.add_mutually_exclusive_group()  # or --dataset labels
    tgt.add_argument("--target_text", help="Global target text (same for all images)")
    tgt.add_argument("--labels_file", help="Per-pair labels JSON")

    p.add_argument("--judge", choices=[*JUDGES, "none"], default=DEFAULT_PROMPT,
                   help=f"choice reads the option letter of each answer; none only collects the responses; default {DEFAULT_PROMPT}")
    p.add_argument("--model", default="qwen/qwen3-vl-8b-instruct", help="Victim VLM on --server_url")
    p.add_argument("--server_url", required=True, help="OpenAI-compatible victim server, e.g. http://localhost:8000")
    p.add_argument("--api_key", default="dummy", help="API key for --server_url")
    p.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL, help="Judge model on --server_url (the paper: Qwen3-8B)")
    p.add_argument("--output", "-o", help="Output directory for eval_results.json")
    p.add_argument("--workers", type=int, default=4, help="Concurrent VLM queries and judge calls")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    labels = args.labels_file or (DATASETS[args.dataset]["labels"] if args.dataset and not args.target_text else None)
    try:
        evaluate(args.attack_dirs, server_url=args.server_url, labels=labels, target_text=args.target_text, question=args.question,
                 model=args.model, judge=None if args.judge == "none" else args.judge, judge_model=args.judge_model,
                 api_key=args.api_key, output=args.output, workers=args.workers)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
