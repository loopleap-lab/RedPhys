"""Summary table, eval_results.json content and the cross-run report."""

import csv
import json
import time
from pathlib import Path

from ..judge import JUDGE_THRESHOLD
from .metrics import label_of, target_of


def print_summary(metrics: dict[str, dict], target_text: str | None, labels: dict | None, judge: str):
    nw, w = max([20, *map(len, metrics)]), 10
    rule = "=" * (nw + 4 * (w + 1))
    print(f"\n{rule}\nTarget: {repr(target_text) if target_text else 'per-pair labels'}  |  Judge: {judge}\n{rule}")
    print(f"{'Source':<{nw}} {'ASR':>{w}} {'Match':>{w}} {'Total':>{w}} {'Accuracy':>{w}}")
    pct = lambda v: f"{100 * v:{w - 1}.1f}%" if v is not None else f"{'n/a':>{w}}"
    for name, m in metrics.items():
        levels = "  " + " ".join(f"@{t}:{100 * v:.1f}%" for t, v in m["asr_at"].items()) + f" avg {m['avg_score']:.3f}" if "asr_at" in m else ""
        print(f"{name:<{nw}} {pct(m['asr'])} {m['n_match']:{w}} {m['n_total']:{w}} {pct(m['accuracy'])}{levels}")
    print(f"{rule}\n")


def results_dict(model: str, question: str, target_text: str | None, clean: list[dict], attacks: dict[str, list[dict]],
                 verdicts: dict[str, dict], metrics: dict[str, dict], labels: dict | None, judge: str | None, judge_model: str | None) -> dict:
    """Settings, metrics and every response with its verdict."""

    def _detail(name, responses):
        return [{"id": r["id"], "image": str(r["image"]), "mtime": r.get("mtime"), "question": r["question"], "response": r["response"],
                 "target": target_of(r["id"], labels, target_text), "source": label_of(r["id"], labels, "source"),
                 **({"options": r["options"]} if r.get("options") else {}), "verdict": verdicts.get(name, {}).get(r["id"]),
                 "status": r["status"]} for r in responses]

    return {
        "model": model, "question": question, "target_text": target_text, "per_pair_labels": bool(labels),
        "judge": {"prompt": judge, "model": judge_model if judge != "substring" else None, "threshold": JUDGE_THRESHOLD}, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clean_baseline": {"metrics": metrics.get("clean", {}), "responses": _detail("clean", clean)},
        "attacks": {name: {"metrics": metrics.get(name, {}), "responses": _detail(name, rs)} for name, rs in attacks.items()},
    }


def report(results: str | list[str], output: str | None = None) -> str:
    """Markdown table of evaluate/score runs: a row per run (its dir, model) and defense ("attack+defense" results), clean accuracy
    and ASR / accuracy (%) per attack; output: .md or .csv."""
    rows = {}
    for p in [results] if isinstance(results, str) else results:
        f = Path(p) / "eval_results.json" if Path(p).is_dir() else Path(p)
        r = json.loads(f.read_text())
        rows.setdefault((f.parent.name, "-", r["model"], r["clean_baseline"]["metrics"].get("accuracy")), {})
        for name, v in r["attacks"].items():
            a, _, d = name.partition("+")
            rows.setdefault((f.parent.name, d or "-", r["model"], r["clean_baseline"]["metrics"].get("accuracy")), {})[a] = v["metrics"]
    attacks = list(dict.fromkeys(a for m in rows.values() for a in m))
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}"
    cells = [(run, d, model, pct(acc), [(pct(m.get(a, {}).get("asr")), pct(m.get(a, {}).get("accuracy"))) for a in attacks])
             for (run, d, model, acc), m in rows.items()]
    if output and output.endswith(".csv"):
        with open(output, "w", newline="") as f:
            csv.writer(f).writerows([["run", "defense", "model", "clean_accuracy", *(f"{a}_{k}" for a in attacks for k in ("asr", "accuracy"))],
                                     *([run, d, model, acc, *(v for pair in per for v in pair)] for run, d, model, acc, per in cells)])
    lines = [["run", "defense", "model", "clean acc", *(f"{a} ASR / acc" for a in attacks)], ["---"] * (len(attacks) + 4),
             *([run, d, model, acc, *(f"{s} / {c}" for s, c in per)] for run, d, model, acc, per in cells)]
    table = "\n".join("| " + " | ".join(row) + " |" for row in lines)
    if output and not output.endswith(".csv"):
        Path(output).write_text(table + "\n")
    return table
