"""Summary table and eval_results.json content."""

import time

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
        return [{"id": r["id"], "image": str(r["image"]), "question": r["question"], "response": r["response"],
                 "target": target_of(r["id"], labels, target_text), "source": label_of(r["id"], labels, "source"),
                 **({"options": r["options"]} if r.get("options") else {}), "verdict": verdicts.get(name, {}).get(r["id"]),
                 "status": r["status"]} for r in responses]

    return {
        "model": model, "question": question, "target_text": target_text, "per_pair_labels": bool(labels),
        "judge": {"prompt": judge, "model": judge_model if judge != "substring" else None, "threshold": JUDGE_THRESHOLD}, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clean_baseline": {"metrics": metrics.get("clean", {}), "responses": _detail("clean", clean)},
        "attacks": {name: {"metrics": metrics.get(name, {}), "responses": _detail(name, rs)} for name, rs in attacks.items()},
    }
