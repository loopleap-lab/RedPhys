"""The printed summary table and the eval_results.json content of an evaluation."""

import time

from ..judge import JUDGE_THRESHOLD
from .metrics import VICTIM, get_source, get_target


def _pct(rate: float, n: int, w: int) -> str:
    return f"{rate*100:{w-1}.1f}%" if n else f"{'n/a':>{w}}"


def print_summary(clean_rate: dict, asrs: dict[str, dict], mrs: dict[str, dict], target_text: str | None, judge: str, labels: dict | None):
    w = 12
    has_mr = bool(mrs)
    nw = max([20, *map(len, asrs)])
    total_w = nw + (5 if has_mr else 4) * (w + 1)
    print(f"\n{'=' * total_w}")
    print(f"Target: {'per-pair labels' if labels else repr(target_text)}  |  Judge: {judge}")
    print(f"{'=' * total_w}")
    hdr = f"{'Source':<{nw}} {'ASR':>{w}} {'Match':>{w}} {'Total':>{w}} {'AvgScore':>{w}}"
    if has_mr:
        hdr += f" {'Accuracy':>{w}}"
    print(hdr)
    print(f"{'-' * total_w}")
    line = (f"{'clean (baseline)':<{nw}} {_pct(clean_rate['rate'], clean_rate['n_total'], w)} "
            f"{clean_rate['n_match']:{w}} {clean_rate['n_total']:{w}}")
    if has_mr and "mr" in clean_rate:
        line += f" {'':>{w}} {(1 - clean_rate['mr'])*100:{w-1}.1f}%"
    print(line)
    print(f"{'-' * total_w}")
    for name, a in sorted(asrs.items()):
        line = (f"{name:<{nw}} {_pct(a['asr'], a['n_total'], w)} "
                f"{a['n_match']:{w}} {a['n_total']:{w}} {a.get('avg_score', a['asr']):{w-1}.3f}")
        if has_mr:
            mr = mrs.get(name)
            line += f" {(1 - mr['mr'])*100:{w-1}.1f}%" if mr else f" {'n/a':>{w}}"
        print(line)
    print(f"{'=' * total_w}\n")


def results_dict(model: str, question: str, target_text: str | None, clean_responses: list[dict], attack_data: dict[str, list[dict]],
                 clean_rate: dict, asrs: dict[str, dict], mrs: dict[str, dict], labels: dict | None, judge: str, judge_model: str) -> dict:
    """What eval_results.json holds: settings, then the metrics and every judged response of the clean baseline and of each attack."""

    def _detail(responses, metrics):
        dm = {d["id"]: d for d in metrics.get("details", [])}
        sm = {d["id"]: d for d in metrics.get("mr_details", [])}
        return [{
            "id": r["id"], "image": str(r["image"]), "question": r.get("question", question), "response": r["response"],
            "target": get_target(r["id"], labels, target_text),
            "match": dm.get(r["id"], {}).get("match"),  # None: not judged
            "score": dm.get(r["id"], {}).get("score"),
            "source": get_source(r["id"], labels),
            "source_match": sm.get(r["id"], {}).get("source_match"),
            "status": r["status"],
        } for r in responses]

    def _metrics(m):
        return {k: v for k, v in m.items() if k not in ("details", "mr_details")}

    metrics = {name: {**asrs[name], **mrs.get(name, {})} for name in asrs}
    return {
        "model": model, "question": question, "target_text": target_text,
        "per_pair_labels": bool(labels), "victim": VICTIM, "judge": {"prompt": judge, "model": judge_model, "threshold": JUDGE_THRESHOLD},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clean_baseline": {"metrics": _metrics(clean_rate), "responses": _detail(clean_responses, clean_rate)},
        "attacks": {name: {"metrics": _metrics(metrics[name]), "responses": _detail(attack_data[name], metrics[name])} for name in asrs},
    }
