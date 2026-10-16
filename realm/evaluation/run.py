"""realm.evaluate and scripts/evaluate_adversarial.py."""

import json
import time
from pathlib import Path

from openai import OpenAI

from ..judge import DEFAULT_PROMPT, no_reasoning
from .metrics import compute_asr, compute_clean_rate, compute_mr, load_attack_dir, load_labels, run_queries, victim_item
from .report import print_summary, results_dict


def _errors(responses: list[dict], *metrics: dict | None) -> str:
    nv = sum(r["status"] != "ok" for r in responses)
    nj = sum(m.get("n_judge_errors", 0) + m.get("mr_judge_errors", 0) for m in metrics if m)
    return f"  ({nv} failed queries and {nj} failed judge calls left out)" if nv or nj else ""


def evaluate(attack_dirs: str | list[str], *, server_url: str, targets: str | None = None, labels: str | None = None,
             target_text: str | None = None, question: str | None = None, model: str = "qwen/qwen3-vl-8b-instruct", judge: str = DEFAULT_PROMPT,
             judge_model: str = "openai/gpt-4o", api_key: str = "dummy", output: str | None = None, workers: int = 4) -> dict:
    """Victim accuracy and ASR on each realm.attack / realm.defend dir vs. its clean images.

    Target: `targets` (realm.target dir), `labels` (its labels.json) or `target_text`; judge: a prompt name or "substring".
    Returns the eval_results.json content (saved under `output` if given).
    """
    attack_dirs = [attack_dirs] if isinstance(attack_dirs, str) else attack_dirs
    labels = load_labels(labels or f"{targets}/labels.json") if labels or targets else None
    attacks = {}
    for d in attack_dirs:
        try:
            data = load_attack_dir(d)
        except Exception as e:
            print(f"WARNING: skipping {d}: {e}")
            continue
        name = data.get("attack", Path(d).name)
        name = f"{name}:{d}" if name in attacks else name
        attacks[name] = data["valid_results"]
        print(f"Loaded:   {name} -- {len(attacks[name])} samples")
    if not attacks:
        raise ValueError("no valid attack directories")
    labels = labels or {r["id"]: r["label"] for res in attacks.values() for r in res if "label" in r} or None
    if labels is None and not target_text:
        raise ValueError("a target is required: targets=, labels= or target_text=")

    manifest_question = next((r["question"] for results in attacks.values() for r in results if r.get("question")), None)
    if not (question or manifest_question):
        raise ValueError("no question: pass question= (the manifests have none)")
    question = question or manifest_question

    print(f"Server:   {server_url}  model={model}")
    print(f"Question: {question!r}")
    print(f"Judge:    {judge}" + (f" ({judge_model})" if judge != "substring" else ""))
    print(f"Targets:  per-case ({len(labels)} cases)" if labels else f"Target:   {target_text!r}")

    client = OpenAI(base_url=f"{server_url}/v1", api_key=api_key)
    for m in {model, judge_model} if judge != "substring" else {model}:  # fail fast
        try:
            client.chat.completions.create(model=m, messages=[{"role": "user", "content": "Hi"}], max_tokens=16, extra_body=no_reasoning(client))
        except Exception as e:
            raise ValueError(f"{m} on {server_url}: {e} (no OpenAI-compatible server? start one, e.g. `vllm serve <model>`)") from e

    clean = {}
    for results in attacks.values():
        for r in results:
            clean.setdefault(r["id"], {"id": r["id"], "image": r["source"], **({"question": r["question"]} if r.get("question") else {})})
    print(f"\nPhase 1: Clean baseline ({len(clean)} images)")
    t0 = time.time()
    clean_responses = run_queries(client, model, [clean[k] for k in sorted(clean)], question, workers, "clean")
    clean_rate = compute_clean_rate(clean_responses, target_text, labels, judge, client, judge_model, workers)
    clean_rate.update(compute_mr(clean_responses, labels, judge, client, judge_model, workers) or {})
    mr_str = f"  accuracy={(1 - clean_rate['mr'])*100:.1f}%" if "mr" in clean_rate else ""
    print(f"  Baseline: {clean_rate['rate']*100:.1f}% ({clean_rate['n_match']}/{clean_rate['n_total']}){mr_str} "
          f"[{time.time()-t0:.1f}s]{_errors(clean_responses, clean_rate)}")

    print(f"\nPhase 2: Attack evaluation ({judge} judge)")
    attack_responses, attack_asrs, attack_mrs = {}, {}, {}
    for name, results in attacks.items():
        t0 = time.time()
        responses = attack_responses[name] = run_queries(client, model, [victim_item(r, question) for r in results], question, workers, name)
        tq = time.time() - t0
        t0 = time.time()
        asr = attack_asrs[name] = compute_asr(responses, target_text, labels, judge, client, judge_model, workers)
        mr = compute_mr(responses, labels, judge, client, judge_model, workers)
        if mr:
            attack_mrs[name] = mr
        timing = f"query={tq:.1f}s" + (f" judge={time.time() - t0:.1f}s" if judge != "substring" else "")
        mr_str = f"  accuracy={(1 - mr['mr'])*100:.1f}%" if mr else ""
        print(f"  {name}: ASR={asr['asr']*100:.1f}% ({asr['n_match']}/{asr['n_total']}){mr_str} [{timing}]{_errors(responses, asr, mr)}")

    print_summary(clean_rate, attack_asrs, attack_mrs, target_text, judge, labels)
    result = results_dict(model, question, target_text, clean_responses, attack_responses, clean_rate, attack_asrs, attack_mrs, labels, judge, judge_model)
    if output:
        path = Path(output) / "eval_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved {path}")
    return result
