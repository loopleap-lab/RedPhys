"""redphys.evaluate and scripts/evaluate.py."""

import json
import time
from pathlib import Path

from openai import OpenAI

from ..data import image_files, write_json

from ..judge import DEFAULT_PROMPT, JUDGES, judge_model_for, no_reasoning, options
from .metrics import VICTIM, judge_all, label_of, letter_of, load_attack_dir, rates, run_queries, target_of, victim_item, victim_prompt, with_options
from .report import print_summary, results_dict


def _errors(responses: list[dict]) -> str:
    n = sum(r["status"] != "ok" for r in responses)
    return f"  ({n} failed queries left out)" if n else ""


def _client(server_url: str, api_key: str, models: set[str]) -> OpenAI:
    client = OpenAI(base_url=f"{server_url}/v1", api_key=api_key)
    for m in models:  # fail fast
        try:
            client.chat.completions.create(model=m, messages=[{"role": "user", "content": "Hi"}], max_tokens=16, extra_body=no_reasoning(client, m))
        except Exception as e:
            raise ValueError(f"{m} on {server_url}: {e} (no OpenAI-compatible server? start one, e.g. `vllm serve <model>`)") from e
    return client


def _check_judge(judge: str | None, rows) -> None:
    """Fail before any query: an unknown judge, or `choice` on questions without options or labels that are not options.

    rows: (id, options, target, source) per sample.
    """
    if judge not in (None, *JUDGES):
        raise ValueError(f"judge must be one of {JUDGES} or None, got {judge!r}")
    if judge != "choice":
        return
    if not all(opts for _, opts, _, _ in rows):
        raise ValueError("judge='choice' reads option letters: some questions have no A./B. options; use judge='main_object'")
    if bad := [i for i, opts, t, s in rows if letter_of(opts, t) is None or (s and letter_of(opts, s) is None)]:
        raise ValueError(f"judge='choice': {len(bad)} samples (e.g. {bad[0]}) have a target or source label that is none of their options")


def _score(model: str, question: str, target_text: str | None, labels: dict | None, clean: list[dict],
           attacks: dict[str, list[dict]], judge: str | None, client: OpenAI | None, judge_model: str, workers: int) -> dict:
    """Judged results (judge=None: unjudged)."""
    verdicts, metrics = {}, {}
    if judge:
        print(f"\nJudging ({judge}" + (f", {judge_model})" if judge != "substring" else ")"))
        for name, responses in [("clean", clean), *attacks.items()]:
            verdicts[name] = judge_all(responses, judge, client, judge_model, workers, labels, target_text)
            metrics[name] = rates(responses, verdicts[name], judge)
        print_summary(metrics, target_text, labels, judge)
    return results_dict(model, question, target_text, clean, attacks, verdicts, metrics, labels, judge, judge_model)


def _save(result: dict, output: str | None, quiet: bool = False) -> dict:
    if output:
        path = Path(output) / "eval_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, result, ensure_ascii=False)
        if not quiet:
            print(f"Saved {path}")
    return result


def _prior(output: str | None, model: str) -> dict:
    """Saved ok responses with the same model and victim settings, by (set, id)."""
    f = Path(output or "") / "eval_results.json"
    if not output or not f.is_file() or not f.stat().st_size:
        return {}
    data = json.loads(f.read_text())
    if data["model"] != model or {k: v for k, v in data.get("victim", {}).items() if k != "max_image_mb"} != VICTIM:
        return {}
    sets = {"clean": data["clean_baseline"]["responses"], "target": data.get("reference", {}).get("responses", []),
            **{n: a["responses"] for n, a in data["attacks"].items()}}
    return {(s, r["id"]): r for s, rs in sets.items() for r in rs if r["status"] == "ok"}


def _query(client: OpenAI, model: str, items: list[dict], question: str, workers: int, label: str, prior: dict,
           max_image_mb: float | None) -> list[dict]:
    """run_queries for items not yet answered."""
    old = {i["id"]: p for i in items  # the same image and question sent, else asked again
           if (p := prior.get((label, i["id"]))) and p["image"] == str(i["image"]) and p["question"] == victim_prompt(i.get("question", question))[1]}
    new = {r["id"]: r for r in run_queries(client, model, [i for i in items if i["id"] not in old], question, workers, label,
                                                  max_image_mb=max_image_mb)}
    return [{**i, "question": old[i["id"]]["question"], "response": old[i["id"]]["response"], "status": "ok"} if i["id"] in old
            else new[i["id"]] for i in items]


def evaluate(attack_dirs: str | list[str], *, server_url: str, targets: str | None = None, labels: str | None = None,
             target_text: str | None = None, question: str | None = None, model: str = "qwen/qwen3-vl-8b-instruct",
             judge: str | None = DEFAULT_PROMPT, judge_model: str | None = None, api_key: str = "dummy",
             output: str | None = None, workers: int = 4, max_image_mb: float | None = None, reference: str = "label") -> dict:
    """Victim accuracy and ASR per attack dir vs. clean; resumes.

    Target: target_text (wins, as in the attacks), else `targets` / `labels`, else the manifests' labels or target texts. judge: one of redphys.judge.JUDGES, or None to collect only (then redphys.score).
    max_image_mb: downscale larger images (API limits).
    reference: "label", or "target_image": the victim's answer on each target image (manifests, else `targets`/target) is the
    target, as M-Attack / FOA (judge="gptscore").
    """
    judge_model = judge_model or judge_model_for(judge)
    if reference not in ("label", "target_image") or (reference == "target_image" and target_text):
        raise ValueError(f"reference must be 'label' or 'target_image' (without target_text), got {reference!r}")
    attack_dirs = [attack_dirs] if isinstance(attack_dirs, (str, Path)) else attack_dirs
    labels = json.loads(Path(labels or f"{targets}/labels.json").read_text()) if labels or targets else None
    attacks = {}
    for d in attack_dirs:
        data = load_attack_dir(d)  # a missing dir (a typo) raises
        if not data["valid_results"]:
            print(f"WARNING: skipping {d}: no sample with an existing output")
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

    if labels and not target_text:
        for name, res in attacks.items():
            if miss := [r["id"] for r in res if r["id"] not in labels]:
                print(f"WARNING: {name}: {len(miss)} samples without a label left out (e.g. {miss[0]})")
                attacks[name] = [r for r in res if r["id"] in labels]
    _check_judge(judge, [(r["id"], options(r.get("question", question)), target_of(r["id"], labels, target_text),
                          label_of(r["id"], labels, "source")) for res in attacks.values() for r in res])
    target_images = {}
    if reference == "target_image":
        tdir = Path(targets or "") / "target"
        found = {p.stem: p for p in image_files(tdir)} if targets and tdir.is_dir() else {}
        target_images = {r["id"]: r.get("target_image") or found.get(r["id"]) for res in attacks.values() for r in res}
        if miss := [k for k, v in target_images.items() if not v]:
            raise ValueError(f"{len(miss)} samples (e.g. {miss[0]}) have no target image (pass targets=)")

    print(f"Server:   {server_url}  model={model}")
    print(f"Target:   {target_text!r}" if target_text else f"Targets:  per-case ({len(labels)} cases)")
    client = _client(server_url, api_key, {model} | ({judge_model} if judge and judge != "substring" else set()))

    prior = _prior(output, model)
    if prior:
        print(f"Resuming: {len(prior)} responses in {output}")
    clean = {}
    for results in attacks.values():
        for r in results:
            item = {"id": r["id"], "image": r["source"], **({"question": r["question"]} if r.get("question") else {})}
            clean.setdefault(r["id"], with_options(item, r.get("question", question)))
    victim, attack_responses, target_responses = {**VICTIM, "max_image_mb": max_image_mb}, {}, []

    def save(j=None, quiet=True):  # after every set: a crash keeps the answers paid for
        result = _score(model, question, target_text, labels, clean_responses, attack_responses, j, client, judge_model, workers)
        return _save({**result, "victim": victim, "reference": {"mode": reference, "responses": target_responses}}, output, quiet)

    t0 = time.time()
    clean_responses = _query(client, model, [clean[k] for k in sorted(clean)], question, workers, "clean", prior, max_image_mb)
    print(f"  clean: {len(clean_responses)} responses [{time.time() - t0:.1f}s]{_errors(clean_responses)}")
    if target_images:
        items = [{**clean[k], "image": v} for k, v in sorted(target_images.items())]
        target_responses = [{**r, "image": str(r["image"])}
                            for r in _query(client, model, items, question, workers, "target", prior, max_image_mb)]
        print(f"  target: {len(target_responses)} responses{_errors(target_responses)}")
        labels = {r["id"]: {"source": label_of(r["id"], labels, "source"), "target": r["response"]} for r in target_responses if r["status"] == "ok"}
        attacks = {name: [r for r in res if r["id"] in labels] for name, res in attacks.items()}  # failed target answers drop their samples
        clean_responses = [r for r in clean_responses if r["id"] in labels]
    save()
    for name, results in attacks.items():
        t0 = time.time()
        attack_responses[name] = _query(client, model, [victim_item(r, question) for r in results], question, workers, name, prior, max_image_mb)
        print(f"  {name}: {len(results)} responses [{time.time() - t0:.1f}s]{_errors(attack_responses[name])}")
        save()
    return save(judge, quiet=False)


def score(results: str, *, server_url: str, judge: str = DEFAULT_PROMPT, judge_model: str | None = None, api_key: str = "dummy",
          output: str | None = None, workers: int = 4) -> dict:
    """Judge redphys.evaluate output `results`; writes eval_results.json to `output` (default `results`)."""
    data, judge_model = json.loads((Path(results) / "eval_results.json").read_text()), judge_model or judge_model_for(judge)
    clean, attacks = data["clean_baseline"]["responses"], {name: a["responses"] for name, a in data["attacks"].items()}
    labels = {r["id"]: {"source": r["source"], "target": r["target"]} for r in clean} if data["per_pair_labels"] else None
    _check_judge(judge, [(r["id"], r.get("options") or options(r["question"]), r["target"], r["source"]) for r in clean])
    result = _score(data["model"], data["question"], data["target_text"], labels, clean, attacks, judge,
                    _client(server_url, api_key, {judge_model} if judge and judge != "substring" else set()), judge_model, workers)
    result.update(victim=data["victim"], **({"reference": data["reference"]} if "reference" in data else {}))
    return _save(result, output or results)
