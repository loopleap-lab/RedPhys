"""Attack targets with the scenario agent (realm.target and scripts/generate_targets.py)."""

import dataclasses
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from ...configs import default_config_path, load_config, with_overrides
from ...data import DATASETS, DEFAULT_QUESTION, image_files
from .agent import Case, ScenarioAgent, ScenarioConfig


def scenario_config(args) -> ScenarioConfig:
    """Agent YAML + args.params."""
    cfg = load_config(args.config or default_config_path("agents", "scenario"), ScenarioConfig)
    cfg = with_overrides(cfg, ScenarioConfig, getattr(args, "params", None) or {})
    return ScenarioConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()})


def load_cases(args) -> list[Case]:
    if args.cases:
        return [Case(**{k: v for k, v in c.items() if k in Case.__dataclass_fields__}) for c in json.loads(Path(args.cases).read_text())]  # extra fields (source, failure_mode) ignored
    if args.dataset:
        args.source, args.labels_file = DATASETS[args.dataset]["source"], DATASETS[args.dataset]["labels"]
    if not args.labels_file:
        raise ValueError("source needs labels_file")
    labels = json.loads(Path(args.labels_file).read_text())
    return [Case(p.stem, str(p), args.question, labels[p.stem]["source"], target=labels[p.stem].get("target") if args.keep_targets else None)
            for p in image_files(args.source) if p.stem in labels]


def run(args) -> Path:
    """Targets (answer + image) for the cases of args.cases / args.dataset / args.source; output: target/<id>.png, labels.json, targets.json."""
    if not os.environ.get("REALM_AGENT_LOCAL"):  # the agent runs in its container; inside it, this is set
        from .docker import run_in_docker

        return run_in_docker(args)
    config = scenario_config(args)
    agent, cases, out = ScenarioAgent(config), load_cases(args)[:args.max_samples], Path(args.output)
    (out / "target").mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps([dataclasses.asdict(c) for c in cases], indent=2))
    specs_file = out / "targets.json"
    specs = json.loads(specs_file.read_text()) if specs_file.exists() else {}
    todo = [c for c in cases if c.id not in specs or not (out / "target" / f"{c.id}.png").exists()]
    print(f"Cases: {len(cases)} ({len(cases) - len(todo)} done)  model={config.model}  generator={config.generator}")
    with ThreadPoolExecutor(args.workers) as pool:  # results saved here, in completion order
        jobs = {pool.submit(agent.run, case): case for case in todo}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc="targets"):
            case = jobs[job]
            try:
                spec, img = job.result()
            except Exception as e:
                print(f"ERROR on {case.id}: {e}")
                continue
            img.save(out / "target" / f"{case.id}.png")
            specs[case.id] = dataclasses.asdict(spec)
            specs_file.write_text(json.dumps(specs, indent=2))
    labels = {k: {"source": s["source_text"], "target": s["target_text"]} for k, s in specs.items()}
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    unverified = [k for k, s in specs.items() if not s["verified"]]
    print(f"Done: {len(specs)}/{len(cases)} targets ({len(specs) - len(unverified)} verified) -> {out}")
    if unverified:
        print(f"Not verified (the checker did not accept the last image): {', '.join(unverified)}")
    return out


def target(cases: str | None = None, output: str = "", *, dataset: str | None = None, source: str | None = None, labels: str | None = None,
           question: str = DEFAULT_QUESTION, keep_targets: bool = False, max_samples: int | None = None, config: str | None = None,
           workers: int = 8, **params) -> Path:
    """Targets (answer + image) for the cases of a JSON file (id, image, question, answer, [options, domain, task]), or a dataset / image dir; resumes.

    params: settings over configs/agents/scenario.yaml, e.g. max_rounds=3.
    """
    return run(SimpleNamespace(cases=cases, dataset=dataset, source=source, labels_file=labels, question=question, keep_targets=keep_targets,
                               output=output, max_samples=max_samples, config=config, workers=workers, params=params))
