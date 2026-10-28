"""redphys.target and scripts/target.py."""

import dataclasses
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from ...configs import default_config_path, load_config, with_overrides
from ...data import DATASETS, DEFAULT_QUESTION, PROJECT_ROOT, image_files, relpath, write_json
from .agent import Case, ScenarioAgent, ScenarioConfig


def scenario_config(args) -> ScenarioConfig:
    """Agent YAML + args.params."""
    cfg = load_config(args.config or default_config_path("agents", "scenario"), ScenarioConfig)
    cfg = with_overrides(cfg, ScenarioConfig, args.params)
    return ScenarioConfig(**cfg)


def case_image(path: str) -> str:
    """A case image: absolute, else relative to the working dir if it exists there, else to the project root."""
    p = Path(path)
    return str(p if p.is_absolute() or p.exists() else PROJECT_ROOT / p)


def load_cases(args) -> list[Case]:
    if args.cases:
        rows = json.loads(Path(args.cases).read_text())
        if bad := [c.get("id") for c in rows if not {"image", "question", "answer"} <= c.keys()]:
            raise ValueError(f"{len(bad)} cases (e.g. {bad[0]}) lack image, question or answer")
        if bad := [c["id"] for c in rows if c.get("options") and c["answer"] not in c["options"]]:
            raise ValueError(f"{len(bad)} multiple-choice cases (e.g. {bad[0]}) whose answer is not an option letter")
        if bad := [c["id"] for c in rows if c.get("options") and c.get("target") not in (None, *c["options"])]:
            raise ValueError(f"{len(bad)} multiple-choice cases (e.g. {bad[0]}) whose target is not an option letter")
        return [Case(**{k: v for k, v in c.items() if k in Case.__dataclass_fields__} | {"image": case_image(c["image"])})
                for c in rows]  # extra fields (source, failure_mode) ignored
    if args.dataset:
        args.source, args.labels_file = DATASETS[args.dataset]["source"], DATASETS[args.dataset]["labels"]
    if not args.labels_file:
        raise ValueError("source needs labels_file")
    labels = json.loads(Path(args.labels_file).read_text())
    return [Case(p.stem, str(p), args.question, labels[p.stem]["source"], target=labels[p.stem].get("target") if args.keep_targets else None)
            for p in image_files(args.source) if p.stem in labels]


def run(args) -> Path:
    """Targets -> target/<id>.png, labels.json, targets.json."""
    if not os.environ.get("REDPHYS_AGENT_LOCAL"):  # set inside the container
        from .docker import run_in_docker

        return run_in_docker(args)
    config = scenario_config(args)
    agent, cases, out = ScenarioAgent(config), load_cases(args)[:args.max_samples or None], Path(args.output)  # 0: all
    (out / "target").mkdir(parents=True, exist_ok=True)
    write_json(out / "cases.json", [dataclasses.asdict(c) | {"image": relpath(c.image)} for c in cases])
    specs_file = out / "targets.json"
    specs = json.loads(specs_file.read_text()) if specs_file.exists() else {}
    todo = [c for c in cases if c.id not in specs or not (out / "target" / f"{c.id}.png").exists()]
    print(f"Cases: {len(cases)} ({len(cases) - len(todo)} done)  model={config.model}  generator={config.generator}")
    with ThreadPoolExecutor(args.workers) as pool:  # saved as they finish
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
            write_json(specs_file, specs)  # atomic: a killed run keeps a readable file
    labels = {k: {"source": s["source_text"], "target": s["target_text"]} for k, s in specs.items()}
    write_json(out / "labels.json", labels)
    unverified = [k for k, s in specs.items() if not s["verified"]]
    print(f"Done: {len(specs)}/{len(cases)} targets ({len(specs) - len(unverified)} verified) -> {out}")
    if unverified:
        print(f"Not verified (the checker did not accept the last image): {', '.join(unverified)}")
    return out


def target(cases: str | None = None, output: str | None = None, *, dataset: str | None = None, source: str | None = None, labels: str | None = None,
           question: str = DEFAULT_QUESTION, keep_targets: bool = False, max_samples: int | None = None, config: str | None = None,
           workers: int = 8, **params) -> Path:
    """Target answer + image per case; resumes.

    cases: JSON (id, image, question, answer, [options, domain, task, target]); or dataset / source.
    params: settings over configs/agents/scenario.yaml, e.g. max_rounds=3.
    """
    if not output:
        raise ValueError("output is required (the only dir the agent may write)")
    return run(SimpleNamespace(cases=cases, dataset=dataset, source=source, labels_file=labels, question=question, keep_targets=keep_targets,
                               output=output, max_samples=max_samples, config=config, workers=workers, params=params))
