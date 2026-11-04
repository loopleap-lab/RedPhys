"""redphys.defend and scripts/defense.py."""

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from ..configs import build_config, config_snapshot, seeded
from ..data import input_stamp, list_images, manifest_samples, relpath, resolve, save_manifest
from .registry import DefenseRegistry


def load_samples(path_str: str, max_samples: int | None) -> tuple[str, list[tuple[Path, dict]]]:
    """(attack name, [(image, manifest entry)]); max_samples None or 0: all."""
    p = Path(path_str)
    if p.is_file():
        if (p.parent / "manifest.json").exists():
            name, samples = load_samples(str(p.parent), None)
            hit = [(img, s) for img, s in samples if img.resolve() == p.resolve()]
            if hit:
                return name, hit
        print("WARNING: no manifest entry; evaluation will treat the input image as its own clean source")
        return p.parent.name, [(p, {"id": p.stem, "source": relpath(p)})]
    manifest = p / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        samples = [(img, s) for s in manifest_samples(p.resolve(), data) if (img := resolve(s["output"])).exists()]
        return data.get("attack", p.name), samples[:max_samples] if max_samples else samples
    files = list_images(p, max_samples)
    print("WARNING: no manifest.json; evaluation will treat the input images as their own clean sources")
    return p.name, [(f, {"id": f.stem, "source": relpath(f)}) for f in files]


def fit(args) -> Path:
    """A training-based defense (e.g. AdPO): fit it on the attack dirs in args.input -> args.output (checkpoint, manifest.json)."""
    inputs = [args.input] if isinstance(args.input, (str, Path)) else list(args.input)
    out_dir = Path(args.output)
    if any(out_dir.resolve() == Path(i).resolve() for i in inputs):
        raise ValueError("output must differ from the input directories")
    config_class = DefenseRegistry.get_spec(args.defense).config_class
    defense = DefenseRegistry.create(args.defense, **build_config("defenses", args.defense, config_class, args.config, args.params))
    config, inputs_rel = config_snapshot(defense.config), [relpath(i) for i in inputs]
    if (out_dir / "manifest.json").exists():  # resume only the same fit
        prior = json.loads((out_dir / "manifest.json").read_text())
        if prior.get("config") != config or prior.get("inputs") != inputs_rel:
            raise ValueError(f"{out_dir} holds another {args.defense} fit (other config or inputs); use another output")
    out_dir.mkdir(parents=True, exist_ok=True)
    main = int(os.environ.get("RANK", 0)) == 0  # under accelerate every rank runs this; one writes the manifest
    if main:
        save_manifest(out_dir, args.defense, [], 0.0, defense=args.defense, config=config, inputs=inputs_rel)  # claims the dir
    t0 = datetime.now()
    outputs = defense.fit([str(i) for i in inputs], out_dir)
    if main:
        save_manifest(out_dir, args.defense, [], (datetime.now() - t0).total_seconds(), defense=args.defense,
                      config=config, inputs=inputs_rel, outputs=outputs)
    print(f"\nDone: {args.defense} fitted on {len(inputs)} attack dirs -> {out_dir}")
    return out_dir


def run(args) -> Path:
    """Clean args.input with args.defense -> args.output."""

    print(f"Defense: {args.defense}")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    if getattr(DefenseRegistry.get_spec(args.defense).cls, "trains", False):
        return fit(args)
    if not isinstance(args.input, (str, Path)):
        raise ValueError(f"{args.defense} cleans one input; several attack dirs are for training-based defenses")

    source_attack, samples = load_samples(args.input, args.max_samples)
    print(f"Images:  {len(samples)} from '{source_attack}'")
    if not samples:
        raise ValueError(f"No images found in {args.input} (no manifest sample with an existing output)")

    if any("defense" in s.get("metadata", {}) for _, s in samples):
        raise ValueError("the input is already defended; chaining defenses would drop the first one's metadata (e.g. BlueSuffix's prompt)")
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if out_dir.resolve() == (in_dir if in_dir.is_dir() else in_dir.parent).resolve():
        raise ValueError("output must differ from the input directory")

    config_class = DefenseRegistry.get_spec(args.defense).config_class
    defense = DefenseRegistry.create(args.defense, **build_config("defenses", args.defense, config_class, args.config, args.params))

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{source_attack}+{args.defense}"
    config = config_snapshot(defense.config)

    results, prior_duration = [], 0.0
    if (out_dir / "manifest.json").exists():
        prior = json.loads((out_dir / "manifest.json").read_text())
        if prior.get("attack") != name or prior.get("config") != config:
            raise ValueError(f"{out_dir} holds another run ('{prior.get('attack')}', other config or older version); use another output")
        prior_duration = prior.get("duration_s", 0.0)  # accumulated over resumes
        inputs = {e["id"]: input_stamp(img) for img, e in samples}
        results = [r for r in prior["samples"]  # inputs outside this selection are kept
                   if "error" not in r and r.get("input") == inputs.get(r["id"], r.get("input")) and resolve(r["output"]).exists()]
        if n := sum(r["id"] in inputs for r in results):
            print(f"Resume:  {n} already done, skipping them")
    done = {r["id"] for r in results}
    t0 = datetime.now()

    warned = False
    for img_path, entry in tqdm([(i, e) for i, e in samples if e["id"] not in done], desc="Cleaning"):
        try:
            meta = entry.get("metadata", {})
            prompt = meta.get("adversarial_question") or meta.get("text_prompt") or entry.get("question") or args.question
            c = defense.config
            if not prompt and not warned and (getattr(c, "enable_text_purifier", False) or getattr(c, "enable_suffix_generator", False)):
                print(f"\nWARNING: no question for {entry['id']} (manifest or question=): its text steps are skipped")
                warned = True
            result = seeded(lambda: defense.clean(str(img_path), prompt=prompt or None), defense.config.seed, entry["id"], img_path.name)
            out_path = out_dir / f"{img_path.stem}.png"
            result.cleaned_sample.save(out_path)
            results.append({**{k: v for k, v in entry.items() if k != "perturbation_norm"},  # the attack's: stale after cleaning
                            "output": relpath(out_path), "input": input_stamp(img_path),
                            "metadata": {**meta, "defense": {"name": args.defense, **result.metadata}}})
        except Exception as e:
            print(f"\nERROR on {img_path.name}: {e}")
            traceback.print_exc()
            results.append({"id": entry["id"], "source": entry.get("source"), "error": str(e)})
        save_manifest(out_dir, name, results, prior_duration + (datetime.now() - t0).total_seconds(), defense=args.defense, config=config)

    duration = (datetime.now() - t0).total_seconds()
    save_manifest(out_dir, name, results, prior_duration + duration, defense=args.defense, config=config)
    selected = {e["id"] for _, e in samples}
    n_ok = sum("error" not in r for r in results if r["id"] in selected)
    n_new = len(results) - len(done)
    avg = f"{duration/n_new:.1f}s/img" if n_new else "N/A"
    print(f"\nDone: {n_ok}/{len(selected)} images in {duration:.1f}s ({avg})")
    if selected and not n_ok:
        raise ValueError(f"every sample failed ({next(r['error'] for r in reversed(results) if 'error' in r)[:300]})")
    return out_dir


def defend(name: str, input: str | list[str], output: str, *, question: str | None = None, config: str | None = None,
           max_samples: int | None = None, **params) -> Path:
    """Clean `input` (a redphys.attack dir) with defense `name` -> `output` (images, manifest.json); params override its YAML.
    A training-based defense (adpo) takes one or more attack dirs and writes a checkpoint instead."""
    return run(SimpleNamespace(defense=name, input=input, output=output, question=question, config=config, max_samples=max_samples, params=params))
