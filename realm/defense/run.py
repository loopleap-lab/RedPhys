"""Clean adversarial images with a defense (realm.defend and scripts/clean_adversarial.py)."""

import dataclasses
import json
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from ..configs import default_config_path, load_config, with_overrides
from ..data import image_files, relpath, resolve, save_manifest
from .registry import DefenseRegistry


def load_samples(path_str: str, max_samples: int | None) -> tuple[str, list[tuple[Path, dict]]]:
    """(attack name, [(image, manifest entry)])."""
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
        samples = [(resolve(s["output"]), s) for s in data.get("samples", []) if "error" not in s]
        samples = [(img, s) for img, s in samples if img.exists()]
        return data.get("attack", p.name), samples[:max_samples]
    print("WARNING: no manifest.json; evaluation will treat the input images as their own clean sources")
    files = image_files(p)[:max_samples]
    if not files:
        raise ValueError(f"No images found in {path_str}")
    if len({f.stem for f in files}) < len(files):
        raise ValueError(f"Duplicate file stems in {path_str} (e.g. a.jpg and a.png): outputs are named <stem>.png")
    return p.name, [(f, {"id": f.stem, "source": relpath(f)}) for f in files]


def _stamp(img: Path) -> list:
    """[path, mtime_ns] of an input."""
    return [relpath(img), img.stat().st_mtime_ns]


def run(args) -> Path:
    """Clean the images of args.input with defense args.defense; images and manifest.json go to args.output."""

    print(f"Defense: {args.defense}")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")

    source_attack, samples = load_samples(args.input, args.max_samples)
    print(f"Images:  {len(samples)} from '{source_attack}'")

    if any("defense" in s.get("metadata", {}) for _, s in samples):
        raise ValueError("the input is already defended; chaining defenses would drop the first one's metadata (e.g. BlueSuffix's prompt)")
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if out_dir.resolve() == (in_dir if in_dir.is_dir() else in_dir.parent).resolve():
        raise ValueError("output must differ from the input directory")

    spec = DefenseRegistry.get_spec(args.defense)
    config_path = args.config or default_config_path("defenses", args.defense)
    print(f"Config:  {config_path or '(code defaults)'}")
    defense = DefenseRegistry.create(args.defense, **with_overrides(load_config(config_path, spec.config_class) if config_path else {}, spec.config_class, args.params))

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{source_attack}+{args.defense}"
    config = {k: v for k, v in dataclasses.asdict(defense.config).items() if k != "device" and "key" not in k}

    results = []
    if (out_dir / "manifest.json").exists():
        prior = json.loads((out_dir / "manifest.json").read_text())
        if prior.get("attack") != name or prior.get("config") != config:
            raise ValueError(f"{out_dir} holds another run ('{prior.get('attack')}', other config or older version); use another output")
        inputs = {e["id"]: _stamp(img) for img, e in samples}
        results = [r for r in prior["samples"]
                   if "error" not in r and r.get("input") == inputs.get(r["id"]) and resolve(r["output"]).exists()]
        if results:
            print(f"Resume:  {len(results)} already done, skipping them")
    done = {r["id"] for r in results}
    t0 = datetime.now()

    warned = False
    for img_path, entry in tqdm([(i, e) for i, e in samples if e["id"] not in done], desc="Cleaning"):
        try:
            meta = entry.get("metadata", {})
            prompt = meta.get("adversarial_question") or meta.get("text_prompt") or entry.get("question") or args.question
            if not prompt and not warned and hasattr(defense.config, "enable_text_purifier"):
                print(f"\nWARNING: no question for {entry['id']} (manifest or question=): its text steps are skipped")
                warned = True
            result = defense.clean(str(img_path), **({"prompt": prompt} if prompt else {}))
            out_path = out_dir / f"{img_path.stem}.png"
            result.cleaned_sample.save(out_path)
            defense_meta = {"name": args.defense, "detection_confidence": result.detection_confidence,
                            "regions_removed": result.regions_removed, **result.metadata}
            results.append({**entry, "output": relpath(out_path), "input": _stamp(img_path),
                            "metadata": {**meta, "defense": defense_meta}})
        except Exception as e:
            print(f"\n[ERROR] {img_path.name}: {e}")
            traceback.print_exc()
            results.append({"id": entry["id"], "source": entry.get("source"), "error": str(e)})
        save_manifest(out_dir, name, results, (datetime.now() - t0).total_seconds(), defense=args.defense, config=config)

    duration = (datetime.now() - t0).total_seconds()
    ok = [r for r in results if "error" not in r]
    n_new = len(results) - len(done)
    avg = f"{duration/n_new:.1f}s/img" if n_new else "N/A"
    print(f"\nDone: {len(ok)}/{len(results)} images in {duration:.1f}s ({avg})")
    if results and not ok:
        raise ValueError(f"every sample failed ({results[-1]['error'][:300]})")
    return out_dir


def defend(name: str, input: str, output: str, *, question: str | None = None, config: str | None = None, max_samples: int | None = None,
           **params) -> Path:
    """Clean the adversarial images of `input` (an attack output dir) with defense `name`; images and manifest.json go to `output`.

    params: hyperparameters over the defense's YAML.
    """
    return run(SimpleNamespace(defense=name, input=input, output=output, question=question, config=config, max_samples=max_samples, params=params))
