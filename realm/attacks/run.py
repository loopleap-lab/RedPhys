"""Run an attack over a folder of images (realm.attack and scripts/generate_adversarial.py)."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from ..configs import default_config_path, load_config, with_overrides
from ..data import DATASETS, DEFAULT_QUESTION, image_files, relpath, resolve, save_manifest
from ..data.base_dataset import Sample, load_rgb
from .registry import AttackRegistry, register_all_attacks
from .surrogates import ALREADY_UNIFIED, NO_SURROGATES

STEM_TARGET_ATTACKS = {"foa", "mattack", "oattack", "mpcattack", "coa", "physpatch", "anyattack", "advedm", "imagemix"}
IMGFOLDER_TARGET_ATTACKS = {"advdiffvlm"}
IMAGE_ATTACKS = STEM_TARGET_ATTACKS | IMGFOLDER_TARGET_ATTACKS

TEXT_GUIDED_ATTACKS = {"advedm_r", "vattack", "figstep", "promptinject"}
UNTARGETED_ATTACKS = {"paattack"}
ALL_ATTACKS = sorted(IMAGE_ATTACKS | TEXT_GUIDED_ATTACKS | UNTARGETED_ATTACKS)

_CONFIG_MODULE = {"advedm_r": "advedm"}


def collect_images(source: str, max_samples: int | None = None) -> list[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = image_files(p)
        if not files:
            raise ValueError(f"No images found in {source}")
        return files[:max_samples] if max_samples else files
    raise ValueError(f"Source not found: {source}")


def make_sample(img_path: Path, question: str = "") -> Sample:
    return Sample(id=img_path.stem, images=[load_rgb(img_path)], question=question, ground_truth="", task_type="attack",
                  metadata={"image_file": str(img_path)})


def _make_tmpdir() -> str:
    d = tempfile.mkdtemp(prefix="attack_tmp_")
    atexit.register(lambda: shutil.rmtree(d, ignore_errors=True))
    return d


def prepare_target_dir(target_arg: str, source_images: list[Path]) -> str:
    """Temp dir of <stem>.<lowercase ext> symlinks to the target(s)."""
    target = Path(target_arg)
    tmpdir = _make_tmpdir()
    if target.is_dir():  # relinked with lowercase suffixes, which the attacks look up
        for f in collect_images(str(target)):
            os.symlink(f.resolve(), Path(tmpdir) / f"{f.stem}{f.suffix.lower()}")
        return tmpdir
    if not target.is_file():
        raise ValueError(f"Target not found: {target_arg}")
    for src in source_images:
        os.symlink(target.resolve(), Path(tmpdir) / f"{src.stem}{target.suffix.lower()}")
    return tmpdir


def prepare_imagefolder_dir(target_arg: str, source_images: list[Path]) -> str:
    """Restructure target into ImageFolder format (advdiffvlm)."""
    target = Path(target_arg)
    tmpdir = _make_tmpdir()
    classdir = Path(tmpdir) / "0"
    classdir.mkdir()
    if target.is_file():
        abs_target = target.resolve()
        for i in range(len(source_images)):
            os.symlink(abs_target, classdir / f"{i:05d}{target.suffix}")
    elif target.is_dir():
        by_stem = {p.stem: p for p in collect_images(str(target))}
        for i, src in enumerate(source_images):
            if src.stem not in by_stem:
                raise ValueError(f"No target image for '{src.stem}' in {target_arg}")
            os.symlink(by_stem[src.stem].resolve(), classdir / f"{i:05d}{by_stem[src.stem].suffix}")
    else:
        raise ValueError(f"Target not found: {target_arg}")
    return tmpdir


def create_attack(name: str, args, source_images: list[Path]):
    context = {
        "source_dir": str(Path(args.source).parent) if Path(args.source).is_file() else str(args.source),
        "output_dir": str(args.output),
        "source_images": [p.name for p in source_images],
    }

    if name in STEM_TARGET_ATTACKS and args.target:  # CoA pairs the whole dir
        context["target_dir"] = prepare_target_dir(args.target, collect_images(context["source_dir"]))
    elif name in IMGFOLDER_TARGET_ATTACKS and args.target:
        context["target_dir"] = prepare_imagefolder_dir(args.target, source_images)

    mod_name = _CONFIG_MODULE.get(name, name)
    config_mod = __import__(f"realm.attacks.{mod_name}.config", fromlist=["resolve_cli_kwargs"])
    config_class = AttackRegistry.get_spec(name).config_class
    config_path = args.config or default_config_path("attacks", name)
    cli = config_mod.resolve_cli_kwargs(name, args, context)
    kwargs = {**cli, **(load_config(config_path, config_class) if config_path else {})}  # YAML wins over auto-detected
    if cli.get("target_text"):
        kwargs["target_text"] = cli["target_text"]
    kwargs = with_overrides(kwargs, config_class, args.params)
    print(f"Config:  {config_path or '(code defaults)'}")
    if args.surrogates != "default":
        if name in NO_SURROGATES:
            print(f"WARNING: {name} has no swappable surrogates; surrogates={args.surrogates} ignored")
        elif name not in ALREADY_UNIFIED:
            kwargs["surrogates"] = args.surrogates
    return AttackRegistry.create(name, **kwargs)


def validate_args(args):
    if args.dataset:
        ds = DATASETS[args.dataset]
        args.source = args.source or str(ds["source"])
        args.target = args.target or str(ds["target"])
        args.labels_file = args.labels_file or str(ds["labels"])

    errors = []
    if not args.source:
        errors.append("a source is required")
    if args.attack in IMAGE_ATTACKS and not args.target:
        errors.append(f"{args.attack} needs target images (targets= or target=)")
    if args.attack in TEXT_GUIDED_ATTACKS and not args.target_text and not args.labels_file:
        errors.append(f"{args.attack} needs a target text or per-case labels (target_text=, targets= or labels=)")
    if args.attack == "promptinject" and not args.question:
        args.question = DEFAULT_QUESTION
    if errors:
        raise ValueError("; ".join(errors))


def run(args) -> Path:
    """Attack args.source into args.output (+ manifest.json)."""
    validate_args(args)
    register_all_attacks()

    images = collect_images(args.source, args.max_samples)
    print(f"Attack:  {args.attack}")
    print(f"Source:  {args.source} ({len(images)} images)")
    print(f"Output:  {args.output}")
    print(f"Surrogates: {args.surrogates}")

    per_sample_labels = {}
    if args.labels_file:
        if not Path(args.labels_file).exists():
            raise ValueError(f"labels file not found: {args.labels_file}")
        per_sample_labels = json.loads(Path(args.labels_file).read_text())
        print(f"Labels:  {args.labels_file} ({len(per_sample_labels)} entries)")

    dataset = None
    stem_to_idx = {}
    if args.dataset:
        dataset = DATASETS[args.dataset]["class"](
            data_root=str(DATASETS[args.dataset]["source"].parent),
            question=args.question or DEFAULT_QUESTION,
            max_samples=args.max_samples,
        )
        stem_to_idx = {p.resolve(): i for i, p in enumerate(dataset.entries)}
        print(f"Dataset: {args.dataset} ({len(dataset)} samples)")

    attack = create_attack(args.attack, args, images)
    if args.attack == "vattack" and not per_sample_labels and not attack.config.source_text:
        raise ValueError("vattack needs a source text too: labels_file (or dataset), or source_text in its YAML")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_results = []
    if (out_dir / "manifest.json").exists():
        prior = json.loads((out_dir / "manifest.json").read_text())
        if prior.get("attack", args.attack) != args.attack:
            raise ValueError(f"{out_dir} holds a '{prior['attack']}' run; use another output")
        current = {p.stem for p in images}
        prior_results = [s for s in prior.get("samples", [])
                         if "output" in s and "error" not in s and s["id"] in current and resolve(s["output"]).exists()]
        if prior_results:
            print(f"Resume:  {len(prior_results)} already done, skipping them")
    completed_ids = {s["id"] for s in prior_results}
    results = list(prior_results)
    remaining = [p for p in images if p.stem not in completed_ids]
    t0 = datetime.now()

    cases_file = Path(args.source) / "cases.json"  # a physical-world dataset: the question of each case
    questions = {c["id"]: c["question"] for c in json.loads(cases_file.read_text())} if cases_file.is_file() else {}
    position = {}
    for i, p in enumerate(images):
        position.setdefault(p.stem, i)  # the first of equal stems
    target_files = {p.stem: p for p in collect_images(args.target)} if args.target and Path(args.target).is_dir() else {}
    for img_path in tqdm(remaining, desc=args.attack):
        stem = img_path.stem
        if dataset is not None and img_path.resolve() in stem_to_idx:
            sample = dataset.get_sample(stem_to_idx[img_path.resolve()])
        else:
            sample = make_sample(img_path, question=questions.get(stem, args.question or ""))
        sample_meta = sample.metadata
        if stem in per_sample_labels:
            src, tgt = per_sample_labels[stem].get("source", ""), per_sample_labels[stem].get("target", "")
            sample_meta.update(attack_source_text=src, source_label=src, attack_target_text=tgt, target_label=tgt)

        try:
            try:
                result = attack.generate(sample, sample_idx=position[stem])  # to_pil raises on NaN
            except FloatingPointError as e:
                print(f"\nWARNING: {img_path.name}: {e}, retrying once")
                result = attack.generate(sample, sample_idx=position[stem])
            out_file = out_dir / f"{img_path.stem}.png"
            result.adversarial_sample.save(out_file)
            entry = {
                "id": img_path.stem,
                "source": relpath(img_path),
                "output": relpath(out_file),
                "question": sample.question,
                "perturbation_norm": result.perturbation_norm,
            }
            if sample_meta.get("source_label"):
                entry["source_label"] = sample_meta["source_label"]
            if sample_meta.get("target_label"):
                entry["target_label"] = sample_meta["target_label"]
            if stem in target_files:
                entry["target"] = relpath(target_files[stem])
            used_target = args.target_text or sample_meta.get("attack_target_text")
            if used_target:
                entry["target_text"] = used_target
            if result.metadata:
                entry["metadata"] = result.metadata
            results.append(entry)
        except Exception as e:
            print(f"\nERROR on {img_path.name}: {e}")
            traceback.print_exc()
            results.append({"id": img_path.stem, "source": relpath(img_path), "error": str(e)})

        save_manifest(out_dir, args.attack, results, (datetime.now() - t0).total_seconds())

    duration = (datetime.now() - t0).total_seconds()
    save_manifest(out_dir, args.attack, results, duration)

    n_ok = sum(1 for r in results if "output" in r)
    print(f"\nDone: {n_ok}/{len(images)} images in {duration:.1f}s" + (f" (resumed {len(completed_ids)})" if completed_ids else ""))
    print(f"Output: {out_dir}")
    return out_dir


def attack(name: str, source: str, output: str, *, targets: str | None = None, target: str | None = None, labels: str | None = None,
           target_text: str | None = None, question: str | None = None, config: str | None = None, surrogates: str = "default",
           max_samples: int | None = None, **params) -> Path:
    """Attack `source` images into `output` (+ manifest.json); resumes.

    targets: realm.target output dir (or target / labels separately); params: YAML overrides, e.g. epsilon=8.
    """
    if targets:
        target, labels = target or f"{targets}/target", labels or f"{targets}/labels.json"
    return run(SimpleNamespace(attack=name, dataset=None, source=source, output=output, max_samples=max_samples, surrogates=surrogates,
                               config=config, target=target, target_text=target_text, question=question, labels_file=labels, params=params))
