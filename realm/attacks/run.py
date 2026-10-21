"""Run an attack over a folder of images (realm.attack and scripts/generate_adversarial.py)."""

from __future__ import annotations

import atexit
import dataclasses
import json
import os
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from ..configs import build_config, config_snapshot, seeded
from ..data import DATASETS, DEFAULT_QUESTION, PROJECT_ROOT, Sample, input_stamp, list_images, load_rgb, relpath, resolve, save_manifest
from ..judge import options
from .registry import AttackRegistry
from .surrogates import NO_SURROGATES, SURROGATE_MODES

ALL_ATTACKS = AttackRegistry.names()
IMAGEFOLDER_TARGETS = {"advdiffvlm"}  # pairs targets by ImageFolder order, not by stem
RUN_INPUTS = ("target_images_dir", "clean_images_dir", "source_images_dir", "source_images", "gradcam_masks_dir")  # per run: not in the manifest config


def config_fields(name: str) -> set[str]:
    return {f.name for f in dataclasses.fields(AttackRegistry.get_spec(name).config_class)}



def run_inputs(context: dict) -> dict:
    """Run inputs by the config field that receives them: an attack gets those its config declares (see AttackConfig)."""
    return {
        "target_images_dir": context.get("target_dir"),
        "clean_images_dir": context["source_dir"],
        "source_images_dir": context["source_dir"],
        "source_images": context["source_images"],
        "gradcam_masks_dir": str(Path(context["output_dir"]) / "_gradcam_masks"),
    }


def case_question(case: dict) -> str:
    """A cases.json question, with its options appended as "A. ..." lines when the text has none."""
    q, opts = case["question"], case.get("options")
    return q + "\n" + "\n".join(f"{k}. {v}" for k, v in opts.items()) if opts and not options(q) else q


def _case_image(image: str, cases_dir: Path) -> Path:
    """A cases.json image path: absolute, else from the project root, else from the cases.json dir."""
    p = Path(image)
    return (p if p.is_absolute() else next((q for q in (PROJECT_ROOT / p, cases_dir / p) if q.exists()), PROJECT_ROOT / p)).resolve()


def case_questions(images: list[Path]) -> dict:
    """stem -> question from the nearest cases.json at or above each image's dir (up to dataset/, elsewhere the dir and its parent).

    Rows match by image path, else by id == stem in the nearest cases.json.
    """
    datasets, cache, out = PROJECT_ROOT / "dataset", {}, {}
    for img in images:
        chain = [img.absolute().parent, *img.absolute().parent.parents]  # the dirs as given: symlinked images keep their source's cases.json
        top = next((i for i, x in enumerate(chain) if x.resolve() == datasets), 2)
        found = [f for f in (x / "cases.json" for x in chain[:top]) if f.is_file()]
        for f in found:
            if f not in cache:
                rows = [c for c in json.loads(f.read_text()) if "question" in c]
                cache[f] = ({_case_image(c["image"], f.parent): c for c in rows if "image" in c}, {c["id"]: c for c in rows if "id" in c})
        case = next((cache[f][0][img.resolve()] for f in found if img.resolve() in cache[f][0]), None) \
            or (cache[found[0]][1].get(img.stem) if found else None)
        if case:
            out[img.stem] = case_question(case)
    return out


def make_sample(img_path: Path, question: str = "") -> Sample:
    return Sample(id=img_path.stem, images=[load_rgb(img_path)], question=question)


def _make_tmpdir() -> str:
    d = tempfile.mkdtemp(prefix="attack_tmp_")
    atexit.register(lambda: shutil.rmtree(d, ignore_errors=True))
    return d


def prepare_target_dir(target_arg: str, source_images: list[Path]) -> str:
    """Temp dir of <stem>.<lowercase ext> symlinks to the target(s)."""
    target = Path(target_arg)
    tmpdir = _make_tmpdir()
    if target.is_dir():  # relinked with lowercase suffixes, which the attacks look up
        for f in list_images(target):
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
        by_stem = {p.stem: p for p in list_images(target)}
        for i, src in enumerate(source_images):
            if src.stem not in by_stem:
                raise ValueError(f"No target image for '{src.stem}' in {target_arg}")
            os.symlink(by_stem[src.stem].resolve(), classdir / f"{i:05d}{by_stem[src.stem].suffix}")
    else:
        raise ValueError(f"Target not found: {target_arg}")
    return tmpdir


def attack_kwargs(name: str, args, source_images: list[Path]) -> dict:
    """Config kwargs: run inputs < the YAML < target_text= < params."""
    is_file = Path(args.source).is_file()
    source_dir = str(Path(args.source).parent) if is_file else str(args.source)
    context = {"source_dir": source_dir, "output_dir": str(args.output), "source_images": [p.name for p in source_images]}
    fields = config_fields(name)
    if args.target and "target_images_dir" in fields:
        # a dir source: one target link per image of the whole dir, as CoA's caption files have one line per sorted image
        context["target_dir"] = (prepare_imagefolder_dir(args.target, source_images) if name in IMAGEFOLDER_TARGETS
                                 else prepare_target_dir(args.target, source_images if is_file else list_images(source_dir)))
    params = {**({"target_text": args.target_text} if args.target_text and "target_text" in fields else {}), **args.params}
    kwargs = build_config("attacks", name, AttackRegistry.get_spec(name).config_class, args.config, params,
                          base={k: v for k, v in run_inputs(context).items() if k in fields})
    if args.surrogates != "default":
        if name in NO_SURROGATES:
            print(f"WARNING: {name} has no swappable surrogates; surrogates={args.surrogates} ignored")
        else:  # attacks whose own surrogates are the unified ones ignore it
            kwargs["surrogates"] = args.surrogates
    return kwargs


def validate_args(args):
    if args.dataset:
        ds = DATASETS[args.dataset]
        args.source = args.source or str(ds["source"])
        args.target = args.target or str(ds["target"])
        args.labels_file = args.labels_file or str(ds["labels"])
        args.question = args.question or DEFAULT_QUESTION

    errors = []
    if args.attack not in ALL_ATTACKS:
        raise ValueError(f"unknown attack {args.attack!r}; available: {', '.join(ALL_ATTACKS)}")
    fields = config_fields(args.attack)
    if args.surrogates not in SURROGATE_MODES:
        errors.append(f"surrogates must be one of {SURROGATE_MODES}, got {args.surrogates!r}")
    if not args.source:
        errors.append("a source is required")
    if "target_images_dir" in fields and not args.target:
        errors.append(f"{args.attack} needs target images (targets= or target=)")
    if "target_text" in fields and not args.target_text and not args.labels_file:
        errors.append(f"{args.attack} needs a target text or per-case labels (target_text=, targets= or labels=)")
    if AttackRegistry.get_spec(args.attack).config_class.uses_question and not args.question:
        args.question = DEFAULT_QUESTION  # cases.json questions still come first
    if errors:
        raise ValueError("; ".join(errors))


def run(args) -> Path:
    """Attack args.source into args.output (+ manifest.json)."""
    validate_args(args)

    images = list_images(args.source, args.max_samples)
    print(f"Attack:  {args.attack}")
    print(f"Source:  {args.source} ({len(images)} images)")
    print(f"Output:  {args.output}")
    print(f"Surrogates: {args.surrogates}")
    src, out_dir = Path(args.source), Path(args.output)
    if out_dir.resolve() == (src if src.is_dir() else src.parent).resolve():
        raise ValueError("output must differ from the source directory")

    per_sample_labels = {}
    if args.labels_file:
        if not Path(args.labels_file).exists():
            raise ValueError(f"labels file not found: {args.labels_file}")
        per_sample_labels = json.loads(Path(args.labels_file).read_text())
        print(f"Labels:  {args.labels_file} ({len(per_sample_labels)} entries)")

    spec, fields = AttackRegistry.get_spec(args.attack), config_fields(args.attack)
    kwargs = attack_kwargs(args.attack, args, images)
    if spec.config_class.needs_source_text and not per_sample_labels and not kwargs.get("source_text"):
        raise ValueError(f"{args.attack} needs the correct answer too: labels (labels=, targets= or a dataset)"
                         + (", or source_text in its YAML" if "source_text" in fields else ""))
    attack = AttackRegistry.create(args.attack, **kwargs)

    config = config_snapshot(attack.config, drop=RUN_INPUTS)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, prior_duration = [], 0.0
    if (out_dir / "manifest.json").exists():
        prior = json.loads((out_dir / "manifest.json").read_text())
        if "config" not in prior and prior.get("attack") == args.attack:  # manifests before the config field
            print(f"WARNING: {out_dir} has no recorded config; resuming with the current one")
        elif prior.get("attack") != args.attack or prior.get("config") != config:
            raise ValueError(f"{out_dir} holds another run ('{prior.get('attack')}', other config or older version); use another output")
        prior_duration = prior.get("duration_s", 0.0)  # accumulated over resumes
        inputs = {p.stem: input_stamp(p) for p in images}
        results = [s for s in prior.get("samples", [])  # inputs outside this selection are kept
                   if "error" not in s and s.get("input", inputs.get(s["id"])) == inputs.get(s["id"], s.get("input"))  # no stamp: older manifest
                   and resolve(s["output"]).exists()]
        if n := sum(s["id"] in inputs for s in results):
            print(f"Resume:  {n} already done, skipping them")
    done = {s["id"] for s in results}
    remaining = [p for p in images if p.stem not in done]
    t0 = datetime.now()

    questions = case_questions(remaining)  # datasets with a cases.json: the question of each case
    position = {p.stem: i for i, p in enumerate(images)}
    target_files = {p.stem: p for p in list_images(args.target)} if args.target and Path(args.target).is_dir() else {}
    for img_path in tqdm(remaining, desc=args.attack):
        stem = img_path.stem
        sample = make_sample(img_path, question=questions.get(stem, args.question or ""))
        sample_meta = sample.metadata
        if stem in per_sample_labels:
            src_text, tgt_text = per_sample_labels[stem].get("source", ""), per_sample_labels[stem].get("target", "")
            sample_meta.update(attack_source_text=src_text, source_label=src_text, attack_target_text=tgt_text, target_label=tgt_text)

        try:
            result = seeded(lambda: attack.generate(sample, sample_idx=position[stem]), attack.config.seed, stem, img_path.name)
            out_file = out_dir / f"{stem}.png"
            result.adversarial_sample.save(out_file)
            entry = {
                "id": stem,
                "source": relpath(img_path),
                "output": relpath(out_file),
                "input": input_stamp(img_path),
                "question": sample.question,
                "perturbation_norm": result.perturbation_norm,
            }
            if sample_meta.get("source_label"):
                entry["source_label"] = sample_meta["source_label"]
            if sample_meta.get("target_label"):
                entry["target_label"] = sample_meta["target_label"]
            if stem in target_files:
                entry["target"] = relpath(target_files[stem])
            if "target_text" in fields and (used := getattr(attack.config, "target_text", None) or sample_meta.get("attack_target_text")):
                entry["target_text"] = used  # what the attack aimed at: evaluate's target when no labels are given
            if result.metadata:
                entry["metadata"] = result.metadata
            results.append(entry)
        except Exception as e:
            print(f"\nERROR on {img_path.name}: {e}")
            traceback.print_exc()
            results.append({"id": stem, "source": relpath(img_path), "error": str(e)})

        save_manifest(out_dir, args.attack, results, prior_duration + (datetime.now() - t0).total_seconds(), config=config)

    duration = (datetime.now() - t0).total_seconds()
    save_manifest(out_dir, args.attack, results, prior_duration + duration, config=config)

    selected = {p.stem for p in images}
    n_ok = sum("output" in r for r in results if r["id"] in selected)
    print(f"\nDone: {n_ok}/{len(images)} images in {duration:.1f}s" + (f" (resumed {len(done & selected)})" if done & selected else ""))
    print(f"Output: {out_dir}")
    if not n_ok:
        raise ValueError(f"every sample failed ({next(r['error'] for r in reversed(results) if 'error' in r)[:300]})")
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
