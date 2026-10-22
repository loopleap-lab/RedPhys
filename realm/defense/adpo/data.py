"""(clean, adversarial) pairs from REALM attack outputs."""

import filecmp
import json
import random
from pathlib import Path

from ...data import manifest_samples, resolve


def load_pairs(attack_dirs, holdout=(), test_fraction=0.2, seed=42) -> list[dict]:
    """Split by source id; unchanged images (text-side attacks) are skipped."""
    pairs, same = [], 0
    for d in attack_dirs:
        d = resolve(d)
        data = json.loads((d / "manifest.json").read_text())
        for s in manifest_samples(d, data):
            clean, adv = resolve(s["source"]), resolve(s["output"])
            if filecmp.cmp(clean, adv, shallow=False):
                same += 1
                continue
            pairs.append({"id": s["id"], "attack": data["attack"], "clean": str(clean), "adv": str(adv), "question": s.get("question", "")})
    if same:
        print(f"AdPO: skipped {same} pairs whose image is unchanged")
    if not pairs:
        raise ValueError(f"no image-perturbed samples in {list(attack_dirs)}")
    for p in pairs:
        p["split"] = "test" if random.Random(f"{seed}-{p['id']}").random() < test_fraction else "train"
        p["holdout"] = p["attack"] in holdout
    return pairs


def write_jsonl(path, rows) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
