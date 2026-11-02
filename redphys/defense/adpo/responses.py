"""Stage 1: training rows; redphys: with the original model's y_w (clean) and y_l (adversarial)."""

import json
import random
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from ...data import image_data_url
from ...judge import no_reasoning
from .data import read_jsonl
from .models import answer, encode


def answer_remote(client, model_id: str, image, prompt: str, max_new_tokens: int) -> tuple[str, bool]:
    content = [{"type": "image_url", "image_url": {"url": image_data_url(image)}}, {"type": "text", "text": prompt}]
    out = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": content}], temperature=0.0,
                                         max_tokens=max_new_tokens, extra_body=no_reasoning(client, model_id)).choices[0]
    return (out.message.content or "").strip(), out.finish_reason != "length"


def answers(cfg, todo: list[dict], rows: list[dict], model, vlm):
    """The rows of todo with y_w (once per clean image and prompt) and y_l, yielded as they are made."""
    client = None
    if cfg.server_url:
        from openai import OpenAI
        client = OpenAI(base_url=f"{cfg.server_url}/v1", api_key="EMPTY")

    def respond(image_path, prompt):
        img = vlm.sized(image_path, cfg.max_pixels)
        if client:
            return answer_remote(client, cfg.model, img, prompt, cfg.max_new_tokens)
        return answer(model, vlm, {k: v.to(model.device) for k, v in encode(vlm, img, prompt).items()}, cfg.max_new_tokens)

    def row(p):
        (y_w, w_end), (y_l, l_end) = clean[p["clean"], p["prompt"]], respond(p["adv"], p["prompt"])
        return {**p, "y_w": y_w, "w_end": w_end, "y_l": y_l, "l_end": l_end}

    run = ThreadPoolExecutor(16).map if client else map
    clean = {(r["clean"], r["prompt"]): (r["y_w"], r["w_end"]) for r in rows}
    need = list({(p["clean"], p["prompt"]) for p in todo} - clean.keys())
    clean.update(zip(need, tqdm(run(lambda k: respond(*k), need), total=len(need), desc="AdPO y_w")))
    yield from tqdm(run(row, todo), total=len(todo), desc="AdPO y_l")


def prepare(cfg, pairs: list[dict], model, vlm, path) -> list[dict]:
    """The training rows, new ones appended to path (resumable); pgd: one row per source, answered in training."""
    key = lambda p: (p["id"], p["attack"], p["adv"])
    rows = read_jsonl(path)
    done = {key(r) for r in rows}
    train = [p for p in pairs if p["split"] == "train" and not p["holdout"]]
    if cfg.adv_source == "pgd":
        train = list({p["clean"]: p for p in train}.values())
    if cfg.max_pairs:  # a random subset, not the first attack's pairs
        train = random.Random(cfg.seed).sample(train, min(cfg.max_pairs, len(train)))
    todo = [{**p, "prompt": cfg.prompt or p["question"]} for p in train if key(p) not in done]
    new = []
    with path.open("a") as f:
        for r in todo if cfg.adv_source == "pgd" else answers(cfg, todo, rows, model, vlm):
            f.write(json.dumps(r) + "\n")
            f.flush()
            new.append(r)
    return rows + new
