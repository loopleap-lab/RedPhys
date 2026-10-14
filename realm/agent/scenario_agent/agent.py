"""Agentic target generation (REALM paper, Sec. 3.4): an LLM calls tools (plan, edit, generate, check, submit) per test case."""

import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import requests
from openai import OpenAI
from PIL import Image

from ...data import load_rgb
from ...judge import clean_llm_text, no_reasoning
from .prompts import AGENT_TOOLS, BLIND_CHECK, CHECK, DOMAIN_CONSTRAINTS, GENERAL_SYSTEM, PHYSICAL_SYSTEM


@dataclass
class ScenarioConfig:
    model: str = "openai/gpt-6-luna"
    server_url: str = "https://openrouter.ai/api"
    api_key_env: str = "OPENROUTER_API_KEY"
    generator: str = "openrouter"  # image model: "openrouter" or "local" (target_generator server)
    generator_url: str = "http://localhost:8091"
    generator_model: str = "qwen/qwen-image-3"
    checker_models: tuple = ("anthropic/claude-haiku-5.5", "meta-llama/llama-4-maverick", "mistralai/mistral-medium-3.1")  # blind answerers
    checker_server_url: str = ""  # "" = server_url
    checker_api_key_env: str = ""  # "" = api_key_env
    max_rounds: int = 5  # images the agent may make
    max_steps: int = 24  # tool-calling turns
    num_inference_steps: int = 40
    seed: int = 0

    def __post_init__(self):
        if self.generator not in ("local", "openrouter"):
            raise ValueError(f"generator: local|openrouter; got {self.generator!r}")


@dataclass
class Case:
    id: str
    image: str
    question: str
    answer: str  # option letter when options are given, else text
    options: Optional[dict] = None  # {"A": ...}: multiple choice, physical-world prompt
    domain: str = "general"
    task: str = ""
    target: Optional[str] = None  # fixed wrong answer (open-ended)


@dataclass
class TargetSpec:
    id: str
    failure_mode: str
    attack_target: str
    target_text: str
    source_text: str
    source_cue: str
    target_cue: str
    prompt: str
    rounds: int
    verified: bool
    log: list = field(default_factory=list)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _b64(img: Image.Image) -> str:
    return base64.b64encode(_png(img)).decode()


def openrouter_image(server_url: str, key: str, model: str, prompt: str, refs=()) -> Image.Image:
    """One image from an OpenRouter image model (refs: reference images for edit mode)."""
    body = {"model": model, "prompt": prompt}
    if refs:
        body["input_references"] = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(r)}"}} for r in refs]
    r = requests.post(f"{server_url}/v1/images", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=600)
    return _decoded(r)


def _decoded(r) -> Image.Image:
    """Image from an images-API reply (data[0].b64_json)."""
    if not r.ok:
        raise RuntimeError(f"{r.status_code} from the image model: {r.text[:300]}")
    return Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"]))).convert("RGB")


def _fields(text: str, names) -> dict:
    """NAME: value blocks; a value runs until the next NAME:."""
    pat = "|".join(names)
    return {m[1]: m[2].strip() for m in re.finditer(rf"^\s*({pat})\s*:\s*(.*?)(?=^\s*(?:{pat})\s*:|\Z)", text, re.M | re.S)}


def _tool(name: str, description: str, **fields: str) -> dict:
    props = {k: {"type": "string", "description": d} for k, d in fields.items()}
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props, "required": list(props)}}}


TOOLS = [
    _tool("plan", "Record the attack plan; returns what to fix, or ok.", failure_mode="One of the failure modes",
          attack_target="The wrong option letter (multiple choice) or the wrong answer", source_cue="Visual cue in the source image",
          target_cue="Visual cue the target image must show"),
    _tool("edit", "Edit the source photo with this prompt (needs a plan): the scene, camera and layout stay; the image is shown to you.",
          prompt="Editing prompt: what to change"),
    _tool("generate", "Render a new image from this prompt alone, without the source photo (needs a plan); the image is shown to you.",
          prompt="Image prompt: the whole scene, photorealistic, in the source's camera view"),
    _tool("check", "An independent checker judges whether the last image shows the target cue, with a diagnosis and a revised prompt."),
    _tool("submit", "Finish with the last generated image."),
]


class ScenarioAgent:
    FIELDS = ("failure_mode", "attack_target", "source_cue", "target_cue")

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.key = os.environ.get(config.api_key_env, "dummy")
        self.client = OpenAI(base_url=f"{config.server_url}/v1", api_key=self.key)
        self.checker = self.client if not config.checker_server_url else OpenAI(
            base_url=f"{config.checker_server_url}/v1", api_key=os.environ.get(config.checker_api_key_env or config.api_key_env, "dummy"))

    def _chat(self, messages, tools=None, model=None, client=None):
        client = client or self.client
        return client.chat.completions.create(model=model or self.config.model, messages=messages, temperature=0.0, max_tokens=1200,
                                              extra_body=no_reasoning(client), **({"tools": tools} if tools else {})).choices[0].message

    @staticmethod
    def _user(text: str, img: Image.Image) -> dict:
        return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(img)}"}},
                                            {"type": "text", "text": text}]}

    @staticmethod
    def _invalid(case: Case, plan: dict) -> Optional[str]:
        if missing := [k for k in ScenarioAgent.FIELDS if not str(plan.get(k, "")).strip()]:
            return f"Missing fields: {', '.join(missing)}."
        if case.options:
            letter = plan["attack_target"].strip().strip(".()").upper()[:1]
            if letter not in case.options:
                return f"attack_target must be one of the option letters {', '.join(case.options)}."
            if letter == case.answer.strip().upper()[:1]:
                return "attack_target must be a wrong option, not the correct answer."
            plan["attack_target"] = letter
        elif plan["attack_target"].strip().lower() == case.answer.strip().lower():
            return "attack_target must differ from the correct answer."
        return None

    def _render(self, case: Case, src: Image.Image, prompt: str, seed: int, edit: bool) -> Image.Image:
        """Target image from the local server or OpenRouter (both return data[0].b64_json): the source edited, or a new one from the prompt."""
        c = self.config
        if edit:
            prompt = f"{DOMAIN_CONSTRAINTS.get(case.domain, DOMAIN_CONSTRAINTS['general'])} {prompt}"
        if c.generator == "openrouter":
            return openrouter_image(c.server_url, self.key, c.generator_model, prompt, [src] if edit else [])
        body = {"prompt": prompt, "num_inference_steps": c.num_inference_steps, "seed": seed}
        return _decoded(requests.post(f"{c.generator_url}/v1/images/edits", data=body, timeout=600,
                                      files=[("image", ("source.png", _png(src), "image/png"))]) if edit
                        else requests.post(f"{c.generator_url}/v1/images/generations", json=body, timeout=600))

    def _use(self, name: str, args: dict, case: Case, src: Image.Image, st: dict) -> tuple[str, Optional[Image.Image]]:
        """Run one tool call: (text for the agent, image to show it)."""
        plan = st["plan"]
        if name == "plan":
            if problem := self._invalid(case, args):
                return problem, None
            st["plan"] = args
            return "ok", None
        if name == "submit":
            if not st["img"]:
                return "Make an image first.", None
            if not st["ok"] and st["rounds"] < self.config.max_rounds:
                return "The last image has not passed check. Check it, or generate a better one (images left: %d)." % (self.config.max_rounds - st["rounds"]), None
            st["done"] = True
            return "submitted", None
        if not plan:
            return "Call plan first.", None
        if name in ("edit", "generate"):
            if st["rounds"] >= self.config.max_rounds:
                return f"No more images ({self.config.max_rounds} allowed): submit.", None
            st.update(img=self._render(case, src, args["prompt"], self.config.seed + st["rounds"], name == "edit"), prompt=args["prompt"], tool=name,
                      ok=False, rounds=st["rounds"] + 1)
            return f"Image {st['rounds']} ({name}).", st["img"]
        if name == "check" and st["img"]:
            st["ok"], diagnosis, revised = self._check(case, plan, st)
            st["log"].append({"round": st["rounds"], "tool": st["tool"], "prompt": st["prompt"], "verdict": st["ok"], "diagnosis": diagnosis})
            return f"VERDICT: {'YES' if st['ok'] else 'NO'}\nDIAGNOSIS: {diagnosis}" + (f"\nGENERATION_PROMPT: {revised}" if revised else ""), None
        return "Make an image first." if name == "check" else f"Unknown tool {name}.", None

    def _check(self, case: Case, plan: dict, st: dict) -> tuple[bool, str, str]:
        """(passes, diagnosis, revised prompt); multiple choice: most blind checker models must pick the attack target; open-ended: an informed checker judges."""
        if case.options:
            opts = "\n".join(f"{k}. {v}" for k, v in case.options.items())
            message = self._user(BLIND_CHECK.format(question=case.question, options=opts), st["img"])

            def vote(model):
                try:
                    reply = _fields(clean_llm_text(self._chat([message], model=model, client=self.checker).content), ("ANSWER", "REASON"))
                except Exception:  # a failed call does not vote
                    return None
                return model.split("/")[-1], reply.get("ANSWER", "").strip().strip(".()").upper()[:1], reply.get("REASON", "")

            with ThreadPoolExecutor(len(self.config.checker_models)) as pool:
                votes = [v for v in pool.map(vote, self.config.checker_models) if v]
            if not votes:
                raise RuntimeError("no checker model answered")
            hits = sum(letter == plan["attack_target"] for _, letter, _ in votes)
            picked = "; ".join(f"{name} picked {letter or 'no option'} ({case.options.get(letter, '-')}): {why}" for name, letter, why in votes)
            return hits * 2 > len(votes), f"{hits} of {len(votes)} checkers picked {plan['attack_target']}, seeing only the image. {picked}", ""
        text = self._chat([self._user(CHECK.format(question=case.question, target=self._target_text(case, plan), target_cue=plan["target_cue"],
                                                   prompt=st["prompt"]), st["img"])]).content
        reply = _fields(clean_llm_text(text), ("VERDICT", "DIAGNOSIS", "GENERATION_PROMPT"))
        return reply.get("VERDICT", "").upper().startswith("YES"), reply.get("DIAGNOSIS", ""), reply.get("GENERATION_PROMPT", "")

    @staticmethod
    def _target_text(case: Case, plan: dict) -> str:
        return case.options[plan["attack_target"]] if case.options else plan["attack_target"]

    def run(self, case: Case) -> tuple[TargetSpec, Image.Image]:
        """The agent plans, generates, checks and submits through its tools."""
        src = load_rgb(case.image)
        if case.options:
            opts = "\n".join(f"{k}. {v}" for k, v in case.options.items())
            system, text = PHYSICAL_SYSTEM, (f"Domain: {case.domain}\nTask family: {case.task or 'unknown'}\n"
                                             f"Question: {case.question}\nOptions:\n{opts}\nCorrect answer: {case.answer}")
        else:
            system, text = GENERAL_SYSTEM, f"Question: {case.question}\nCorrect answer: {case.answer}"
            if case.target:
                text += f"\nWrong answer to use: {case.target}"
        messages = [{"role": "system", "content": system + AGENT_TOOLS}, self._user(text, src)]
        st = {"plan": None, "img": None, "prompt": "", "tool": "", "rounds": 0, "ok": False, "done": False, "log": []}
        for _ in range(self.config.max_steps):
            reply = self._chat(messages, TOOLS)
            calls = reply.tool_calls or []
            messages.append({"role": "assistant", "content": reply.content or "", **({"tool_calls": [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in calls]} if calls else {})})
            shown = []
            for t in calls:
                try:
                    out, img = self._use(t.function.name, json.loads(t.function.arguments or "{}"), case, src, st)
                except (ValueError, KeyError) as e:
                    out, img = f"Bad call: {e}", None
                messages.append({"role": "tool", "tool_call_id": t.id, "content": out})
                shown += [img] if img else []
            messages += [self._user("The generated image.", img) for img in shown]
            if st["done"]:
                break
            if not calls:
                messages.append({"role": "user", "content": "Use the tools; call submit when the image is final."})
        if not (st["plan"] and st["img"]):
            raise RuntimeError(f"{case.id}: no plan and image after {self.config.max_steps} steps")
        plan, target = st["plan"], st["plan"]["attack_target"]
        source_text = case.options[case.answer.strip().upper()[:1]] if case.options else case.answer
        return TargetSpec(case.id, plan["failure_mode"], target, self._target_text(case, plan), source_text, plan["source_cue"],
                          plan["target_cue"], st["prompt"], st["rounds"], st["ok"], st["log"]), st["img"]
