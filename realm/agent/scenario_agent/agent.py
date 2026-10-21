"""Scenario agent (paper Sec. 3.4): plan, edit, generate, check, submit."""

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

from ...data import image_bytes, image_data_url, load_rgb
from ...judge import clean_llm_text, no_reasoning
from .prompts import AGENT_TOOLS, BLIND_CHECK, CHECK, DOMAIN_CONSTRAINTS, GENERAL_SYSTEM, PHYSICAL_SYSTEM, SALIENT_DOMAIN, SALIENT_TOOLS


@dataclass
class ScenarioConfig:
    model: str = "openai/gpt-6-luna"
    server_url: str = "https://openrouter.ai/api"
    api_key_env: str = "OPENROUTER_API_KEY"
    mode: str = "scene"  # or "salient"
    generator: str = "openrouter"  # or "local"
    generator_url: str = "http://localhost:8091"
    generator_model: str = "qwen/qwen-image-3"
    checker_models: tuple = ("anthropic/claude-haiku-5.5", "meta-llama/llama-4-maverick", "mistralai/mistral-medium-3.1")
    checker_server_url: str = ""  # "" = server_url
    checker_api_key_env: str = ""  # "" = api_key_env
    max_rounds: int = 5  # images
    max_steps: int = 24  # LLM turns
    num_inference_steps: int = 40
    seed: int = 0

    def __post_init__(self):
        for k, ok in (("mode", ("scene", "salient")), ("generator", ("local", "openrouter"))):
            if getattr(self, k) not in ok:
                raise ValueError(f"{k}: {'|'.join(ok)}; got {getattr(self, k)!r}")


@dataclass
class Case:
    id: str
    image: str
    question: str
    answer: str  # letter or text
    options: Optional[dict] = None  # multiple choice
    domain: str = "general"
    task: str = ""
    target: Optional[str] = None  # given wrong answer


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


def openrouter_image(server_url: str, key: str, model: str, prompt: str, refs=()) -> Image.Image:
    """OpenRouter image; refs: images to edit."""
    body = {"model": model, "prompt": prompt}
    if refs:
        body["input_references"] = [{"type": "image_url", "image_url": {"url": image_data_url(r)}} for r in refs]
    r = requests.post(f"{server_url}/v1/images", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=600)
    return _decoded(r)


def _decoded(r) -> Image.Image:
    """Image from an images-API reply."""
    if not r.ok:
        raise RuntimeError(f"{r.status_code} from the image model: {r.text[:300]}")
    return Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"]))).convert("RGB")


def _fields(text: str, names) -> dict:
    """NAME: value blocks (any case, markdown bold ignored)."""
    pat, text = "|".join(names), text.replace("*", "")
    return {m[1].upper(): m[2].strip()
            for m in re.finditer(rf"^\s*({pat})\s*:\s*(.*?)(?=^\s*(?:{pat})\s*:|\Z)", text, re.M | re.S | re.I)}


def _letter(s: str, allowed: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str:
    """The first standalone letter of `allowed` in s ("B", "(B)", "Option B", "I think b."), else ""."""
    m = re.search(rf"\b([{allowed}])\b", s.upper())
    return m[1] if m else ""


def _options(case) -> str:
    return "\n".join(f"{k}. {v}" for k, v in case.options.items())


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
TOOLS_SALIENT = [t for t in TOOLS if t["function"]["name"] not in ("edit", "generate")] + [
    _tool("generate", "Render the picture from this prompt alone (needs a plan); the image is shown to you.",
          prompt="Image prompt, per the picture rules")]


class ScenarioAgent:
    FIELDS = ("failure_mode", "attack_target", "source_cue", "target_cue")

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.salient = config.mode == "salient"
        self.key = os.environ.get(config.api_key_env, "dummy")
        self.client = OpenAI(base_url=f"{config.server_url}/v1", api_key=self.key)
        self.checker = self.client if not (config.checker_server_url or config.checker_api_key_env) else OpenAI(
            base_url=f"{config.checker_server_url or config.server_url}/v1",
            api_key=os.environ.get(config.checker_api_key_env or config.api_key_env, "dummy"))

    def _chat(self, messages, tools=None, model=None, client=None):
        client = client or self.client
        return client.chat.completions.create(model=model or self.config.model, messages=messages, temperature=0.0, max_tokens=1200,
                                              extra_body=no_reasoning(client), **({"tools": tools} if tools else {})).choices[0].message

    @staticmethod
    def _user(text: str, url: str) -> dict:
        return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                            {"type": "text", "text": text}]}

    @staticmethod
    def _invalid(case: Case, plan: dict) -> Optional[str]:
        if missing := [k for k in ScenarioAgent.FIELDS if not str(plan.get(k, "")).strip()]:
            return f"Missing fields: {', '.join(missing)}."
        if case.options:
            letter = _letter(plan["attack_target"], "".join(case.options))
            if letter not in case.options:
                return f"attack_target must be one of the option letters {', '.join(case.options)}."
            if letter == _letter(case.answer):
                return "attack_target must be a wrong option, not the correct answer."
            plan["attack_target"] = letter
        elif plan["attack_target"].strip().lower() == case.answer.strip().lower():
            return "attack_target must differ from the correct answer."
        if case.target:  # given wrong answer
            given = _letter(case.target, "".join(case.options)) if case.options else case.target.strip().lower()
            if (plan["attack_target"] if case.options else plan["attack_target"].strip().lower()) != given:
                return f"attack_target must be the given wrong answer: {case.target}."
        return None

    def _render(self, case: Case, src: Image.Image, prompt: str, seed: int, edit: bool) -> Image.Image:
        """Edited source or new image."""
        c = self.config
        if edit:
            prompt = f"{DOMAIN_CONSTRAINTS.get(case.domain, DOMAIN_CONSTRAINTS['general'])} {prompt}"
        if c.generator == "openrouter":
            return openrouter_image(c.server_url, self.key, c.generator_model, prompt, [src] if edit else [])
        body = {"prompt": prompt, "num_inference_steps": c.num_inference_steps, "seed": seed}
        return _decoded(requests.post(f"{c.generator_url}/v1/images/edits", data=body, timeout=600,
                                      files=[("image", ("source.png", image_bytes(src), "image/png"))]) if edit
                        else requests.post(f"{c.generator_url}/v1/images/generations", json=body, timeout=600))

    def _use(self, name: str, args: dict, case: Case, src: Image.Image, st: dict) -> tuple[str, Optional[str]]:
        """Tool call -> (reply, data URL of the image to show)."""
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
                left = self.config.max_rounds - st["rounds"]
                return f"The last image has not passed check. Check it, or generate a better one (images left: {left}).", None
            st["done"] = True
            return "submitted", None
        if not plan:
            return "Call plan first.", None
        if name == "edit" and self.salient:
            return "No edit in salient mode: use generate.", None
        if name in ("edit", "generate"):
            if st["rounds"] >= self.config.max_rounds:
                return f"No more images ({self.config.max_rounds} allowed): submit.", None
            img = self._render(case, src, args["prompt"], self.config.seed + st["rounds"], name == "edit")
            st.update(img=img, url=image_data_url(img), prompt=args["prompt"], tool=name, ok=False, rounds=st["rounds"] + 1)  # encoded once
            return f"Image {st['rounds']} ({name}).", st["url"]
        if name == "check" and st["img"]:
            st["ok"], diagnosis, revised = self._check(case, plan, st)
            st["log"].append({"round": st["rounds"], "tool": st["tool"], "prompt": st["prompt"], "verdict": st["ok"], "diagnosis": diagnosis})
            return f"VERDICT: {'YES' if st['ok'] else 'NO'}\nDIAGNOSIS: {diagnosis}" + (f"\nGENERATION_PROMPT: {revised}" if revised else ""), None
        return ("Make an image first." if name == "check" else f"Unknown tool {name}."), None

    def _check(self, case: Case, plan: dict, st: dict) -> tuple[bool, str, str]:
        """(ok, diagnosis, revised prompt): most checkers pick the target; salient: none the answer."""
        if case.options:
            message = self._user(BLIND_CHECK.format(question=case.question, options=_options(case)), st["url"])

            def vote(model):
                try:
                    reply = _fields(clean_llm_text(self._chat([message], model=model, client=self.checker).content), ("ANSWER", "REASON"))
                except Exception:
                    return None
                return model.split("/")[-1], _letter(reply.get("ANSWER", ""), "".join(case.options)), reply.get("REASON", "")

            with ThreadPoolExecutor(len(self.config.checker_models)) as pool:
                votes = [v for v in pool.map(vote, self.config.checker_models) if v]
            if not votes:
                raise RuntimeError("no checker model answered")
            hits = sum(letter == plan["attack_target"] for _, letter, _ in votes)
            picked = "; ".join(f"{name} picked {letter or 'no option'} ({case.options.get(letter, '-')}): {why}" for name, letter, why in votes)
            answered = self.salient and any(letter == _letter(case.answer) for _, letter, _ in votes)
            ok = hits * 2 > len(self.config.checker_models) and not answered  # a majority of all checkers, not of those that answered
            return ok, f"{hits} of {len(votes)} checkers picked {plan['attack_target']}, seeing only the image. {picked}", ""
        text = self._chat([self._user(CHECK.format(question=case.question, target=self._target_text(case, plan), target_cue=plan["target_cue"],
                                                   prompt=st["prompt"]), st["url"])]).content
        reply = _fields(clean_llm_text(text), ("VERDICT", "DIAGNOSIS", "GENERATION_PROMPT"))
        return reply.get("VERDICT", "").upper().startswith("YES"), reply.get("DIAGNOSIS", ""), reply.get("GENERATION_PROMPT", "")

    @staticmethod
    def _target_text(case: Case, plan: dict) -> str:
        return case.options[plan["attack_target"]] if case.options else plan["attack_target"]

    def run(self, case: Case) -> tuple[TargetSpec, Image.Image]:
        """Tool loop for one case."""
        src = load_rgb(case.image)
        if case.options:
            system, text = PHYSICAL_SYSTEM, (f"Domain: {case.domain}\nTask family: {case.task or 'unknown'}\n"
                                             f"Question: {case.question}\nOptions:\n{_options(case)}\nCorrect answer: {case.answer}")
        else:
            system, text = GENERAL_SYSTEM, f"Question: {case.question}\nCorrect answer: {case.answer}"
        if case.target:
            text += f"\nWrong answer to use: {case.target}"
        tools = (SALIENT_TOOLS + "\n" + SALIENT_DOMAIN.get(case.domain, "")).rstrip() if self.salient else AGENT_TOOLS
        messages = [{"role": "system", "content": system + tools}, self._user(text, image_data_url(src))]
        st = {"plan": None, "img": None, "url": None, "prompt": "", "tool": "", "rounds": 0, "ok": False, "done": False, "log": []}
        for _ in range(self.config.max_steps):
            reply = self._chat(messages, TOOLS_SALIENT if self.salient else TOOLS)
            calls = reply.tool_calls or []
            tool_calls = [{"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in calls]
            messages.append({"role": "assistant", "content": reply.content or "", **({"tool_calls": tool_calls} if calls else {})})
            shown = []
            for t in calls:
                try:
                    out, url = self._use(t.function.name, json.loads(t.function.arguments or "{}"), case, src, st)
                except Exception as e:  # a failed image/checker call must not end the case
                    out, url = f"Tool error: {e}", None
                messages.append({"role": "tool", "tool_call_id": t.id, "content": out})
                shown += [url] if url else []
            messages += [self._user("The generated image.", url) for url in shown]
            if st["done"]:
                break
            if not calls:
                messages.append({"role": "user", "content": "Use the tools; call submit when the image is final."})
        if not (st["plan"] and st["img"]):
            raise RuntimeError(f"{case.id}: no plan and image after {self.config.max_steps} steps")
        plan = st["plan"]
        source_text = case.options[_letter(case.answer)] if case.options else case.answer
        return TargetSpec(case.id, plan["failure_mode"], plan["attack_target"], self._target_text(case, plan), source_text, plan["source_cue"],
                          plan["target_cue"], st["prompt"], st["rounds"], st["ok"], st["log"]), st["img"]
