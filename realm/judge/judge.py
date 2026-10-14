"""Response-to-label matching."""

import json
import re

from openai import OpenAI

from .prompts import GPTSCORE_TEMPLATE, KMR_TEMPLATE, MAIN_OBJECT_TEMPLATE

JUDGE_THRESHOLD = 0.5  # score >= threshold counts as a match


def no_reasoning(client: OpenAI) -> dict | None:
    return {"reasoning": {"enabled": False}} if "openrouter.ai" in str(client.base_url) else None


def clean_llm_text(text: str) -> str:
    """Drop <think> blocks and one pair of enclosing quotes."""
    text = re.sub(r"<think>.*?(</think>|$)\s*", "", text or "", flags=re.DOTALL).strip()
    return text[1:-1].strip() if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'" else text


def _parse_gptscore(answer: str) -> float:
    m = re.search(r"\d*\.?\d+", answer)
    return min(max(float(m.group()), 0.0), 1.0) if m else 0.0


def _parse_kmr(answer: str) -> float:
    body = answer[answer.find("{"): answer.rfind("}") + 1]
    try:
        matches = json.loads(re.sub(r"(?m)^\s*//.*$", "", body) or "{}")
        no_match = {"", "none", "null", "n/a"}  # judges also write misses as {"kw": null}
        return 1.0 if any(str(v or "").strip().lower() not in no_match for v in matches.values()) else 0.0
    except (json.JSONDecodeError, AttributeError):
        return 0.0


def _parse_yes_no(answer: str) -> float:
    return 1.0 if answer.upper().startswith("YES") else 0.0


# name -> (template, parser returning a score in [0, 1])
JUDGE_PROMPTS = {
    "gptscore": (GPTSCORE_TEMPLATE, _parse_gptscore),
    "kmr": (KMR_TEMPLATE, _parse_kmr),
    "main_object": (MAIN_OBJECT_TEMPLATE, _parse_yes_no),
}
DEFAULT_PROMPT = "main_object"


def substring_match(response: str, target: str) -> bool:
    t = target.lower().strip()
    if not t:
        return False
    return t in response.lower()


def vlm_judge(client: OpenAI, model: str, response: str, target: str,
              prompt: str = DEFAULT_PROMPT, question: str = "") -> tuple[bool, float] | None:
    """(score >= threshold, score), or None if the call fails."""
    template, parse = JUDGE_PROMPTS[prompt]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": template.replace("__QUESTION__", question).replace("__TARGET__", target)
                       .replace("__RESPONSE__", response)}],  # response last: it may contain placeholders
            max_tokens=256, temperature=0.0, extra_body=no_reasoning(client),
        )
        score = parse(clean_llm_text(resp.choices[0].message.content))
        return score >= JUDGE_THRESHOLD, score
    except Exception:
        return None
