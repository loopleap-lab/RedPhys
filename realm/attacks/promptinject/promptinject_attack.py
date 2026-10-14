"""Text injection written by an LLM."""

import functools
import os

from ..base_attack import AttackResult, BaseAttack
from ...data.base_dataset import Sample
from ...judge import clean_llm_text, no_reasoning
from .config import (
    PromptInjectConfig,
    _INJECTION_PROMPT_INSTRUCTION,
    _INJECTION_PROMPT_AUTHORITY,
)


class PromptInjectAttack(BaseAttack):

    def __init__(self, config: PromptInjectConfig):
        super().__init__(config)
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            if not (self.config.llm_server_url and self.config.llm_model):
                raise ValueError("PromptInject needs llm_server_url and llm_model in its YAML")
            from openai import OpenAI
            self._llm_client = OpenAI(base_url=f"{self.config.llm_server_url}/v1",
                                      api_key=os.environ.get(self.config.llm_api_key_env, "dummy"))
        return self._llm_client

    @functools.lru_cache(maxsize=None)  # same target -> same output
    def _llm_generate_injection(self, target_text: str,
                                question: str = "") -> str:
        client = self._get_llm_client()
        level = self.config.level
        if level == "authority":
            prompt_tpl = _INJECTION_PROMPT_AUTHORITY
        else:
            prompt_tpl = _INJECTION_PROMPT_INSTRUCTION
        prompt = (prompt_tpl
                  .replace("{target}", target_text)
                  .replace("{question}", question or ""))
        resp = client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system",
                 "content": "You are a concise assistant. /no_think"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256, temperature=0.0, extra_body=no_reasoning(client),
        )
        text = clean_llm_text(resp.choices[0].message.content)
        if not text:
            raise RuntimeError(
                f"LLM returned empty injection text for target '{target_text}'"
            )
        return text

    def _resolve_target(self, sample: Sample) -> str:
        """Get target_text from config or sample metadata."""
        if self.config.target_text:
            return self.config.target_text
        target = sample.metadata.get("attack_target_text")
        if target:
            return target
        raise ValueError(
            f"No target_text: set target_text or provide "
            f"metadata['attack_target_text'] on sample '{sample.id}'."
        )

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        target_text = self._resolve_target(sample)

        # Reuse pre-computed injection from metadata if available.
        injection = sample.metadata.get("attack_injection_text") or self.config.injection_text
        if not injection:
            injection = self._llm_generate_injection(
                target_text, question=sample.question,
            )

        adversarial_question = sample.question.rstrip() + " " + injection

        return AttackResult(
            adversarial_sample=sample.images[0],
            perturbation_norm=0.0,
            metadata={
                "target_phrase": target_text,
                "level": self.config.level,
                "injection_text": injection,
                "adversarial_question": adversarial_question,
                "attack": "promptinject",
            },
        )
