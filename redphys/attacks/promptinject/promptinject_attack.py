"""Text injection written by an LLM."""

from ..base_attack import AttackResult, BaseAttack, LLMClientMixin, target_text
from ...data import Sample
from .config import INJECTION_PROMPTS, PromptInjectConfig


class PromptInjectAttack(LLMClientMixin, BaseAttack):

    def __init__(self, config: PromptInjectConfig):
        super().__init__(config)
        self._injections = {}  # same (target, question) -> same text

    def _llm_generate_injection(self, target: str, question: str) -> str:
        key = target, question
        if key not in self._injections:
            tpl = INJECTION_PROMPTS[self.config.level]
            self._injections[key] = self.llm_text(tpl.replace("{target}", target).replace("{question}", question))
        return self._injections[key]

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        target = target_text(self.config, sample)
        injection = self.config.injection_text or self._llm_generate_injection(target, sample.question)  # the YAML's, else LLM-written
        adversarial_question = sample.question.rstrip() + " " + injection

        return AttackResult(
            adversarial_sample=sample.images[0],
            perturbation_norm=0.0,
            metadata={
                "injection_text": injection,
                "adversarial_question": adversarial_question,
            },
        )
