"""Leading question with a false premise toward the target (MMR/MMVU, CVPR 2025); the image and options are unchanged."""

from ..base_attack import AttackResult, BaseAttack, LLMClientMixin, target_text
from ...data import Sample, image_data_url
from ...judge import clean_llm_text, no_reasoning
from .config import PROMPT, LeadingQConfig


class LeadingQAttack(LLMClientMixin, BaseAttack):

    def __init__(self, config: LeadingQConfig):
        super().__init__(config)
        self.llm()  # fails here without llm_model, a key or a server

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        target = target_text(self.config, sample)
        stem, sep, options = sample.question.partition("\nA. ")
        options = sep.lstrip("\n") + options
        content = [{"type": "image_url", "image_url": {"url": image_data_url(sample.images[0])}},
                   {"type": "text", "text": PROMPT.format(target=target, question=stem, options=options)}]
        client = self.llm()
        out = client.chat.completions.create(model=self.config.llm_model, messages=[{"role": "user", "content": content}],
                                             temperature=self.config.temperature, max_tokens=200, extra_body=no_reasoning(client))
        leading = clean_llm_text(out.choices[0].message.content)
        if not leading:
            raise RuntimeError(f"empty leading question for sample {sample.id}")
        question = leading + ("\n" + options if options else "")
        return AttackResult(adversarial_sample=sample.images[0], perturbation_norm=0.0,
                            metadata={"leading_question": leading,
                                      "adversarial_question": question})
