"""FigStep typographic injection attack: pairs a clean image with rendered annotation text to make a VLM hallucinate the described object."""

import functools
import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..base_attack import AttackResult, BaseAttack
from ...data.base_dataset import Sample
from ...judge import clean_llm_text, no_reasoning
from .config import FigStepConfig, _INJECTION_PROMPT

_ASSETS = Path(__file__).parent / "assets"


class FigStepAttack(BaseAttack):

    def __init__(self, config: FigStepConfig):
        super().__init__(config)
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            if not (self.config.llm_server_url and self.config.llm_model):
                raise ValueError("FigStep needs llm_server_url and llm_model in its YAML")
            from openai import OpenAI
            self._llm_client = OpenAI(base_url=f"{self.config.llm_server_url}/v1",
                                      api_key=os.environ.get(self.config.llm_api_key_env, "dummy"))
        return self._llm_client

    @functools.lru_cache(maxsize=None)  # same target -> same output
    def _llm_generate_injection(self, target_text: str) -> str:
        client = self._get_llm_client()
        prompt = _INJECTION_PROMPT.replace("{target}", target_text)
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

    def _make_injection_image(self, text: str) -> Image.Image:
        """FigStep's text_to_image; the font shrinks to fit."""
        c = self.config
        (w, h), (x, y) = c.injection_image_size, c.text_origin
        text = textwrap.fill(text, width=c.wrap_width)
        p = _ASSETS / c.font_path  # installed asset, else a system font
        img = Image.new("RGB", (w, h), "#FFFFFF")
        draw = ImageDraw.Draw(img)
        for size in range(c.font_size, 0, -1):
            font = ImageFont.truetype(str(p) if p.is_file() else c.font_path, size)
            _, _, r, b = draw.multiline_textbbox((x, y), text, font=font, spacing=c.line_spacing)
            if r <= w - x and b <= h - y:  # same margins
                break
        draw.multiline_text((x, y), text, fill="#000000", font=font, spacing=c.line_spacing)
        return img

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

        # Reuse an injection the caller already generated
        injection_text = sample.metadata.get("attack_injection_text") or self.config.injection_text
        if not injection_text:
            injection_text = self._llm_generate_injection(target_text)

        injection_image = self._make_injection_image(injection_text)

        return AttackResult(
            adversarial_sample=injection_image,
            perturbation_norm=None,  # not a perturbation
            metadata={
                "target_phrase": target_text,
                "injection_text": injection_text,
                "text_prompt": self.config.text_prompt,
                "attack": "figstep",
            },
        )
