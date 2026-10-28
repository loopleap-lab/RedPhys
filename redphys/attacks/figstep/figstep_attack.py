"""FigStep: the clean image plus rendered text naming the target."""

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..base_attack import AttackResult, BaseAttack, LLMClientMixin, target_text
from ...data import Sample
from .config import FigStepConfig, _INJECTION_PROMPT

_ASSETS = Path(__file__).parent / "assets"


class FigStepAttack(LLMClientMixin, BaseAttack):

    def __init__(self, config: FigStepConfig):
        super().__init__(config)
        self._injections = {}  # same target -> same text

    def _llm_generate_injection(self, target: str) -> str:
        if target not in self._injections:
            self._injections[target] = self.llm_text(_INJECTION_PROMPT.replace("{target}", target))
        return self._injections[target]

    def _make_injection_image(self, text: str) -> Image.Image:
        """FigStep's text_to_image; the font shrinks to fit."""
        c = self.config
        (w, h), (x, y) = c.injection_image_size, c.text_origin
        text = textwrap.fill(text, width=c.wrap_width)
        p = _ASSETS / c.font_path
        font_path = str(p) if p.is_file() else c.font_path  # installed asset, else a system font
        img = Image.new("RGB", (w, h), "#FFFFFF")
        draw = ImageDraw.Draw(img)
        for size in range(c.font_size, 0, -1):
            font = ImageFont.truetype(font_path, size)
            _, _, r, b = draw.multiline_textbbox((x, y), text, font=font, spacing=c.line_spacing)
            if r <= w - x and b <= h - y:  # same margins
                break
        draw.multiline_text((x, y), text, fill="#000000", font=font, spacing=c.line_spacing)
        return img

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        target = target_text(self.config, sample)
        injection_text = self.config.injection_text or self._llm_generate_injection(target)  # the YAML's, else LLM-written
        injection_image = self._make_injection_image(injection_text)

        return AttackResult(
            adversarial_sample=injection_image,
            perturbation_norm=None,  # not a perturbation
            metadata={
                "injection_text": injection_text,
                "text_prompt": self.config.text_prompt,
                "attack": "figstep",  # read by the evaluation (defended manifests rename the attack)
            },
        )
