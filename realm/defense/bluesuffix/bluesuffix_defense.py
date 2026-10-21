"""BlueSuffix defense: diffusion image purifier, text purifier, suffix generator."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ...data import load_rgb
from ...data.tensors import to_tensor
from ..base_defense import BaseDefense, DefenseConfig, DefenseResult

ASSETS = Path(__file__).parent / "assets"


@dataclass
class BlueSuffixDefenseConfig(DefenseConfig):
    enable_image_purifier: bool = True
    enable_text_purifier: bool = True
    enable_suffix_generator: bool = True

    max_timesteps: str = "400"
    num_denoising_steps: str = "400"
    sampling_method: str = "ddpm"
    diffusion_checkpoint: Optional[str] = None  # default: FreqPure's guided-diffusion weights
    suffix_generator_dir: Optional[str] = None  # default: assets/suffix_generator

    llm_server_url: Optional[str] = None  # OpenAI-compatible server; None: OpenAI
    llm_api_key_env: str = "OPENAI_API_KEY"
    text_purifier_model: str = "gpt-4o"
    suffix_max_new_tokens: int = 10  # fixed suffix length
    suffix_top_p: float = 0.92
    suffix_temperature: float = 0.7


class BlueSuffixDefense(BaseDefense):
    def __init__(self, config: BlueSuffixDefenseConfig):
        super().__init__(config)
        self._purification_forward = None
        self._suffix_model = None
        self._suffix_tokenizer = None

    def _initialize_image_purifier(self):
        if self._purification_forward is not None:
            return

        from .._diffusion import get_diffusion_params, load_diffusion
        from .core.purification import PurificationForward

        print("Loading BlueSuffix image purifier (diffusion denoising)...")
        c, device = self.config, torch.device(self.config.device)
        model, gaussian_diffusion = load_diffusion(c.diffusion_checkpoint, device, use_fp16=True)  # as DiffPure
        max_timestep, steps = get_diffusion_params(c.max_timesteps, c.num_denoising_steps)
        self._purification_forward = PurificationForward(model, max_timestep, steps, c.sampling_method, device, gaussian_diffusion)

    def _initialize_suffix_generator(self):
        if self._suffix_model is not None:
            return

        from .core.suffix_generator import load_suffix_generator

        suffix_dir = Path(self.config.suffix_generator_dir or ASSETS / "suffix_generator")
        if not suffix_dir.exists():
            raise FileNotFoundError(f"Suffix generator not found at {suffix_dir}: run install.sh")
        print("Loading BlueSuffix suffix generator...")
        self._suffix_model, self._suffix_tokenizer = load_suffix_generator(str(suffix_dir), device=self.config.device)

    def clean(self, image_path: str, prompt: Optional[str] = None, **kwargs) -> DefenseResult:
        steps_applied = []
        api_key = os.environ.get(self.config.llm_api_key_env, "dummy" if self.config.llm_server_url else None)
        if self.config.enable_text_purifier and prompt is not None and not api_key:
            raise ValueError(f"BlueSuffix's text purifier needs {self.config.llm_api_key_env} (or enable_text_purifier: false)")

        image = purified_pil = load_rgb(image_path)
        if self.config.enable_image_purifier:
            self._initialize_image_purifier()
            with torch.no_grad():
                purified = self._purification_forward(to_tensor(image).to(self.config.device))
            if not torch.isfinite(purified).all():  # as to_pil: run.py retries once with another seed
                raise FloatingPointError("non-finite purified image")
            purified_pil = Image.fromarray((purified[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8"))  # truncated

            steps_applied.append("image_purifier")

        purified_prompt = prompt
        text_purifier_error = None
        if self.config.enable_text_purifier and prompt is not None:
            import openai
            from .core.text_purifier import purify_text
            try:
                url = self.config.llm_server_url
                purified_prompt = purify_text(prompt, api_key, model=self.config.text_purifier_model,
                                              base_url=f"{url}/v1" if url else None)
                steps_applied.append("text_purifier")
            except openai.BadRequestError as e:  # e.g. a content filter; outages and auth errors fail the image (retried on resume)
                print(f"[BlueSuffix] Text purifier refused, using original prompt: {e}")
                text_purifier_error = str(e)

        suffix = ""
        if self.config.enable_suffix_generator and prompt is not None:
            self._initialize_suffix_generator()
            from .core.suffix_generator import generate_suffix
            suffix = generate_suffix(
                self._suffix_model, self._suffix_tokenizer,
                purified_prompt, device=self.config.device, max_new_tokens=self.config.suffix_max_new_tokens,
                top_p=self.config.suffix_top_p, temperature=self.config.suffix_temperature,
            )
            steps_applied.append("suffix_generator")

        final_prompt = f"{purified_prompt.rstrip('?')} {suffix}?" if suffix else purified_prompt  # as trained; suffix is stripped

        return DefenseResult(cleaned_sample=purified_pil, metadata={
            "original_prompt": prompt, "purified_prompt": purified_prompt, "suffix": suffix, "final_prompt": final_prompt,
            "steps_applied": steps_applied, "text_purifier_error": text_purifier_error})
