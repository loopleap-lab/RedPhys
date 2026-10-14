"""BlueSuffix defense: diffusion image purifier, text purifier, suffix generator."""

import os
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from ..base_defense import BaseDefense, DefenseConfig, DefenseResult


@dataclass
class BlueSuffixDefenseConfig(DefenseConfig):
    enable_image_purifier: bool = True
    enable_text_purifier: bool = True
    enable_suffix_generator: bool = True

    max_timesteps: str = "400"
    num_denoising_steps: str = "400"
    sampling_method: str = "ddpm"
    diffusion_checkpoint: Optional[str] = None

    suffix_generator_dir: Optional[str] = None

    llm_server_url: Optional[str] = None  # OpenAI-compatible server; None: OpenAI
    llm_api_key_env: str = "OPENAI_API_KEY"
    text_purifier_model: str = "gpt-4o"
    suffix_max_new_tokens: int = 10  # fixed suffix length
    suffix_top_p: float = 0.92
    suffix_temperature: float = 0.7

    device: str = "cuda"

    def __post_init__(self):
        # string booleans from the CLI
        for field_name in ("enable_image_purifier", "enable_text_purifier", "enable_suffix_generator"):
            val = getattr(self, field_name)
            if isinstance(val, str):
                setattr(self, field_name, val.lower() not in ("false", "0", "no"))


class BlueSuffixDefense(BaseDefense):
    def __init__(self, config: BlueSuffixDefenseConfig):
        super().__init__(config)
        self.config: BlueSuffixDefenseConfig = config
        self._purification_forward = None
        self._suffix_model = None
        self._suffix_tokenizer = None

    def _initialize_image_purifier(self):
        if self._purification_forward is not None:
            return

        from .core.load_diffusion_model import load_diffusion_models
        from .core.purification import PurificationForward
        from .core.purify import get_diffusion_params

        if self.config.diffusion_checkpoint is None:
            from .config import get_diffusion_checkpoint_path
            self.config.diffusion_checkpoint = get_diffusion_checkpoint_path()

        print("Loading BlueSuffix image purifier (diffusion denoising)...")

        diffusion, gaussian_diffusion = load_diffusion_models(
            self.config.diffusion_checkpoint,
            torch.device(self.config.device),
        )

        def_max_timesteps, def_diffusion_steps = get_diffusion_params(
            self.config.max_timesteps,
            self.config.num_denoising_steps,
        )

        print(f"  Max timesteps: {def_max_timesteps}")
        print(f"  Diffusion steps: {def_diffusion_steps}")
        print(f"  Sampling method: {self.config.sampling_method}")

        self._purification_forward = PurificationForward(
            diffusion,
            def_max_timesteps,
            def_diffusion_steps,
            self.config.sampling_method,
            is_imagenet=True,
            device=torch.device(self.config.device),
            gaussian_diffusion=gaussian_diffusion,
        )

        print("BlueSuffix image purifier loaded")

    def _initialize_suffix_generator(self):
        if self._suffix_model is not None:
            return

        from .core.suffix_generator import load_suffix_generator
        from .config import get_suffix_generator_path

        suffix_dir = self.config.suffix_generator_dir
        if suffix_dir is None:
            suffix_dir = get_suffix_generator_path()

        print("Loading BlueSuffix suffix generator...")
        self._suffix_model, self._suffix_tokenizer = load_suffix_generator(
            suffix_dir, device=self.config.device,
        )
        print("BlueSuffix suffix generator loaded")

    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        prompt = kwargs.get("prompt")
        steps_applied = []
        self._set_seed(self.config.seed)  # per sample
        api_key = os.environ.get(self.config.llm_api_key_env, "dummy" if self.config.llm_server_url else None)
        if self.config.enable_text_purifier and prompt is not None and not api_key:
            raise ValueError(f"BlueSuffix's text purifier needs {self.config.llm_api_key_env} (or enable_text_purifier: false)")

        purified_pil = None
        if self.config.enable_image_purifier:
            self._initialize_image_purifier()

            image = Image.open(image_path).convert("RGB")
            transform = transforms.ToTensor()

            with torch.no_grad():
                x = transform(image).unsqueeze(0).to(torch.float32).to(self.config.device)
                purified_tensor = self._purification_forward(x)

                vis = purified_tensor.permute(0, 2, 3, 1)
                vis = torch.clamp(vis, 0, 1)

                purified_np = vis[0].cpu().numpy()
                purified_pil = Image.fromarray((purified_np * 255).astype("uint8"))

            steps_applied.append("image_purifier")
        else:
            purified_pil = Image.open(image_path).convert("RGB")

        purified_prompt = prompt
        text_purifier_error = None
        if self.config.enable_text_purifier and prompt is not None:
            from .core.text_purifier import purify_text
            try:
                url = self.config.llm_server_url
                purified_prompt = purify_text(prompt, api_key, model=self.config.text_purifier_model,
                                              base_url=f"{url}/v1" if url else None)
                steps_applied.append("text_purifier")
            except Exception as e:
                print(f"[BlueSuffix] Text purifier failed, using original prompt: {e}")
                text_purifier_error = str(e)

        suffix = ""
        if self.config.enable_suffix_generator and prompt is not None:
            self._initialize_suffix_generator()
            from .core.suffix_generator import generate_suffix
            target_prompt = purified_prompt if purified_prompt is not None else prompt
            suffix = generate_suffix(
                self._suffix_model, self._suffix_tokenizer,
                target_prompt, device=self.config.device, max_new_tokens=self.config.suffix_max_new_tokens,
                top_p=self.config.suffix_top_p, temperature=self.config.suffix_temperature,
            )
            steps_applied.append("suffix_generator")

        final_prompt = prompt
        if prompt is not None:
            base = purified_prompt if purified_prompt is not None else prompt
            final_prompt = f"{base.rstrip('?')} {suffix.strip()}?" if suffix else base  # as trained

        return DefenseResult(
            cleaned_sample=purified_pil,
            original_image_path=image_path,
            detection_confidence=0.0,
            regions_removed=0,
            metadata={
                "original_prompt": prompt,
                "purified_prompt": purified_prompt,
                "suffix": suffix,
                "final_prompt": final_prompt,
                "steps_applied": steps_applied,
                "enable_image_purifier": self.config.enable_image_purifier,
                "enable_text_purifier": self.config.enable_text_purifier,
                "enable_suffix_generator": self.config.enable_suffix_generator,
                "max_timesteps": self.config.max_timesteps,
                "num_denoising_steps": self.config.num_denoising_steps,
                "sampling_method": self.config.sampling_method,
                "text_purifier_error": text_purifier_error,
                "purification_method": "bluesuffix",
            },
        )
