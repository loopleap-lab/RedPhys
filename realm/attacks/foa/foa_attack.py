"""FOA-Attack: optimal-transport loss over a CLIP ensemble."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
import re
import torch
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..base_attack import BaseAttack, LLMClientMixin, AttackConfig, AttackResult, find_by_stem, load_surrogates, pixel_linf, random_crops, resize_center_crop, to_pil, to_tensor
from ...data import Sample, file_data_url, image_data_url, load_rgb

@dataclass
class FOAAttackConfig(AttackConfig):
    epsilon: float = 16.0
    max_iterations: int = 300
    alpha: float = 1.0

    attack_method: str = "fgsm"         # "fgsm" (paper default), "mifgsm", "pgd" (Adam)

    backbone: List[str] = field(
        default_factory=lambda: ["B16", "B32", "Laion"]
    )

    cluster_number: int = 3
    eta: float = 0.2  # Eq. 10
    temperature: float = 1.0  # Eq. 13
    sinkhorn_lambda: float = 0.1  # Eq. 7-8
    use_adaptive_cluster: bool = True   # auto-escalate 3→5 on failure

    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: Tuple[float, float] = (0.5, 0.9)
    crop_ratio: Tuple[float, float] = (3 / 4, 4 / 3)  # x image W/H
    momentum: float = 0.9  # MI-FGSM

    llm_description_model: str = "gpt-4o-mini"
    llm_scorer_model: str = "gpt-4o-mini"
    llm_similarity_threshold: float = 0.5
    adaptive_clusters: Tuple[int, int] = (3, 5)
    llm_server_url: Optional[str] = None  # OpenAI-compatible server; None: OpenAI
    llm_api_key_env: str = "OPENAI_API_KEY"

    target_images_dir: Optional[str] = None

    input_res: int = 224

    def __post_init__(self):
        super().__post_init__()
        if self.attack_method not in ("fgsm", "mifgsm", "pgd"):
            raise ValueError(f"attack_method must be 'fgsm', 'mifgsm' or 'pgd', got {self.attack_method!r}")


class FOAAttack(LLMClientMixin, BaseAttack):
    config: FOAAttackConfig

    def __init__(self, config: FOAAttackConfig):
        super().__init__(config)
        if config.use_adaptive_cluster and not config.llm_server_url and not os.environ.get(config.llm_api_key_env):
            raise RuntimeError(f"FOA's cluster check calls an LLM: set {config.llm_api_key_env}, or llm_server_url in configs/attacks/foa.yaml, "
                               "or use_adaptive_cluster: false")

        self._model_cache = {}
        self._desc_cache = {}
        self._models = None

    def _ensemble(self, cluster_number: int):
        """(extractor, loss) for cluster_number, built once; the surrogates are shared."""
        if cluster_number not in self._model_cache:
            from .surrogates.FeatureExtractors.Base import EnsembleFeatureExtractor_ot, EnsembleFeatureLoss_OT_foa_attack
            from .surrogates.FeatureExtractors.ClipB16 import ClipB16FeatureExtractor
            from .surrogates.FeatureExtractors.ClipB32 import ClipB32FeatureExtractor
            from .surrogates.FeatureExtractors.ClipL336 import ClipL336FeatureExtractor
            from .surrogates.FeatureExtractors.ClipLaion import ClipLaionFeatureExtractor

            if self._models is None:
                print("Loading CLIP ensemble models...")
                self._models = load_surrogates({"B16": ClipB16FeatureExtractor, "B32": ClipB32FeatureExtractor, "L336": ClipL336FeatureExtractor,
                                                "Laion": ClipLaionFeatureExtractor}, self.config.backbone, self.config.device)
            c = self.config
            self._model_cache[cluster_number] = (
                EnsembleFeatureExtractor_ot(self._models, cluster_number=cluster_number),
                EnsembleFeatureLoss_OT_foa_attack(self._models, cluster_number=cluster_number, eta=c.eta, temperature=c.temperature,
                                                  sinkhorn_lambda=c.sinkhorn_lambda))
        return self._model_cache[cluster_number]

    def _load_target_image(self, sample: Sample) -> Tuple[torch.Tensor, str]:
        """(target tensor, path): the target image with the sample's stem."""
        target_path = find_by_stem(self.config.target_images_dir, str(sample.id), required=True)
        return to_tensor(resize_center_crop(load_rgb(target_path), self.config.input_res), 255).to(self.config.device), str(target_path)

    def _run_single_attack(
        self,
        clean_image: torch.Tensor,
        target_image: torch.Tensor,
        source_crop,
        target_crop,
        cluster_number: int,
    ) -> torch.Tensor:
        ensemble_extractor, ensemble_loss = self._ensemble(cluster_number)
        ensemble_loss.previous_loss_list = []  # fresh dynamic weights per sample

        from .core import attacks
        attack_fn = getattr(attacks, f"{self.config.attack_method}_attack")  # fgsm, mifgsm or pgd

        adv_image = attack_fn(
            image_tensor=clean_image,
            tgt_tensor=target_image,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,  # crops off: random_crops gives identities
            num_iters=self.config.max_iterations,
            epsilon=self.config.epsilon,
            alpha=self.config.alpha,
            **({"momentum_decay": self.config.momentum} if self.config.attack_method == "mifgsm" else {}),
        )

        return adv_image

    def _describe(self, url: str, cache_key: Optional[str] = None) -> str:
        """One-sentence LLM description of an image data URL (cached by cache_key: target images)."""
        if cache_key in self._desc_cache:
            return self._desc_cache[cache_key]

        client = self.llm()
        response = client.chat.completions.create(
            model=self.config.llm_description_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one concise sentence, no longer than 20 words."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }],
            max_completion_tokens=100,
        )
        desc = (response.choices[0].message.content or "").strip()
        if cache_key is not None:
            self._desc_cache[cache_key] = desc
        return desc

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
    def _llm_similarity(self, target_path: str, adv: Image.Image) -> float:
        """LLM-rated similarity in [0, 1] (the official prompt)."""
        client = self.llm()
        target_desc = self._describe(file_data_url(target_path), cache_key=target_path)
        adv_desc = self._describe(image_data_url(adv))  # PNG bytes, as the saved file

        prompt = f"""Rate the semantic similarity between the following two texts on a scale from 0 to 1.
        
                    **Criteria for similarity measurement:**
                    1. **Main Subject Consistency:** If both descriptions refer to the same key subject or object (e.g., a person, food, an event), they should receive a higher similarity score.
                    2. **Relevant Description**: If the descriptions are related to the same context or topic, they should also contribute to a higher similarity score.
                    3. **Ignore Fine-Grained Details:** Do not penalize differences in **phrasing, sentence structure, or minor variations in detail**. Focus on **whether both descriptions fundamentally describe the same thing.**
                    4. **Partial Matches:** If one description contains extra information but does not contradict the other, they should still have a high similarity score.
                    5. **Similarity Score Range:** 
                        - **1.0**: Nearly identical in meaning.
                        - **0.8-0.9**: Same subject, with highly related descriptions.
                        - **0.7-0.8**: Same subject, core meaning aligned, even if some details differ.
                        - **0.5-0.7**: Same subject but different perspectives or missing details.
                        - **0.3-0.5**: Related but not highly similar (same general theme but different descriptions).
                        - **0.0-0.2**: Completely different subjects or unrelated meanings.
                        
                    Text 1: {target_desc}
                    Text 2: {adv_desc}

                Output only a single number between 0 and 1. Do not include any explanation or additional text."""

        response = client.chat.completions.create(
            model=self.config.llm_scorer_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=100,
        )
        score_text = (response.choices[0].message.content or "").strip()
        match = re.search(r"\d+\.?\d*", score_text)  # the LLM may add text around the number
        if not match:
            print(f"Warning: could not parse a score from {score_text!r}; using 0.0")
            return 0.0
        return min(1.0, max(0.0, float(match.group())))

    def _run_adaptive_attack(
        self,
        clean_image: torch.Tensor,
        target_image: torch.Tensor,
        source_crop,
        target_crop,
        target_path: str,
    ) -> Tuple[torch.Tensor, int]:
        def run(k):
            return self._run_single_attack(clean_image, target_image, source_crop, target_crop, cluster_number=k)

        if not self.config.use_adaptive_cluster:
            return run(self.config.cluster_number), self.config.cluster_number

        k1, k2 = self.config.adaptive_clusters
        adv_k1 = run(k1)

        sim_score = self._llm_similarity(target_path, to_pil(adv_k1))

        if sim_score >= self.config.llm_similarity_threshold:
            return adv_k1, k1

        return run(k2), k2

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        clean_image = to_tensor(sample.images[0], 255).to(self.config.device)
        target_image, target_path = self._load_target_image(sample)

        source_crop, target_crop = random_crops(self.config, clean_image.shape[-1] / clean_image.shape[-2])

        adv_image, final_cluster = self._run_adaptive_attack(clean_image, target_image, source_crop, target_crop, target_path)

        adv_pil = to_pil(adv_image)

        return AttackResult(
            adversarial_sample=adv_pil,
            perturbation_norm=pixel_linf(adv_pil, sample.images[0]),
            metadata={"cluster_number": final_cluster},
        )
