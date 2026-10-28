"""AdPO (ICLR 2026, arXiv 2504.01735): preference optimization of the vision encoder against adversarial images.

No official code; trainer after ADPO (EMNLP 2025 Findings). Qwen3-VL, Cosmos-Reason2, InternVL3.5-HF; Qwen3.5 and
Gemma 4 need transformers >= 5. adv_source "pgd" is the paper's (Eq. 4, online answers).
Deviations: x_adv from RedPhys attacks by default, with answers precomputed by the original model; the victim's own
encoder is trained (the paper trains TinyLLaVA's and transfers it); projector frozen (Qwen's merger sits inside `visual`);
Eq. 4 features are the encoder output before the projector; RedPhys images instead of ImageNet; constant lr.
L_AIO is dynamic fine-tuning (Eq. 7, token-confidence weighted with stop-gradient) as in the ICLR version; arXiv v1 has the plain NLL.
"""

from .adpo_defense import AdPODefense
from .config import AdPOConfig

__all__ = ["AdPODefense", "AdPOConfig"]
