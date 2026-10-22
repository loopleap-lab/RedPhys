"""AdPO (ICLR 2026, arXiv 2504.01735): preference optimization of the vision encoder against adversarial images.

No official code; trainer after ADPO (EMNLP 2025 Findings). Qwen3-VL, Cosmos-Reason2, InternVL3.5-HF; Qwen3.5 and
Gemma 4 need transformers >= 5. adv_source "pgd" is the paper's (Eq. 4, online answers).
Deviations: default x_adv from REALM attacks with answers precomputed by the original model; projector frozen;
REALM images instead of ImageNet; constant lr.
"""

from .adpo_defense import AdPODefense
from .config import AdPOConfig

__all__ = ["AdPODefense", "AdPOConfig"]
