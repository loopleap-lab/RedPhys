<div align="center">

# REALM: A Unified Red-Teaming Benchmark for Physical-World VLMs

**NeurIPS 2026 · Evaluations & Datasets Track**

<p>
<a href="https://arxiv.org/abs/2606.23892"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper"></a>
<a href="https://github.com/UCF-ML-Research/REALM"><img src="https://img.shields.io/badge/Website-REALM-7c5cff?logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://huggingface.co/datasets/yifeiz29/REALM"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-ffd21e?logo=huggingface" alt="Hugging Face"></a>
</p>

</div>

## Overview

REALM is a unified red-teaming benchmark for vision-language models (VLMs) in safety-critical physical-world tasks such as driving and robotics. It brings **12 red-teaming methods**, **3 model-agnostic defenses** and **13 VLMs** under one black-box threat model with shared data and metrics, and an agentic pipeline builds a scenario-specific target for each scene so that every attack pursues the same failure.

## Key Features

- 🎯 **12 attack methods** — gradient-based visual perturbations, localized patches, diffusion generation and semantic editing, and prompt/typographic injection, plus a non-adversarial baseline.
- 🛡️ **3 model-agnostic defenses** — PAD, FreqPure, and BlueSuffix, applied as input preprocessing with no model retraining.
- 🔒 **Black-box threat model** — attacks are optimized on CLIP surrogates and transferred; no access to the victim's weights, gradients, or logits.
- 🧩 **Extensible** — plugin registries; add a new attack or defense by implementing a single base class and registering it.
- 🔌 **Flexible backends** — evaluate proprietary models through the OpenRouter API or open-source VLMs served locally with vLLM.

## Quick Start

### Installation

```bash
git clone https://github.com/UCF-ML-Research/REALM.git
cd REALM
bash install.sh
hf download yifeiz29/REALM --repo-type dataset --local-dir dataset/realm_dataset
```

`install.sh` sets up REALM, its dependencies and the model weights (Python 3.10+; Docker for `realm.target`).

### Evaluate Robustness

Score any VLM on the released adversarial images, here a proprietary model through OpenRouter:

```python
import os
import realm

realm.evaluate(["dataset/realm_dataset/adversarial/foa", "dataset/realm_dataset/adversarial/figstep"],
               server_url="https://openrouter.ai/api", api_key=os.environ["OPENROUTER_API_KEY"],
               model="openai/gpt-4.1-mini", output="results/gpt-4.1-mini")
```

Reports **accuracy** and **Attack Success Rate (ASR)** of each attack against the clean baseline (judge: `openai/gpt-4o` on the same server) and saves them to `output/eval_results.json`.

### Generate Adversarial Samples

```python
targets = realm.target("dataset/physical/cases.json", "out/targets")  # scenario agent, in Docker (OPENROUTER_API_KEY)
adv = realm.attack("foa", "dataset/physical", "out/foa", targets=targets)
realm.attack("paattack", "dataset/physical", "out/paattack")  # untargeted
```

Each call writes one PNG per case and `manifest.json`, and resumes where it stopped.

### Apply Defenses

```python
clean = realm.defend("freqpure", adv, "out/foa_freqpure")
```

### Evaluate a Local VLM

```bash
.venv-vllm/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
```

```python
realm.evaluate(["out/foa", "out/foa_freqpure"], targets="out/targets", server_url="http://localhost:8000",
               model="Qwen/Qwen3-VL-8B-Instruct", judge_model="Qwen/Qwen3-VL-8B-Instruct")
```

### Configure

Settings are in `configs/attacks/`, `configs/defenses/` and `configs/agents/scenario.yaml` (paper defaults); override any by keyword, e.g. `realm.attack("foa", ..., epsilon=8)`. Helper LLM calls go to `llm_server_url` with the key in `llm_api_key_env`. `bash realm/agent/target_generator/server.sh 8091` serves the target generator locally, and `scripts/` runs the same steps from the shell.

## Acknowledgements

This project integrates adversarial attack and defense methods proposed by prior research. We thank the original authors for making their work publicly available.

## Citation

If you find our repo useful, please cite our paper:

```bibtex
@article{zhao2026realm,
  title={REALM: A Unified Red-Teaming Benchmark for Physical-World VLMs},
  author={Zhao, Yifei and Lou, Qian and Zheng, Mengxin},
  journal={arXiv preprint arXiv:2606.23892},
  year={2026}
}
```
