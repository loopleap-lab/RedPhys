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

REALM is a unified red-teaming benchmark for vision-language models (VLMs) in safety-critical physical-world tasks such as driving and robotics. It brings **12 red-teaming methods**, **3 model-agnostic defenses** and **13 VLMs** under one black-box threat model with shared data and metrics, and an agentic pipeline builds a scenario-specific target for each scene so that all attacks are compared on the same failure.

## Key Features

- 🎯 **12 attack methods** — gradient-based visual perturbations, localized patches, diffusion generation and semantic editing, and prompt/typographic injection, plus a non-adversarial baseline.
- 🛡️ **3 model-agnostic defenses** — PAD, FreqPure, and BlueSuffix, applied as input preprocessing with no model retraining.
- 🔒 **Black-box threat model** — attacks are optimized on CLIP surrogates and transferred; no access to the victim's weights, gradients, or logits.
- 🧩 **Extensible** — plugin registries; add a new attack or defense by implementing a single base class and registering it.
- 🔌 **Flexible backends** — evaluate proprietary models through the OpenRouter API or open-source VLMs served locally with vLLM.

## Installation

```bash
git clone https://github.com/UCF-ML-Research/REALM.git
cd REALM
bash install.sh
hf download yifeiz29/REALM --repo-type dataset --local-dir dataset/realm_dataset
```

`install.sh` sets up REALM, its dependencies and the model weights (Python 3.10+; Docker for `realm.target`).

## How to Use REALM

### Evaluate on the Released Dataset

Score any VLM on the released adversarial images, here a proprietary model through OpenRouter:

```python
import os
import realm

realm.evaluate(["dataset/realm_dataset/adversarial/foa", "dataset/realm_dataset/adversarial/figstep"],
               server_url="https://openrouter.ai/api", api_key=os.environ["OPENROUTER_API_KEY"],
               model="google/gemini-3-flash-preview", output="results/gemini-3-flash")
```

Reports the model's **robustness** to each attack against the clean baseline (judge: `openai/gpt-4o` on the same server) and saves it to `output/eval_results.json`.

### Run the Full Pipeline

`dataset/physical/` has real-world cases, each with a question, options and answer in `cases.json`.

**1. Generate targets** with the scenario agent (in Docker):

```python
import realm

targets = realm.target("dataset/physical/cases.json", "out/targets")
```

**2. Attack.** Every attack (`foa`, `mattack`, `vattack`, `coa`, `physpatch`, `paattack`, `advdiffvlm`, `advedm`, `anyattack`, `figstep`, `promptinject`, `imagemix`) takes the same call:

```python
adv = realm.attack("foa", "dataset/physical", "out/foa", targets=targets)
```

**3. Evaluate.**

```python
import os

realm.evaluate(adv, targets=targets, server_url="https://openrouter.ai/api",
               api_key=os.environ["OPENROUTER_API_KEY"], model="openai/gpt-4.1-mini", output="results/foa")
```

For an open-source VLM, serve it with vLLM and pass `server_url="http://localhost:8000"`:

```bash
.venv-vllm/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
```

Each step resumes where it stopped.

### Configuration

Defaults (the paper settings) are in `configs/`; override any of them by keyword:

```python
realm.attack("foa", "dataset/physical", "out/foa_eps8", targets=targets, epsilon=8)  # configs/attacks/foa.yaml
realm.defend("freqpure", adv, "out/foa_fp30", forward_noise_steps=30)               # configs/defenses/freqpure.yaml
realm.target("dataset/physical/cases.json", "out/targets", generator="local")       # configs/agents/scenario.yaml
```

```yaml
# configs/attacks/figstep.yaml: helper LLM for the injection text
llm_server_url: "https://openrouter.ai/api"
llm_api_key_env: "OPENROUTER_API_KEY"
```

```bash
bash realm/agent/target_generator/server.sh 8091  # local target image generator (generator="local")
```

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
