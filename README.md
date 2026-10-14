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

REALM is a red-teaming **attack-and-defense library** for evaluating the adversarial robustness of Vision-Language Models (VLMs) deployed in safety-critical physical-world tasks — object detection and embodied AI. It provides **12 attack methods**, **3 model-agnostic defenses**, and an automated evaluation pipeline that **measures both attack and defense effectiveness**. All attacks are **black-box** against the victim VLM — optimized on CLIP surrogates and transferred to a range of proprietary and open-source models, reflecting realistic threat models.

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
```

`install.sh` installs REALM, its dependencies and the model weights, matching PyTorch to your GPU driver. It needs Python 3.10+, `git`, `wget` and Docker (for `realm.target`); on a system Python it installs into `.venv` and opens a shell with it active.

### Generate Adversarial Samples

`dataset/physical/` has real-world cases (driving, egocentric, manipulation, grasping, diagnostics, physics, scene QA), each with a question, options and answer in `cases.json`.

```python
import realm

# The scenario agent (an LLM calling the tools plan, generate, check and submit) picks a wrong answer and builds its target image
# for each case; it runs in a Docker container (OpenRouter: needs OPENROUTER_API_KEY)
targets = realm.target("dataset/physical/cases.json", "out/targets")

# Gradient-based (CLIP surrogate)
adv = realm.attack("foa", "dataset/physical", "out/foa", targets=targets)

# Untargeted
realm.attack("paattack", "dataset/physical", "out/paattack")

# Text-guided (target text from the targets' labels, or target_text=...)
realm.attack("vattack", "dataset/physical", "out/vattack", targets=targets)

# Typographic injection (text written by the LLM in the YAML: OpenRouter, needs OPENROUTER_API_KEY)
realm.attack("figstep", "dataset/physical", "out/figstep", targets=targets)

# Prompt manipulation (appended to each case's question)
realm.attack("promptinject", "dataset/physical", "out/promptinject", targets=targets)
```

Each call returns its output directory (one PNG per case and `manifest.json`) and resumes where it stopped.

### Configure

Hyperparameters are in `configs/attacks/<attack>.yaml` and `configs/defenses/<defense>.yaml` (paper defaults). Override any of them per call by keyword: `realm.attack("foa", ..., epsilon=8)`, `realm.defend("freqpure", ..., delta=0.2)`. Same for `realm.target` (`configs/agents/scenario.yaml`), e.g. `checker_server_url="http://localhost:8000"` for local checkers.

Helper LLM calls (FOA's cluster check, CoA's captions, PhysPatch's patch placement, FigStep/PromptInject's injection text, BlueSuffix's text purifier) go to `llm_server_url` (null: OpenAI; FigStep and PromptInject require it set) with the key in the env var named by `llm_api_key_env`, both in the YAML.

### Evaluate

Score attack and defense outputs with a VLM judge. `server_url` points at any OpenAI-compatible endpoint (a local vLLM server, OpenAI, OpenRouter, ...); the judge runs on the same server, `openai/gpt-4o` by default (`judge_model` for another, e.g. on a local server):

```bash
# Serve the victim VLM (OpenAI-compatible endpoint)
.venv-vllm/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000   # vLLM lives in its own venv (install.sh); it downloads the model on first serve
```

```python
import os

realm.evaluate(["out/foa", "out/foa_freqpure"], targets="out/targets", server_url="http://localhost:8000",
               model="Qwen/Qwen3-VL-8B-Instruct", judge_model="Qwen/Qwen3-VL-8B-Instruct", output="results/eval/foa")

# Proprietary victim via OpenRouter (judge: openai/gpt-4o)
realm.evaluate("out/foa", targets="out/targets", server_url="https://openrouter.ai/api", api_key=os.environ["OPENROUTER_API_KEY"],
               model="openai/gpt-4.1-mini")
```

Reports **accuracy** and **Attack Success Rate (ASR)** for each input set against a clean baseline, and returns the same results that `output/eval_results.json` holds. Pass attack and defense output dirs together to compare them in one table. The target is per case (`targets`: the `realm.target` dir, or `labels`: its `labels.json`) or one `target_text`.

### Apply Defenses

```python
clean = realm.defend("freqpure", adv, "out/foa_freqpure")
```

### Target Generator

```bash
bash realm/agent/target_generator/server.sh 8091   # Qwen-Image-2.1 behind the OpenAI images API (generations, edits)
```

The same steps run from the shell as `scripts/generate_targets.py`, `generate_adversarial.py`, `clean_adversarial.py` and `evaluate_adversarial.py` (`--help` lists the options).

## Attacks

12 black-box attacks in three channels: perturbation (FOA, M-Attack, V-Attack, CoA, PhysPatch, PA-Attack), generation and editing (AdvDiffVLM, AdvEDM-A, AnyAttack) and injection (FigStep, PromptInject, ImageMix). Settings, including ε, are in `configs/attacks/*.yaml`.

<details>
<summary>…</summary>

| Attack | Category | ε |
|--------|----------|---|
| FOA | Perturbation (optimal transport) | 16/255 |
| M-Attack | Perturbation (cosine) | 16/255 |
| V-Attack | Perturbation (value features) | 16/255 |
| CoA | Perturbation (multimodal co-optimization) | 16/255 |
| PhysPatch | Perturbation (patch) | 16/255 |
| PA-Attack | Perturbation (untargeted, OOD prototypes) | 16/255 |
| AdvDiffVLM | Generation (diffusion latent) | ∞ |
| AdvEDM-A | Editing (semantic addition) | 16/255 |
| AnyAttack | Generation (single forward pass) | 16/255 |
| FigStep | Injection (typographic) | — |
| PromptInject | Injection (prompt) | — |
| ImageMix | Injection (alpha-blend baseline) | — |

</details>

## Defenses

3 model-agnostic defenses applied before the victim sees the input: patch removal (PAD), frequency purification (FreqPure) and multimodal purification (BlueSuffix). Settings are in `configs/defenses/*.yaml`.

<details>
<summary>…</summary>

| Defense | Type | Description |
|---------|------|-------------|
| PAD | Patch removal | MI/CD heatmap fusion → SAM segmentation → patch removal |
| FreqPure | Frequency purification | FFT amplitude swap + phase clipping + diffusion denoising |
| BlueSuffix | Multimodal purification | Image denoising + text purification + defensive suffix |

</details>

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
