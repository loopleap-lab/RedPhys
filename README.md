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

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/realm-dark.png">
    <img src="assets/realm-light.png" alt="REALM library of scenarios, attacks, defenses and VLMs, and the red-teaming benchmark pipeline" width="100%">
  </picture>
</p>

## Key Features

- 🎯 **12 attacks** — perturbation, patch, diffusion generation and editing, and prompt/typographic injection.
- 🛡️ **3 defenses** — PAD, FreqPure and BlueSuffix, applied to the input with no retraining.
- 🔒 **Black-box** — optimized on CLIP surrogates and transferred to the victim VLM.
- 🧩 **Extensible** — add an attack or defense with one base class.
- 🔌 **Any backend** — proprietary models via OpenRouter, open-source ones via vLLM.

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
               model="google/gemini-3-flash-preview", judge=None, output="results/gemini-3-flash")
realm.score("results/gemini-3-flash", server_url="http://localhost:8000")  # Qwen3-8B judge
```

Collects the answers on each attack and the clean images, then a Qwen3-8B judge reads each chosen option and reports accuracy and ASR in `results/gemini-3-flash/eval_results.json`.

### Run the Full Pipeline

`dataset/physical/` has real-world cases, each with a question, options and answer in `cases.json`.

**1. Generate targets** with the scenario agent (in Docker):

```python
import realm

targets = realm.target("dataset/physical/cases.json", "out/targets")
```

The agent changes each scene to show a wrong answer; `mode="salient"` instead draws a new, unmistakable picture of it (e.g. an arrow sign for a turn).

**2. Attack.** Every attack (`foa`, `mattack`, `vattack`, `coa`, `physpatch`, `paattack`, `advdiffvlm`, `advedm`, `anyattack`, `figstep`, `promptinject`, `leadingq`, `scenetap`, `imagemix`) takes the same call:

```python
adv = realm.attack("foa", "dataset/physical", "out/foa", targets=targets)
```

**3. Evaluate.**

```python
import os

realm.evaluate(adv, targets=targets, server_url="https://openrouter.ai/api",
               api_key=os.environ["OPENROUTER_API_KEY"], model="openai/gpt-4.1-mini", judge=None, output="results/foa")
realm.score("results/foa", server_url="http://localhost:8000")  # Qwen3-8B judge
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

## Extending REALM

<details>
<summary><b>New scenario or dataset</b></summary>

Write a `cases.json` like `dataset/physical/cases.json` and run the pipeline on it:

```json
[{"id": "medical_0001", "image": "dataset/medical/clean/medical_0001.png", "domain": "medical",
  "question": "Which part of the body does this image belong to?\nA. Chest\nB. Neck\nC. Abdomen\nD. Head",
  "options": {"A": "Chest", "B": "Neck", "C": "Abdomen", "D": "Head"}, "answer": "D"}]
```

</details>

<details>
<summary><b>New model</b></summary>

Any model or agent behind an OpenAI-compatible chat API (image + text in, text out):

```python
realm.evaluate(adv, targets=targets, server_url="http://localhost:8000", model="my-model",
               judge=None, output="results/my-model")
realm.score("results/my-model", server_url="http://localhost:8000")  # Qwen3-8B judge
```

</details>

<details>
<summary><b>New attack</b></summary>

Subclass `BaseAttack`, register it and add `configs/attacks/<name>.yaml`:

```python
# realm/attacks/myattack/myattack_attack.py
@dataclass
class MyAttackConfig(AttackConfig):  # epsilon, max_iterations, surrogates, ...
    target_images_dir: str = ""  # run inputs are filled by field name: target_images_dir or target_text (see AttackConfig)
    my_param: float = 0.5

class MyAttack(BaseAttack):
    def generate(self, sample, **kwargs) -> AttackResult:  # sample: images, question, metadata (target text)
        ...
        return AttackResult(adversarial_sample=adv, perturbation_norm=self.config.epsilon)

# realm/attacks/registry.py, in register_all_attacks(): add to the list
("myattack", MyAttack, MyAttackConfig),
```

Then run it by name: `realm.attack("myattack", ...)`.

</details>

<details>
<summary><b>New defense</b></summary>

Subclass `BaseDefense`, register it and add `configs/defenses/<name>.yaml`:

```python
# realm/defense/mydefense/mydefense_defense.py
@dataclass
class MyDefenseConfig(DefenseConfig):  # device, seed, precision
    strength: float = 0.3

class MyDefense(BaseDefense):
    def clean(self, image_path, **kwargs) -> DefenseResult:
        ...
        return DefenseResult(cleaned_sample=cleaned)

# realm/defense/registry.py, in register_all_defenses(): add to the list
("mydefense", MyDefense, MyDefenseConfig),
```

Then run it by name: `realm.defend("mydefense", ...)`.

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
