<div align="center">

# RedPhys: A Unified Red-Teaming Benchmark for Physical-World VLMs

**NeurIPS 2026 · Evaluations & Datasets Track**

<p>
<a href="https://arxiv.org/abs/2606.23892"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper"></a>
<a href="https://github.com/UCF-ML-Research/RedPhys"><img src="https://img.shields.io/badge/Website-RedPhys-7c5cff?logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://huggingface.co/datasets/yifeiz29/RedPhys"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-ffd21e?logo=huggingface" alt="Hugging Face"></a>
</p>

</div>

## Overview

RedPhys is a unified red-teaming benchmark for vision-language models (VLMs) in safety-critical physical-world tasks such as driving and robotics. It brings **12 red-teaming methods**, **3 model-agnostic defenses** and **13 VLMs** under one black-box threat model with shared data and metrics, and an agentic pipeline builds a scenario-specific target for each scene so that all attacks are compared on the same failure.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/dark.png">
    <img src="docs/assets/light.png" alt="RedPhys library of scenarios, attacks, defenses and VLMs, and the red-teaming benchmark pipeline" width="100%">
  </picture>
</p>

## Key Features

- 🎯 **12 attacks** — perturbation, patch, diffusion generation and editing, and prompt/typographic injection.
- 🛡️ **3 defenses** — removal, purification and sanitization.
- 🔒 **Realistic threat model** — optimized on CLIP surrogates and transferred to the victim VLM.
- ⚖️ **Adversarial alignment** — build an adversarial dataset with the attacks and fine-tune the victim on it to strengthen its alignment.
- 🧩 **Extensible** — add an attack or defense with one base class; evaluate any VLM or VLM agent.

## Installation

```bash
git clone https://github.com/UCF-ML-Research/RedPhys.git
cd RedPhys
bash install.sh
hf download yifeiz29/RedPhys --repo-type dataset --local-dir dataset/redphys_dataset
```

`install.sh` sets up RedPhys, its dependencies and the model weights (Python 3.10+; Docker for `redphys.target`).

## How to Use RedPhys

### Evaluate on the Released Dataset

Score any VLM on the released adversarial images, here a proprietary model through OpenRouter:

```python
import os
import redphys

redphys.evaluate(["dataset/redphys_dataset/adversarial/foa", "dataset/redphys_dataset/adversarial/figstep"],
               server_url="https://openrouter.ai/api", api_key=os.environ["OPENROUTER_API_KEY"],
               model="google/gemini-3-flash-preview", judge=None, output="results/gemini-3-flash")
redphys.score("results/gemini-3-flash", server_url="http://localhost:8000")
```

A Qwen3-8B judge scores the answers; accuracy and ASR go to `results/gemini-3-flash/eval_results.json`.

### Run the Full Pipeline

`dataset/physical/` has real-world cases, each with a question, options and answer in `cases.json`.

**1. Generate targets** with the scenario agent (in Docker):

```python
import os
import redphys

targets = redphys.target("dataset/physical/cases.json", "out/targets")
```

The agent changes each scene to show a wrong answer.

**2. Attack.** Every attack (`foa`, `mattack`, `vattack`, `coa`, `physpatch`, `paattack`, `advdiffvlm`, `advedm`, `anyattack`, `figstep`, `promptinject`, `imagemix`) takes the same call:

```python
adv = redphys.attack("foa", "dataset/physical", "out/foa", targets=targets)
```

**3. Evaluate.**

```python
redphys.evaluate(adv, targets=targets, server_url="https://openrouter.ai/api",
               api_key=os.environ["OPENROUTER_API_KEY"], model="openai/gpt-4.1-mini", judge=None, output="results/foa")
redphys.score("results/foa", server_url="http://localhost:8000")
```

For an open-source VLM, serve it with vLLM and pass `server_url="http://localhost:8000"`:

```bash
.venv-vllm/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
```

## FAQ

<details>
<summary><b><a href="docs/guides/configuration">Configuration</a></b></summary>

Defaults (the paper settings) are in `configs/`; override any of them by keyword:

```python
redphys.attack("foa", "dataset/physical", "out/foa_eps8", targets=targets, epsilon=8)  # configs/attacks/foa.yaml
redphys.defend("freqpure", adv, "out/foa_fp30", forward_noise_steps=30)               # configs/defenses/freqpure.yaml
redphys.target("dataset/physical/cases.json", "out/targets", generator="local")       # configs/agents/scenario.yaml
```

```bash
bash redphys/agent/target_generator/server.sh 8091  # local target image generator (generator="local")
```

</details>

<details>
<summary><b><a href="docs/guides/new-scenario">New scenario or dataset</a></b></summary>

Write a `cases.json` like `dataset/physical/cases.json` and run the pipeline on it:

```json
[{"id": "medical_0001", "image": "dataset/medical/clean/medical_0001.png", "domain": "medical",
  "question": "What is the main finding in this chest X-ray?\nA. Atelectasis\nB. Cardiomegaly\nC. No Finding\nD. Pleural Effusion",
  "options": {"A": "Atelectasis", "B": "Cardiomegaly", "C": "No Finding", "D": "Pleural Effusion"}, "answer": "B"}]
```

Targets (a wrong answer and an image that shows it) can be your own images, or come from the scenario agent, which plans the wrong answer, edits or draws the image and checks it with independent VLMs. For a new `domain`, tell it what to keep from the source in `DOMAIN_CONSTRAINTS` (`redphys/agent/scenario_agent/prompts.py`). Try a few cases first:

```python
targets = redphys.target("dataset/medical/cases.json", "out/medical_targets", max_samples=5)
```

</details>

<details>
<summary><b><a href="docs/guides/new-attack-defense">New attack or defense</a></b></summary>

Subclass `BaseAttack` or `BaseDefense`, register it and add its YAML under `configs/attacks/` or `configs/defenses/`:

```python
# redphys/attacks/myattack/myattack_attack.py
@dataclass
class MyAttackConfig(AttackConfig):  # epsilon, max_iterations, surrogates, ...
    target_images_dir: str = ""  # filled per run (or target_text)
    my_param: float = 0.5

class MyAttack(BaseAttack):
    def generate(self, sample, **kwargs) -> AttackResult:  # sample: images, question, metadata
        ...
        return AttackResult(adversarial_sample=adv, perturbation_norm=self.config.epsilon)

# redphys/defense/mydefense/mydefense_defense.py
@dataclass
class MyDefenseConfig(DefenseConfig):  # device, seed, precision
    strength: float = 0.3

class MyDefense(BaseDefense):
    def clean(self, image_path, **kwargs) -> DefenseResult:
        ...
        return DefenseResult(cleaned_sample=cleaned)

# redphys/attacks/registry.py (register_all_attacks) and redphys/defense/registry.py (register_all_defenses)
("myattack", MyAttack, MyAttackConfig),
("mydefense", MyDefense, MyDefenseConfig),
```

Then run them by name: `redphys.attack("myattack", ...)`, `redphys.defend("mydefense", ...)`.

</details>

<details>
<summary><b><a href="docs/guides/alignment">How do I build an adversarial dataset and train a more robust model?</a></b></summary>

**1. Build the dataset.** Run attacks on your cases (see *New scenario or dataset*):

```python
targets = redphys.target("dataset/medical/cases.json", "out/medical/targets")
for name in ["foa", "mattack", "vattack"]:
    redphys.attack(name, "dataset/medical/clean", f"out/medical/{name}", targets=targets)
```

Each attack directory holds the adversarial images and a `manifest.json`, in the format of the released dataset.

**2. Train.** [AdPO](https://arxiv.org/abs/2504.01735) is an additional, training-based defense: unlike the 3 input defenses, it fine-tunes the victim's vision encoder. With $`y_w`$ and $`y_l`$ the model's answers on the clean image $`x`$ and the adversarial image $`x_{adv}`$:

<div align="center">

$`\displaystyle \mathcal{L}_{\text{PIO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x_{adv})}{\pi_{\text{ref}}(y_l \mid x_{adv})}\right)`$

</div>

<div align="center">

$`\displaystyle \mathcal{L}_{\text{AIO}} = -\sum_t \text{sg}\left(\pi_\theta(y_w^t \mid x_{adv})\right) \log \pi_\theta(y_w^t \mid x_{adv}), \qquad \mathcal{L} = \mathcal{L}_{\text{PIO}} + \lambda \mathcal{L}_{\text{AIO}}`$

</div>

$`\mathcal{L}_{\text{PIO}}`$ (DPO) prefers the clean answer over the adversarial one, relative to the original model $`\pi_{\text{ref}}`$; $`\mathcal{L}_{\text{AIO}}`$ makes the model give the clean answer on the adversarial image, weighted by its confidence (sg: stop-gradient).

```python
redphys.defend("adpo", ["out/medical/foa", "out/medical/mattack", "out/medical/vattack"], "out/medical/adpo",
             model="Qwen/Qwen3-VL-8B-Instruct")
```

Supported victims: Qwen3-VL, Cosmos-Reason2, InternVL3.5-HF, Qwen3.5 and Gemma 4. Settings ($`\beta`$, $`\lambda`$, ...) are in `configs/defenses/adpo.yaml`; the trained model is written to `out/medical/adpo/checkpoint`.

</details>

<details>
<summary><b><a href="docs/guides/victim-agent">How do I extend RedPhys to a victim agent?</a></b></summary>

Put the agent behind an OpenAI-compatible chat API ([`agent_server.py`](docs/guides/victim-agent/agent_server.py) is a runnable
example with tools) and evaluate it like any VLM. Every attack, defense and metric applies, so you can analyze the agent's
robustness and compare it with its VLM:

```python
redphys.evaluate(["out/foa", "out/figstep"], targets="out/targets", server_url="http://localhost:9000",
               model="my-agent", judge=None, output="results/my_agent")
redphys.score("results/my_agent", server_url="http://localhost:8000")  # the judge on its own server
```

</details>

## Acknowledgements

This project integrates adversarial attack and defense methods proposed by prior research. We thank the original authors for making their work publicly available.

## License

RedPhys is released under the [Apache License 2.0](LICENSE). Code adapted from other projects keeps its original license (e.g. `redphys/attacks/physpatch/LICENSE`), and each dataset keeps the terms of its sources (its `license.json`).

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
