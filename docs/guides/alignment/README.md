# Alignment

REALM's attacks turn any cases into pairs of clean and adversarial images. The `adpo` defense uses these pairs to fine-tune the
victim's vision encoder with [AdPO](https://arxiv.org/abs/2504.01735) (ICLR 2026), so that it answers the adversarial image as
it answers the clean one.

## 1. Build the dataset

```python
import realm

targets = realm.target("dataset/medical/cases.json", "out/medical/targets", generator="local")
for name in ["foa", "mattack", "vattack"]:
    realm.attack(name, "dataset/medical/clean", f"out/medical/{name}", targets=targets)
```

Each attack directory has `<id>.png` and a `manifest.json`, the format of the released dataset, so released attack
directories (`dataset/realm_dataset/adversarial/<attack>`) can be used directly. Text-only attacks (`promptinject`,
`leadingq`) leave the image unchanged and are skipped.

## 2. Objective

With $`y_w`$ and $`y_l`$ the model's answers on the clean image $`x`$ and the adversarial image $`x_{adv}`$:

<div align="center">

$`\displaystyle \mathcal{L}_{\text{PIO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x_{adv})}{\pi_{\text{ref}}(y_l \mid x_{adv})}\right)`$

</div>

<div align="center">

$`\displaystyle \mathcal{L}_{\text{AIO}} = -\sum_t \text{sg}\left(\pi_\theta(y_w^t \mid x_{adv})\right) \log \pi_\theta(y_w^t \mid x_{adv}), \qquad \mathcal{L} = \mathcal{L}_{\text{PIO}} + \lambda \mathcal{L}_{\text{AIO}}`$

</div>

$`\mathcal{L}_{\text{PIO}}`$ (DPO) prefers the clean answer over the adversarial one, relative to the original model
$`\pi_{\text{ref}}`$; $`\mathcal{L}_{\text{AIO}}`$ makes the model give the clean answer on the adversarial image. Only the
vision encoder is trained.

## 3. Train

```python
realm.defend("adpo", ["out/medical/foa", "out/medical/mattack", "out/medical/vattack"], "out/medical/adpo",
             model="Qwen/Qwen3-VL-8B-Instruct", holdout_attacks=["vattack"])
```

The output directory, built in order (an interrupted run resumes):

| File | Content |
|---|---|
| `pairs.jsonl` | One row per (attack, image), with a train/test split by image and the held-out attacks marked. |
| `responses.jsonl` | $`y_w`$ and $`y_l`$ from the original model, asked "What is the content of the image?" (paper). |
| `visual.safetensors` | The trained vision encoder. |
| `checkpoint/` | The full model with the new encoder, for transformers or vLLM. |

A row of `responses.jsonl` (Qwen3-VL-8B, FOA): the attack turns the fruit into a potato.

```json
{"id": "grasping_0041", "attack": "foa", "split": "train", "prompt": "What is the content of the image?",
 "y_w": "The image shows a robotic arm placing a fruit into a glass jar on a wooden desk. ...",
 "y_l": "The image shows a robotic arm placing a potato into a glass jar on a countertop. ..."}
```

Supported victims: Qwen3-VL, Cosmos-Reason2, InternVL3.5-HF, Qwen3.5 and Gemma 4. Main settings (`configs/defenses/adpo.yaml`):

| Setting | Default | |
|---|---|---|
| `beta`, `lam` | 0.1, 1.0 | paper |
| `lr`, `epochs`, `batch_size` | 1e-5, 2, 128 | paper (AdamW, weight decay 1e-4) |
| `adv_source` | `realm` | the attack outputs; `pgd`: the paper's online PGD (ε = 2/255) |
| `prompt` | "What is the content of the image?" | `null`: each case's question |
| `device` | `cuda` | `auto`: one model over several GPUs (8B on 24 GB cards) |

Several GPUs in data parallel: `torchrun --nproc_per_node N` around the same call.

## 4. Evaluate the trained model

```bash
vllm serve out/medical/adpo/checkpoint --port 8000
```

```python
realm.evaluate(["out/medical/foa", "out/medical/vattack"], targets="out/medical/targets",
               server_url="http://localhost:8000", model="out/medical/adpo/checkpoint", output="results/adpo")
```

Compare with the original model on the test split and on held-out attacks, and check clean accuracy as well.

## 5. A new alignment algorithm

Any training method on the same pairs plugs in as a training defense: set `trains = True` and implement `fit` instead of
`clean`. `realm.defend` passes it the attack directories and the output directory, records the config and resumes only the
same fit.

```python
# realm/defense/mymethod/mymethod_defense.py
from ..adpo.data import load_pairs, write_jsonl     # (attack, image) pairs with the train/test split
from ..adpo.models import load, encode, token_logps  # Qwen3-VL / InternVL / Gemma 4 loading and log-probs

@dataclass
class MyMethodConfig(DefenseConfig):
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    lr: float = 1e-5
    test_fraction: float = 0.2

class MyMethodDefense(BaseDefense):
    trains: ClassVar[bool] = True

    def fit(self, inputs: list[str], out_dir: Path) -> dict:
        pairs = load_pairs(inputs, test_fraction=self.config.test_fraction)
        write_jsonl(out_dir / "pairs.jsonl", pairs)
        model, vlm = load(self.config.model, "cuda")
        ...  # train on the "train" pairs: p["clean"], p["adv"], p["question"]
        model.save_pretrained(out_dir / "checkpoint")
        return {"pairs": "pairs.jsonl", "checkpoint": "checkpoint"}  # recorded in the manifest
```

Register it (`("mymethod", MyMethodDefense, MyMethodConfig)` in `realm/defense/registry.py`), add
`configs/defenses/mymethod.yaml`, and run `realm.defend("mymethod", attack_dirs, "out/mymethod")`. The same split
(`pairs.jsonl`) and evaluation make the methods directly comparable. AdPO (`realm/defense/adpo/`) is the reference
implementation: stage-1 answers, an HF `Trainer` subclass, and an export to a vLLM-servable checkpoint.
