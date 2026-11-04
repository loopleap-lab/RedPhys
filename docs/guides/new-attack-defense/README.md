# New Attack or Defense

An attack or defense is a config dataclass, a class with one method, a registry line and a YAML. `redphys.attack` / `redphys.defend`
do the rest: loading images, targets and questions, seeding, writing `<id>.png` + `manifest.json`, resuming.

## Example: adding MPCAttack

[MPCAttack](https://arxiv.org/abs/2603.04846) (CVPR 2026) attacks an ensemble of CLIP, InternVL and DINOv2 surrogates toward a
target image. Files: `redphys/attacks/mpcattack/{mpcattack_attack.py, core.py}`, `configs/attacks/mpcattack.yaml`.

**1. Config:** every hyperparameter, plus the run inputs it needs (`target_images_dir` is filled by `redphys.attack`).

```python
@dataclass
class MPCAttackConfig(AttackConfig):  # inherits epsilon, max_iterations, alpha, seed, device, precision
    epsilon: float = 16.0
    max_iterations: int = 300
    clip_backbones: List[str] = field(default_factory=lambda: ["openai/clip-vit-base-patch16", ...])
    internvl_model: str = "OpenGVLab/InternVL3-1B-hf"
    dino_model: str = "facebook/dinov2-base"
    lam: float = 0.6
    target_images_dir: Optional[str] = None  # run input
```

**2. Class:** `generate(sample)` returns the adversarial image at the source's native size. The upstream optimization loop
stays in `core.py`; the class only adapts inputs and outputs.

```python
class MPCAttack(BaseAttack):
    def generate(self, sample, **kwargs) -> AttackResult:
        c = self.config
        self._load_models()                                     # lazily, once per run
        src = sample.images[0]                                  # PIL RGB, native size
        tgt = load_rgb(find_by_stem(c.target_images_dir, sample.id, required=True))  # paired by file stem
        adv = to_pil(mpcattack(src, tgt, self._models, c))     # upstream loop (core.py), simplified
        return AttackResult(adversarial_sample=adv, perturbation_norm=pixel_linf(adv, src))
```

**3. Register** in `redphys/attacks/registry.py` (`register_all_attacks`):

```python
("mpcattack", MPCAttack, MPCAttackConfig),
```

**4. YAML** `configs/attacks/mpcattack.yaml`, at the paper's values (keyword arguments override it):

```yaml
epsilon: 16.0
max_iterations: 300          # paper
clip_backbones: ["openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32", "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"]
internvl_model: "OpenGVLab/InternVL3-1B-hf"
```

**5. Run:**

```python
redphys.attack("mpcattack", "dataset/physical", "out/mpcattack", targets="out/targets", max_iterations=10)  # quick test
```

What `redphys.attack` can give an attack:

| Need | Declare | Gets |
|---|---|---|
| Target images | `target_images_dir: str` | the targets' `target/` dir, paired by file stem |
| Target text | `target_text: str` | per case from `labels.json` (`sample.metadata["attack_target_text"]`) |
| The question | `uses_question = True` (class var) | each case's question in `sample.question` |
| The correct answer | `needs_source_text = True` | `sample.metadata["attack_source_text"]` |
| An LLM | mix in `LLMClientMixin` | `self.llm_text(prompt)`, with `llm_server_url` / `llm_model` from the YAML |

## Example: adding OSCP

[OSCP](https://arxiv.org/abs/2408.17064) (CVPR 2025) purifies an image with one ControlNet-guided diffusion step. File:
`redphys/defense/oscp/oscp_defense.py`, weights in `redphys/defense/oscp/assets/`.

```python
@dataclass
class OSCPDefenseConfig(DefenseConfig):  # inherits device, seed, precision
    seed: int = 100                      # paper
    strength: float = 0.2                # t* = 200
    control_scale: float = 0.8
    lora: Optional[str] = None           # default: assets/checkpoints/oscp_lora.safetensors

class OSCPDefense(BaseDefense):
    def clean(self, image_path, **kwargs) -> DefenseResult:
        self._load()                                          # SD 1.5 + Canny ControlNet + LoRA, once
        image = load_rgb(image_path)
        small = image.resize((512, 512))
        out = self._pipe(prompt="", image=small, control_image=canny(small), strength=self.config.strength, ...).images[0]
        return DefenseResult(cleaned_sample=out.resize(image.size))  # back to the input size
```

Register it in `redphys/defense/registry.py` (`("oscp", OSCPDefense, OSCPDefenseConfig)`), add `configs/defenses/oscp.yaml`
and put the LoRA in `install.sh` with its sha256. Then:

```python
cleaned = redphys.defend("oscp", "out/mpcattack", "out/mpcattack_oscp")
redphys.evaluate(["out/mpcattack", cleaned], targets="out/targets", server_url=..., model=..., judge=None)  # attacked vs. defended
```

A defense that also rewrites the text receives the victim's question as `clean(image_path, prompt=...)` and returns
`metadata={"original_prompt": prompt, "final_prompt": new_prompt}`; the evaluation then sends `final_prompt`.

## Checklist

- Port only the code path behind the paper's results; keep the upstream code in its own module (`core.py`).
- Work at the image's native size; `epsilon` is L-inf in pixels (0-255); `perturbation_norm=None` if not a perturbation.
- All hyperparameters in the YAML; weights in `install.sh` with sha256; new packages checked with `uv pip check`.
- Test on a few images with few iterations: outputs exist, keep the source's size, and the loss goes down.
