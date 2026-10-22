# Configuration

Every setting has a default in `configs/` (the paper's values). Change one for a run by keyword, or point to another YAML.

```
configs/
  agents/scenario.yaml        # realm.target
  attacks/<attack>.yaml       # realm.attack
  defenses/<defense>.yaml     # realm.defend
```

## Overrides

```python
realm.attack("foa", "dataset/physical", "out/foa_eps8", targets=targets, epsilon=8, max_iterations=100)
realm.defend("freqpure", "out/foa", "out/foa_fp30", forward_noise_steps=30)
realm.target("dataset/physical/cases.json", "out/targets", generator="local", max_rounds=3)
realm.attack("foa", "dataset/physical", "out/foa_custom", targets=targets, config="my_foa.yaml")  # another YAML
```

Precedence: code defaults < YAML (`config=`, else `configs/<kind>/<name>.yaml`) < keyword arguments. An unknown key is an
error that lists the valid ones. On the command line, `scripts/attack.py` and `scripts/defense.py` take `--config`.

The settings of a run are saved in its `manifest.json`. A rerun into the same output resumes it; a rerun with other settings
into the same output is refused, so give each setting its own output directory.

## Settings shared by attacks

| Setting | Default | |
|---|---|---|
| `epsilon` | 16 | L-inf budget in pixels (0-255) |
| `max_iterations`, `alpha` | per attack | optimization steps and step size (pixels) |
| `seed` | 42 | each image is seeded from (seed, image id), so results do not depend on order or resumes |
| `precision` | `bf16` | `fp32` for exact runs or GPUs without bf16 (V100) |
| `device` | `cuda` | `cpu` works for small surrogates, slowly |

`surrogates="unified"` (keyword of `realm.attack`) swaps an attack's own surrogates for one shared CLIP ensemble
(`realm/attacks/surrogates.py`), to compare attacks on equal footing.

## Target generation (`configs/agents/scenario.yaml`)

| Setting | Default | |
|---|---|---|
| `model` | `openai/gpt-6-luna` | agent VLM (`server_url`, `api_key_env`: OpenRouter) |
| `generator` | `openrouter` | `local`: the server at `generator_url` |
| `generator_model` | `qwen/qwen-image-3` | OpenRouter image model |
| `checker_models` | 3 models | independent voters that check each target image |
| `max_rounds`, `max_steps` | 5, 24 | images and agent turns per case |
| `num_inference_steps`, `seed` | 40, 0 | local generator only |

Local image generator (Qwen-Image, two 24 GB GPUs; restarted if it exits):

```bash
bash realm/agent/target_generator/server.sh 8091  # the default generator_url
```

```python
realm.target(..., generator="local")
```

## Evaluation (`realm.evaluate` / `realm.score` arguments)

| Argument | Default | |
|---|---|---|
| `server_url`, `api_key`, `model` | | the victim: any OpenAI-compatible server (vLLM, OpenRouter, an agent) |
| `judge` | `choice` | reads the chosen option; `None`: collect answers only, judge later with `realm.score` |
| `judge_model` | `Qwen/Qwen3-8B` | the paper's judge, on the same `server_url` (or the one given to `realm.score`) |
| `workers` | 4 | parallel requests |
| `max_image_mb` | `None` | downscale larger images (API limits) |

Victim settings are fixed to the paper's: temperature 0, at most 2048 tokens, thinking off.

## API keys

| Variable | Used by |
|---|---|
| `OPENROUTER_API_KEY` | `realm.target`; FigStep, PromptInject, LeadingQ, SceneTAP; OpenRouter victims |
| `OPENAI_API_KEY` | FOA, CoA, PhysPatch, BlueSuffix (their `llm_*` settings can point to any OpenAI-compatible server) |

An attack or defense that calls an LLM reads `llm_server_url`, `llm_api_key_env` and `llm_model` from its YAML.
