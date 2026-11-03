# Configuration

Every setting has a default in `configs/` (the paper's values). Change one for a run by keyword, or point to another YAML.

```
configs/
  agents/scenario.yaml        # redphys.target
  attacks/<attack>.yaml       # redphys.attack
  defenses/<defense>.yaml     # redphys.defend
```

## Overrides

```python
redphys.attack("foa", "dataset/physical", "out/foa_eps8", targets=targets, epsilon=8, max_iterations=100)
redphys.defend("freqpure", "out/foa", "out/foa_fp30", forward_noise_steps=30)
redphys.target("dataset/physical/cases.json", "out/targets", generator="local", max_rounds=3)
redphys.attack("foa", "dataset/physical", "out/foa_custom", targets=targets, config="my_foa.yaml")  # another YAML
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

By default an attack perturbs the image at its native size, and the surrogates see a random crop of it resized to `input_res`.
`resize_source: true` (M-Attack, FOA, V-Attack, CoA, MPCAttack, O-Attack) does what their official code does: the source is
resized and center-cropped to `input_res` (224), the attack runs on that image, and the output has that size. The resized
clean image is saved in `<output>/_source` and is the manifest's `source`, so evaluation compares against it.

`surrogates="unified"` (keyword of `redphys.attack`) swaps an attack's own surrogates for one shared CLIP ensemble
(`redphys/attacks/surrogates.py`), to compare attacks on equal footing.

## Target generation (`configs/agents/scenario.yaml`)

| Setting | Default | |
|---|---|---|
| `model` | `openai/gpt-6-luna` | agent PhyModel (`server_url`, `api_key_env`: OpenRouter) |
| `generator` | `local` | the server at `generator_url`; `openrouter`: `generator_model` |
| `generator_model` | `qwen/qwen-image-3` | OpenRouter image model |
| `check` | `false` | `true`: `checker_models` vote on each target image (`verified` in `targets.json`) |
| `checker_models` | 3 models | independent voters, with `check` |
| `max_rounds`, `max_steps` | 5, 24 | images and agent turns per case |
| `num_inference_steps`, `seed`, `generator_workers` | 40, 0, 2 | local generator only; `generator_workers`: concurrent requests to it |

Local image generator (Qwen-Image, two 24 GB GPUs; restarted if it exits):

```bash
bash redphys/agent/target_generator/server.sh 8091  # the default generator_url
```

```python
redphys.target(..., generator="local")
```

## Evaluation (`redphys.evaluate` / `redphys.score` arguments)

| Argument | Default | |
|---|---|---|
| `server_url`, `api_key`, `model` | | the victim: any OpenAI-compatible server (vLLM, OpenRouter, an agent) |
| `judge` | `choice` | reads the chosen option; `None`: collect answers only, judge later with `redphys.score` |
| `judge_model` | `Qwen/Qwen3-8B` | the paper's judge, on the same `server_url` (or the one given to `redphys.score`) |
| `workers` | 4 | parallel requests |
| `max_image_mb` | `None` | downscale larger images (API limits) |

Victim settings are fixed to the paper's: temperature 0, at most 2048 tokens, thinking off.

## API keys

| Variable | Used by |
|---|---|
| `OPENROUTER_API_KEY` | `redphys.target`; FigStep, PromptInject, LeadingQ, SceneTAP; OpenRouter victims |
| `OPENAI_API_KEY` | FOA, CoA, PhysPatch, BlueSuffix (their `llm_*` settings can point to any OpenAI-compatible server) |

An attack or defense that calls an LLM reads `llm_server_url`, `llm_api_key_env` and `llm_model` from its YAML.
