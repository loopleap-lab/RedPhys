# RedPhys Roadmap

RedPhys red-teams vision-language models (VLMs) on physical-world tasks. A case (an image, a question and its
answer) goes in; the scenario agent turns it into a target, a wrong answer and an image that shows it; an attack
pushes the victim toward that target and a defense tries to undo it; the judge scores every answer. This roadmap
follows that path: the cases, the targets, the attacks and defenses, the victims and the scoring, then the
leaderboard and the automatic red-teaming built on them.

Done items are checked; they are in the current `main` branch (0.2.0, not yet tagged). Planned items are
unchecked.

Grouped by when they land, the work is:
1. **Library and benchmark (done):** attacks, defenses, the scenario agent and evaluation behind one API, and the
   paper's benchmark of 832 physical-world cases on 13 VLMs.
2. **Benchmark v2 and the live leaderboard (next):** scenes with clearer hazard signals, where a successful attack leads to a dangerous outcome,
   and a public leaderboard that reruns every submission on a hidden split.
3. **Automatic red-teaming (later):** a model or a use case in, a weakness report out, with agents that find
   scenes and integrate, run and evolve attacks and defenses.

## 1. Cases and datasets

- [x] `cases.json`: image, question, options and answer per case; any folder of cases runs through the pipeline
- [x] The paper's benchmark: 832 cases in 7 domains (driving, manipulation, grasping, physics, egocentric, diagnostics, scene QA), with every attack's images and 13 VLMs' answers ([Hugging Face](https://huggingface.co/datasets/yifeiz29/RedPhys))
- [x] `dataset/physical`: 50 cases where a wrong answer hides a hazard (collision, burn, cut, fall, fire), each with a `hazard` note and its source's license
- [x] `dataset/intuitive_physics` (100, Physics-IQ) and `dataset/medical` (100, VQA-RAD, SLAKE, Brain Tumor MRI)
- [x] `dataset/nips2017`: 100 of the NIPS 2017 / COCO pairs that M-Attack and FOA-Attack report on, scored by the captioning protocol (GPTScore, keyword match)
- [ ] [`nips2017`](https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set) adversarial images of every attack on all 1000 pairs, with manifests and captioning scores, on Hugging Face, as baselines for researchers to compare new attacks and defenses against (generated; judging and upload open)
- [ ] Benchmark v2: scenes with clearer hazard signals (a red light, a pedestrian in the lane, an open flame) that an adversarial attack must hide, so a successful attack leads to a dangerous outcome (collision, fall, wrong treatment); each case labeled with its hazard and severity, and ASR weighted by severity

## 2. Targets: the scenario agent

- [x] `redphys.target`: plans a wrong answer for each case, edits or draws the image to show it, and reviews the image itself (`check=True`: independent VLMs vote on it instead)
- [x] Domain rules for what an edit must keep from the source (`DOMAIN_CONSTRAINTS`), with a default for new domains
- [x] Image generators: OpenRouter, or a local Qwen-Image server (`generator="local"`)
- [x] Runs in Docker; resumable; your own target images instead of the agent's

## 3. Attacks and defenses

- [x] One call for every method: `redphys.attack(name, cases, output, targets=...)`, `redphys.defend(name, inputs, output)`
- [x] Perturbation, patch, diffusion generation and editing, typographic and prompt-injection attacks; the paper's 12 run on the benchmark
- [x] Removal, purification and sanitization defenses; the paper's 3 run on the benchmark
- [x] AdPO: a training-based defense that fine-tunes the victim's vision encoder on clean and adversarial pairs (Qwen3-VL, Cosmos-Reason2, InternVL3.5, Qwen3.5, Gemma 4)
- [x] Threat model: optimized on a shared CLIP surrogate ensemble (`surrogates="unified"`) or each method's own (`"default"`), transferred to the victim
- [x] The paper's settings in `configs/`, overridable by keyword or YAML; `manifest.json` records the config of every run
- [x] New methods: a `BaseAttack` or `BaseDefense` subclass, a registry line and a YAML

## 4. Victims and scoring

- [x] Any VLM behind an OpenAI-compatible API: vLLM for open-source models, OpenRouter for proprietary ones
- [x] Victim agents behind the same API, with a runnable example with tools (`docs/guides/victim-agent`)
- [x] `redphys.evaluate` collects answers; `redphys.score` judges them with Qwen3-8B on its own server; accuracy and ASR per attack in `eval_results.json`
- [x] Multiple-choice scoring (the exact wrong option is success) and captioning scoring (similarity to the target image's description, keyword match)
- [ ] Scoring by hazard: ASR per hazard type and weighted by severity (benchmark v2)
- [ ] Agents that act over several steps with tools, scored on the actions they take

## 5. Live leaderboard

- [x] Harness: attack, defense and model tracks, scored against each other
  - attack: mean ASR · defense: harmonic mean of 1 − ASR and clean accuracy · model: accuracy under attack
  - multiple-choice and captioning splits
- [x] Sandbox: submitted code runs with no network, read-only weights and an LLM proxy with a call budget; outputs checked (L∞ ≤ 16 for bounded attacks, options kept for text attacks, SSIM reported) before any victim sees them
- [x] Models that fail most answers marked invalid instead of ranked
- [ ] A submitter guide
- [ ] A hidden split: cases never released, drawn from benchmark v2
- [ ] Submissions from a repository or a Hugging Face model, rerun by us on our GPUs
- [ ] Live ranking: each new submission scored against the current attacks, defenses and models; the website updates after every run

## 6. Automatic red-teaming

- [x] The scenario agent: a target for every case without a person in the loop
- [ ] A weakness report: a model or a use case in → agents find or build scenes for it → integrate new attacks and defenses from papers, checked against their official code → run them → report the weakest scenes and attacks
- [ ] Agentic evolve: agents select and compose existing methods, co-evolve attacks and defenses round by round, and write new losses and purifiers in a sandbox; every new method tested on held-out victims
