# vision_agentic_detector

Adaptive image-level routing for custom object detection. Phases 0–4 are implemented and **disabled by default** (`routing_enabled: false`); Phase 5 is started.

- Specification (requirements + acceptance criteria): [docs/specification.md](docs/specification.md)
- Project board (done, to do, missing, decisions, how it works): [docs/project_board.md](docs/project_board.md)
- Review notes: [docs/reviewer_suggestion.md](docs/reviewer_suggestion.md)

## Setup

Environments are managed with [pixi](https://pixi.sh) only (config in `pyproject.toml`, lock in `pixi.lock`). Tested with **pixi 0.62.2** on Windows 11.

**Prerequisite: install pixi** (once per machine), then open a new shell:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

```bash
# Linux
curl -fsSL https://pixi.sh/install.sh | sh
```

```bash
pixi install            # default env: Python 3.13 + package (editable) + pytest
pixi install -e clip    # adds open_clip + CUDA 12.8 torch for the CLIP scorer
pixi shell              # optional: activate the default env
```

Run any command inside an environment with `pixi run <cmd>` (e.g. `pixi run vad-route --help`).

## Commands

```bash
# JSON Schemas for all persisted contracts (committed under schemas/; a test fails on drift)
pixi run vad-route export-schema --out schemas

# Shadow mode: routes, verifies, audits; never discards. Enables routing for this run only.
pixi run vad-route shadow --config configs/routing/example.yaml --images path/to/manifest.jsonl \
  --primary-predictions preds.jsonl --primary-version mymodel@sha \
  [--secondary-predictions sec.jsonl --secondary-version other@sha] \
  [--relevance-scores rel.jsonl | --clip-weights open_clip_model.safetensors] [--vlm] --out runs/shadow1

# Legacy primary-only export (what runs when the flag is off)
pixi run vad-route legacy --config ... --images ... --primary-predictions ... --primary-version ... --out runs/legacy
```

PowerShell uses a backtick for line continuation:

```powershell
pixi run vad-route shadow --config configs/routing/example.yaml --images path\to\manifest.jsonl `
  --primary-predictions preds.jsonl --primary-version mymodel@sha --out runs\shadow1
```

**Remote primary detector** (shadow only; replaces `--primary-predictions` and `--primary-version`): `--primary-url` is an svc-1.0 detector server, and `--primary-model-id` / `--primary-revision` are required and checked against every response. `--primary-weights-sha256` (verified when given) and `--primary-timeout` (seconds, default 60) are optional. Any server failure sends that image to the error queue.

```powershell
pixi run vad-route shadow --config configs/routing/example.yaml --images path\to\manifest.jsonl `
  --primary-url http://localhost:8001 --primary-model-id grounding-dino-tiny --primary-revision <rev> --out runs\shadow1
```

**Precomputed inputs.** Every manifest image needs an explicit row in the predictions file; `"detections": []` is a valid zero-box result. If an image has no row, it goes to the error queue (`routing_failed_at:primary: MissingPrediction`) and the rest of the batch continues. `legacy` mode instead stops with an error. Duplicate or malformed rows are rejected when the file loads.

**Versions vs. content.** `--primary-version`, `--secondary-version` and `--relevance-version` are identifiers you supply. They are **not** verified hashes of a model. To stop stale cache reuse anyway, each precomputed file's SHA-256 goes into `RunInfo.*_artifact_digest` and therefore into the record key. If you edit a predictions file but keep the same version string, the run gets a new key and nothing is reused from `routes.sqlite`. The model weights behind the predictions are still not verified.

Shadow output: `routes.sqlite` (resumable step cache, records, error queue), `route_trace.jsonl`, `annotations.json`, `report.json`. A shadow run holds an exclusive lock on its `--out` directory; a second concurrent run on the same directory exits with code 2.

**Live relevance (CLIP).** `--clip-weights` scores Stage A with a local open_clip checkpoint (`pixi install -e clip`, then `pixi run -e clip vad-route shadow ...`). Its score is a prompt-contrast softmax in [0, 1] (`clip_prompt_softmax_v1`), not a cosine, so the config must set `relevance.raw_scale: clip_prompt_softmax_v1` and choose bands for that scale; otherwise the run is refused. See [docs/project_board.md](docs/project_board.md), "How it works and why".

`--vlm` uses LM Studio at `http://localhost:1234/v1` with `qwen3-vl-4b-instruct`. Change the model with `vlm.model` in the config.

## Tests

```bash
pixi run test                 # unit + acceptance (no network, no GPU)
pixi run test-lmstudio        # live check against local LM Studio (-rs: lists skips)
pixi run -e clip test-clip    # live CLIP scorer (-rs: lists skips)
pixi run -e clip gpu-check    # torch version, CUDA available, device names
```

## Live prerequisites (not in the lockfile)

The live tests and `--clip-weights` / `--vlm` need assets that pixi does not install:

| Asset | Expected | How it is found |
|---|---|---|
| CLIP checkpoint | Hugging Face `timm/vit_base_patch16_clip_224.openai`, file `open_clip_model.safetensors` (598,516,980 bytes, SHA-256 `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f`) | `VAD_CLIP_WEIGHTS` if set, else the Hugging Face cache. Never downloaded by this repo. |
| LM Studio | OpenAI-compatible server at `http://localhost:1234/v1` with model ID `qwen3-vl-4b-instruct` loaded | `vlm.base_url` / `vlm.model` in the config |
| GPU (optional) | CUDA-capable NVIDIA GPU for the `clip` env (CUDA 12.8 torch) | `pixi run -e clip gpu-check` |

```powershell
$env:VAD_CLIP_WEIGHTS = "C:\path\to\open_clip_model.safetensors"   # PowerShell
```

**Recording live results.** `test-clip` **skips** (it does not fail) when no weights are found, so a zero exit code alone does not prove CLIP ran. Record the passed and skipped counts that pytest prints (the tasks pass `-rs`, so the skip reason is listed), plus the pixi environment. `test-lmstudio` fails if the server or model is unavailable.

## Layout

```
src/vad/routing/
  schemas.py       Pydantic v2 contracts + JSON Schema export
  task_config.py   class registry, thresholds, budgets, feature flag, config hash
  scoring.py       Stage A (quality, relevance) and Stage B (box features, difficulty)
  adapters.py      Detector / relevance / VLM protocols; precomputed + LM Studio adapters
  embedder.py      optional open_clip relevance scorer + CUDA-event GPU timer
  slicing.py       SAHI-style tiling, original-coordinate mapping, IoS union merge
  matching.py      class-aware matching, directional disagreement, provenance-preserving fusion
  policy.py        pure deterministic route-v1
  completeness.py  annotation completeness audit
  pipeline.py      bounded, resumable orchestration (SQLite)
  evaluation.py    split isolation, frozen-set lock, route/cost report
  export.py        COCO-style versioned export + loader
  cli.py           vad-route
```
