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
  [--relevance-scores rel.jsonl | --wemm] [--vlm] --out runs/shadow1

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

**Relevance (WeMM).** `--wemm` scores Stage A on the WeMM embedding server. Its endpoint is read only from `$VAD_WEMM_URL` (never put it in config or git). The score is the max text–image cosine over each class's positive prompts (scale `wemm_text_cosine_v1`; not a probability), so the config must set `relevance.raw_scale: wemm_text_cosine_v1` with bands chosen for that scale, or the run is refused. Images are downscaled so the longest side is at most 448 px (`--wemm-max-side`) before sending. If the server fails for an image, its relevance is `unavailable` (never a score of 0), it goes to review, and the run prints a stderr warning with the failure count. The model is recorded as `@unverified` until you pass `--wemm-build <sha>`, which is then checked on every response.

`--vlm` uses LM Studio at `http://localhost:1234/v1` with `qwen3-vl-4b-instruct`. Change the model with `vlm.model` in the config.

## Tests

```bash
pixi run test                 # unit + acceptance (no network, no GPU)
pixi run test-lmstudio        # live check against local LM Studio (-rs: lists skips)
pixi run test-wemm            # live WeMM scorer, synthetic image (needs $VAD_WEMM_URL; skips without it)
pixi run test-gdino           # live Grounding DINO server (start it first; see its README.md)
pixi run test-e2e             # one shadow run with all three services + one outage case per service (synthetic images)
```

## Live prerequisites (not in the lockfile)

The live tests and `--wemm` / `--vlm` need assets that pixi does not install:

| Asset | Expected | How it is found |
|---|---|---|
| WeMM embedding server | MCP streamable-http server reporting `meta.backend=wemm`, `meta.model=tencent/WeMM-Embedding-4B`, `meta.dim=2560` | `$VAD_WEMM_URL` (set it in your shell or an untracked, gitignored file; never commit the address) |
| LM Studio | OpenAI-compatible server at `http://localhost:1234/v1` with model ID `qwen3-vl-4b-instruct` loaded | `vlm.base_url` / `vlm.model` in the config |
| GPU | CUDA-capable NVIDIA GPU for the Grounding DINO server (CUDA 12.8 torch in its own pixi workspace) | see `src/services/grounding_dino/README.md` |

```powershell
$env:VAD_WEMM_URL = "<your WeMM MCP endpoint>"                       # PowerShell
```

**Recording live results.** The live tasks **skip** (they do not fail) when their service is unreachable: `test-wemm` without `$VAD_WEMM_URL` or when the server cannot be reached, `test-gdino` when the server is not running, `test-lmstudio` when LM Studio is down or does not list the model, and `test-e2e` when any of the three is unreachable. A zero exit code alone therefore does not prove a service ran. Record the passed and skipped counts that pytest prints (the tasks pass `-rs`, so the skip reason is listed), plus the pixi environment.

## Layout

```
src/routing/        package `routing`: routing logic (imports no server code)
  schemas.py       Pydantic v2 contracts + JSON Schema export
  task_config.py   class registry, thresholds, budgets, feature flag, config hash
  scoring.py       Stage A (quality, relevance) and Stage B (box features, difficulty)
  interfaces.py    Detector / RelevanceScorer / VLMVerifier protocols and result types
  detection_mapping.py   raw boxes -> validated original-image detections
  verification_results.py  uncertain-result builder + VLM answer parser
  slicing.py       SAHI-style tiling, original-coordinate mapping, IoS union merge
  matching.py      class-aware matching, directional disagreement, provenance-preserving fusion
  policy.py        pure deterministic route-v1
  completeness.py  annotation completeness audit
  pipeline.py      bounded, resumable orchestration
  store.py         SQLite step/record/error store + per-output-dir job lock
  image_inputs.py  manifest + validated image loading
  evaluation.py    split isolation, frozen-set lock, route/cost report
  export.py        COCO-style versioned export + loader
  cli.py           vad-route
src/services/       package `services`: model-service clients and servers
  contracts.py      svc-1.0 wire models (detector server <-> client)
  http_detector.py  RemoteDetector (svc-1.0 HTTP client)
  wemm_mcp.py       WeMMMCPRelevanceScorer (WeMM MCP relevance client)
  lmstudio.py       LMStudioVerifier (VLM verifier client)
  replay.py         precomputed detections / relevance from JSONL
  runtime.py        file hashing + measured GPU time helpers
  grounding_dino/       Grounding DINO server; own pixi workspace (see its README.md)
tests/routing/, tests/services/
```
