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

# Detection metrics of a shadow run against ground-truth labels (see below)
pixi run vad-route evaluate --config ... --labels gt.json --trace runs/shadow1/route_trace.jsonl --out metrics.json
```

**Evaluation.** Labels are COCO JSON plus `vad_*` fields (format in `src/routing/metrics.py`): every image needs `vad_image_id` and `vad_exhaustive: true`; `ignore: true` marks an object that counts neither as found nor missed. `evaluate` reports detector mAP50–95 / mAP50 and recall by class and COCO size, false positives per image and per negative image, precision/recall of the accepted annotations (no scores, so no AP), missed objects split into failed images, images in review and silent misses, and the run's cost. A labelled image without a result is an operational failure with all its objects missed. Its `evidence_status` is `NOT A FINDING` unless the label file declares `"vad_label_provenance": "human"` at the top level; real-image evaluation waits for the human-labelled sample.

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

**VLM verifier.** `--vlm` calls an OpenAI-compatible Chat Completions endpoint at `vlm.base_url` with model `vlm.model`. There is no built-in endpoint or model: both must be set in the config whenever the VLM is enabled, or the run stops with exit code 2 (the example config uses local LM Studio, `http://localhost:1234/v1`, `qwen3-vl-4b-instruct`). The endpoint must accept image data URIs (`image_url` content parts) and `response_format` of type `json_schema`; a server that rejects either gives `uncertain` answers, and the request is not retried in another format. Each request is one HTTP attempt; any failure becomes `uncertain` and the box goes to review. The VLM only confirms or rejects detector boxes; it never draws one. If the endpoint needs a token, set `vlm.api_key_env` to the **name** of an environment variable that holds it (sent as `Authorization: Bearer`); the token is never written to config, traces or the run key, and a named but unset variable stops the run. The recorded VLM identity is `openai-compatible:<model>@<host:port/path>` (no credentials, no scheme), so the same model name on two endpoints never shares cached results. Because the scheme is not part of the identity and a model alias the server can repoint at the same endpoint is unverified provenance, use a canonical endpoint and a fixed model name for anything injected directly; a `vlm.base_url` with `user:password@` (or `token@`) is rejected, so put tokens in `vlm.api_key_env`. Migration (2026-11-10): this identity replaced `lmstudio:<model>`, and the config's LM Studio defaults were removed, so run keys changed once on purpose; results cached under the old keys in an existing `routes.sqlite` are left readable but are not reused.

## Tests

```bash
pixi run test                 # unit + acceptance (no network, no GPU)
pixi run test-vlm             # live VLM endpoint (VAD_VLM_URL / VAD_VLM_MODEL; default local LM Studio)
pixi run test-wemm            # live WeMM scorer, synthetic image (needs $VAD_WEMM_URL; skips without it)
pixi run test-gdino           # live Grounding DINO server (start it first; see its README.md)
pixi run test-e2e             # one shadow run with all three services + one outage case per service (synthetic images)
pixi run test-live-required   # release check: all live tests; an unreachable service FAILS instead of skipping
```

## Live prerequisites (not in the lockfile)

The live tests and `--wemm` / `--vlm` need assets that pixi does not install:

| Asset | Expected | How it is found |
|---|---|---|
| WeMM embedding server | MCP streamable-http server reporting `meta.backend=wemm`, `meta.model=tencent/WeMM-Embedding-4B`, `meta.dim=2560` | `$VAD_WEMM_URL` (set it in your shell or an untracked, gitignored file; never commit the address) |
| VLM endpoint | OpenAI-compatible Chat Completions server that accepts image data URIs and `json_schema` output; locally LM Studio at `http://localhost:1234/v1` with `qwen3-vl-4b-instruct` | runs: `vlm.base_url` / `vlm.model` (required when the VLM is enabled; / `vlm.api_key_env`) in the config; live tests: `VAD_VLM_URL` / `VAD_VLM_MODEL` (/ `VAD_VLM_API_KEY_ENV`), defaulting to the local example |
| Grounding DINO server | svc-1.0 server on a CUDA-capable NVIDIA GPU (CUDA 12.8 torch in its own pixi workspace) | runs: `--primary-url` and the `--primary-*` identity flags; live tests: `VAD_GDINO_URL` / `VAD_GDINO_MODEL_ID` / `VAD_GDINO_REVISION` / `VAD_GDINO_WEIGHTS_SHA256`, defaulting to the pinned local server (see `src/service_detector_gdino/README.md`) |

```powershell
$env:VAD_WEMM_URL = "<your WeMM MCP endpoint>"                       # PowerShell
```

**Recording live results.** A live test **skips** when its service is unreachable (`test-wemm` also when `$VAD_WEMM_URL` is unset; `test-e2e` when any of the three is unreachable), so a zero exit code alone does not prove a service ran. A service that is reachable must answer one real synthetic inference first; if that fails, the test **fails** rather than skips. For a release check use `pixi run test-live-required` (or set `VAD_LIVE_REQUIRED=1`), where an unreachable service also fails. Record the passed and skipped counts that pytest prints (the tasks pass `-rs`, so the skip reason is listed), plus the configured endpoints and models.

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
  replay.py        precomputed detections / relevance from JSONL (file replay, not a model service)
src/services_embedding/   package `services_embedding`
  client.py        WeMMMCPRelevanceScorer (WeMM MCP relevance client)
src/service_detector_vlm/ package `service_detector_vlm`
  openai_service_call.py  OpenAICompatibleVerifier (VLM verifier client; never authors boxes)
src/service_detector_gdino/  package `service_detector_gdino`; also its own pixi workspace (see its README.md)
  contracts.py     svc-1.0 wire models (detector server <-> client)
  client.py        RemoteDetector (svc-1.0 HTTP client)
  server.py        Grounding DINO server;  smoke.py  real-model checks
  runtime.py       file hashing + measured GPU time helpers
tests/routing/, tests/services/
```
