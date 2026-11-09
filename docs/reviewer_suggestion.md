# Reviewer suggestions — adaptive object detection routing

## Naming and structure review — 2026-11-09

**Scope:** inspected adapter implementations, orchestration/persistence, service contracts, CLI wiring, Grounding DINO startup, packaging, and representative test imports. This is a maintainability review, not a complete correctness audit; no source files were changed or model services called. Earlier test counts below are historical and were not rerun for this review.

**Recommendation:** make a few responsibility-based extractions and clarify provider-specific names. Keep `vad.routing`, the existing protocol methods, CLI flags, and persisted identifiers stable. The following are proposals, not defects.

| Priority | Current → proposed | Implication and migration solution |
|---|---|---|
| First | `remote_adapters.py:104` `RemoteRelevanceScorer` → `WeMMMCPRelevanceScorer` | The implementation checks WeMM metadata, calls `compute_similarity`, and aggregates positive-prompt cosines; it is not a generic remote scorer. Update CLI/test imports and monkeypatch targets; retain the old class as an alias. Keep `wemm_text_cosine_v1`, model-version formatting and aggregation unchanged. |
| First | `pipeline.py:503` `_answers` → `_verification_matches_request`; `:508` `_apply` → `_candidate_status_after_verification` | The former checks request/class/candidate/region binding; the latter returns a status rather than mutating a candidate. Update the two internal callers at lines 467 and 487. Preserve cross-class conflict and `absent_rejects` behavior; run verification regression tests. |
| First | Persistence in `pipeline.py:75–157` → `store.py`; manifest/image loading at `:41–72` → `image_inputs.py` | SQLite, output locking and filesystem input obscure orchestration. Move existing definitions unchanged; keep compatibility re-exports in `pipeline.py` because CLI and multiple tests import them there. Check for import cycles, lock behavior, resume and corrupt-image handling. External consumers are unverified. |
| Next | Mixed `adapters.py`, `remote_adapters.py`, `embedder.py` → provider modules under `backends/`, with protocols in `interfaces.py` | Protocols, replay, HTTP/MCP and local GPU inference have different responsibilities. Extract protocols/result types first, then move implementations incrementally. Keep the old three modules as compatibility facades; retain lazy torch imports. Avoid replacing `adapters.py` with a same-named directory in the same step. |
| Next | `embedder.py:40–74` `file_sha256`, `GpuTimer`, `cuda_timer` → `vad/runtime/hashing.py` and `vad/runtime/gpu_timing.py` | Grounding DINO imports these helpers from the CLIP implementation (`server.py:93`). Shared runtime helpers remove that misleading dependency. Preserve lazy torch loading and `None` for unmeasured GPU time; update both consumers and retain old imports temporarily. No new dependency is needed. |
| Optional | `OpenClipRelevance` → `OpenClipRelevanceScorer`; `LMStudioVLM` → `LMStudioVerifier` | Makes the scorer/verifier roles explicit and consistent with protocols. Lower value than the changes above; combine with provider-module moves, not a separate sweeping rename. Keep aliases and all model/cache identifiers unchanged. |

**Proposed target (only existing responsibilities; reach incrementally):**

```text
src/vad/
  runtime/                 hashing.py, gpu_timing.py
  routing/
    pipeline.py            orchestration + compatibility re-exports
    store.py               RouteStore, job_lock, StoreLockedError
    image_inputs.py        ManifestEntry, load_manifest, load_image
    interfaces.py          Detector/RelevanceScorer/VLMVerifier contracts and result types
    detection_mapping.py   existing to_detections coordinate/class normalization
    verification_results.py  existing uncertain-result builder and answer parser
    backends/
      replay.py            precomputed detector and relevance implementations
      http_detector.py     existing svc-1.0 detector client
      wemm_mcp.py           existing WeMM client
      lmstudio.py           existing verifier client
      open_clip.py          existing optional local scorer
    adapters.py            compatibility facade
    remote_adapters.py     compatibility facade
    embedder.py            compatibility facade
    service_contracts.py   keep existing wire names and svc-1.0
third_party_service/        retain separate service environments and current path
```

**Keep as is:** `Detector.detect`, `RelevanceScorer.score`, `VLMVerifier.verify`, `RoutingPipeline`, `RouteStore`, and domain terms `iou`/`ios`. Keep `schemas.py`, `policy.py`, `matching.py`, `slicing.py`, and `task_config.py` until evidence justifies further splitting. Do not rename serialized `StageA`/`StageB`, `sahi`, schema versions or score-scale strings as part of a naming cleanup. Keep the user-selected `third_party_service/` directory: renaming it to `services/` brings little benefit and affects Pixi commands and documentation. The current Grounding DINO implementation uses Hugging Face Transformers, not TAO.

**Migration order:** internal helper names → store/input extraction → provider naming/modules → shared runtime helpers. Each step should preserve behavior and old imports. Validate current CLI entry points, schema drift, cache/replay, crop offsets, verifier binding and mocked service tests; run Grounding DINO's own suite in its separate environment after shared-helper moves. Update monkeypatch targets deliberately; compatibility aliases alone do not redirect a patched CLI binding. Compare test collection before/after so moving files does not silently drop tests.

**Documentation gap:** the older service proposal below now describes implemented files as future work. `RemoteDetector`, `RemoteRelevanceScorer`, service contracts and the Grounding DINO server exist, and the CLI wires both remote clients. The board's detector checklist/integration notes still say replay-only despite its F-35/F-36 entries. Reconcile these with the code, distinguishing implemented, mocked-tested, live-tested and real-data-benchmarked. Keep historical evidence, but move superseded proposals out of the active task list. This review does not independently verify reported live timings or deployment readiness.

## Earlier functional and environment review (historical)

**Reviewed:** 2026-11-09 · **Status:** phases 0–4 implemented for the greenfield repository; Phase 5 has utilities but no real-data calibration; Phase 6 has not started. Routing remains disabled by default. See [project_board.md](project_board.md) for the plan and project tasks.

**Verification (Pixi review, 2026-11-09):** `.pixi\envs\default\python.exe -m pytest -q` passed **92 tests, 2 deselected**. `pixi run --locked test` stopped before tests because this review sandbox could not persist a Pixi/uv interpreter-cache file (access denied). This is an environment limitation, not evidence of a repository defect. A fresh locked installation, Linux execution and the optional live tests were not verified in this review.

## Latest review: Pixi and project board

No new source-code defect was confirmed in the reviewed Pixi migration. `pyproject.toml` defines separate default/dev and CLIP environments; `pixi.lock` contains Windows and Linux resolutions and CUDA 12.8 torch; `.pixi/` is ignored. The previous P2 handoff is **fixed** in `76455b3`: omitted calibrated classes receive a `human_review` action and a structured suspected-class entry, with an enabled-VLM regression test.

Suggested additions, rather than confirmed runtime defects:

| Gap | Possible solution / completion evidence |
|---|---|
| Reproducibility is not checked on a clean machine or Linux; there is no CI workflow. | Record the tested Pixi version and add Windows/Linux CI using locked install/run commands. Run the default suite and a package-import smoke check; keep GPU and LM Studio checks separate. Verify that the lockfile remains unchanged. |
| Live model prerequisites are outside the lockfile. The CLIP test skips when no local weights exist, so a successful task exit alone does not prove CLIP ran. | Document the checkpoint source, expected digest, `VAD_CLIP_WEIGHTS`, LM Studio endpoint/model ID, and a GPU availability check. Record passed/skipped counts and environment with each live result. Add a required-assets preflight for release validation. |
| Fresh-shell command examples depend on activation, and the board assumes Pixi is already installed. | Add a short Pixi installation prerequisite; use `pixi run vad-route ...` consistently, and `pixi run -e clip ...` for CLIP. Include a PowerShell-compatible example for the Windows development setup. |
| The checklist marks “Rules 1–8” complete while domain-distance and its audit sampling remain absent (T-4). Persistence wording also implies all external calls are at-most-once, although only VLM calls have durable attempt markers. | Mark the rule group partial or enumerate completed rules. Describe detector resume as reuse of persisted results and VLM resume as at-most-once; retain the known crash-before-persist limitation for detector calls. |
| The full specification now exists only in Git history. The board references removed “Part B/C/D” concepts in older supporting material and can drift from the requirements. | Keep a stable, linked `docs/specification.md` for requirements and acceptance criteria, with the board tracking status only. Preserve the full baseline/always-expensive/adaptive benchmark and promotion gates already restored in B-4/B-5. |

The new T-7 (reviewer-outcome ingestion), T-8 (detection metrics) and T-9 (split enforcement) correctly expose previously missing workflows. Keep them open until exercised end to end, even though their supporting utility tests pass.

## Implemented and reviewed

| Area | Current implementation | Evidence / limit |
|---|---|---|
| Contracts and legacy path | Pydantic v2 records, exported JSON Schemas, stable IDs, validated class/box coordinates, feature flag off by default. | Schema drift and legacy-output tests pass. No existing detector repository or trainer was available to integrate. |
| Routing and detection | Separate relevance, difficulty and completeness; deterministic policy; primary prediction replay; optional secondary and native sliced inference; original-image box coordinates and provenance. | Acceptance tests cover zero boxes, disagreement, replay, coordinates and budgets. Slicing uses scripted crop-capable detectors, not a production adapter. |
| Verification and recovery | LM Studio VLM is verifier-only; uncertain responses go to review. SQLite caches steps, records an attempt before a VLM call, and makes an interrupted call `uncertain` on resume. | At-most-once call behavior trades a potentially lost answer for no repeat call. Live VLM testing was reported previously, not rerun in this review. |
| Export and evaluation utilities | Current-run-only annotation export, per-image error queue, frozen-set lock/isolation helpers, VLM-vs-human agreement scoring. | Frozen-set checks are utilities; the shadow CLI does not enforce split isolation. The export loader substitutes for a nonexistent trainer. |
| Calibrator guards | Incomplete raw or calibrated per-class output cannot produce a low relevance band; a missing calibrated class does not use raw thresholds. | Missing probabilities now trigger explicit human review. No fitted calibrator or human-labelled dataset exists yet. |

Earlier review findings are fixed and covered by regressions: SAHI parent provenance and per-tile budgets; cached Stage A and explicit prediction rows; current-run export isolation (R3-1); cross-class VLM review (R3-2); at-most-once recovery (R3-3); incomplete low-band guard (R3-4/R4-1); and blocking a bare human `complete` when evidence remains open (R3-5). The implementation and tests are in commits `f2dd581`–`6a31afd` and `tests/routing/test_r3.py`. An explicit per-finding human adjudication record remains to be designed if reviewers must resolve open findings in the application.

## Remaining work and possible solutions

### Proposed architecture: independent model services

**Recommendation (proposed, not implemented):** keep model serving in a top-level `third_party_service/` directory, outside the packaged `src/vad/` tree. The package owns routing, score calibration/combination, coordinate conversion, fusion, provenance, budgets and evaluation. Services own model loading and inference. The existing `Detector`, `RelevanceScorer` and `VLMVerifier` protocols are the integration boundary; extend them through HTTP clients instead of importing serving code.

```text
src/vad/routing/
  adapters.py                 existing protocols and replay adapters
  service_contracts.py        proposed versioned request/response models
  remote_adapters.py          proposed HTTP detector/relevance clients
third_party_service/
  README.md                   endpoints, prerequisites, startup and model revisions
  grounding_dino/             detector server, own environment/lock or container
  relevance/                 CLIP/SigLIP server, own environment/lock or container
  lm_studio/                 setup/config examples for the existing external VLM
schemas/services/            generated service JSON Schemas
tests/services/              contract tests using fake HTTP responses
```

Create only the services actually needed. Existing hosted APIs and LM Studio need client configuration, not another proxy server. Avoid a mandatory service gateway or importing `third_party_service` from the core package. Each local server runs as a separate process with its own dependencies; keep model weights and credentials outside Git. This isolates dependency conflicts, but does not increase GPU memory: start with sequential inference or server-side concurrency limits on the 12 GB GPU.

| Boundary | Required behavior |
|---|---|
| Detector API | Image/crop plus registry IDs and descriptions in; boxes, raw confidences and explicit coordinate metadata out. Prefer xyxy pixels relative to the submitted image/crop. The router applies the crop offset once. Distinguish a valid empty result from a failed/unprocessed item. Support batching for tiles. |
| Relevance API | Return per-class raw scores with scorer revision, prompt/config hash and score-scale identifier. Preserve each scorer separately; do not average incompatible scales. A scorer failure creates unknown evidence. Adding multiple scorers also requires extending the current single-scorer Stage A contract and a tested combination policy. |
| VLM API | Keep request/class/candidate/region binding and `present / absent / uncertain` verdicts. Provider-specific responses are normalized by the adapter. A VLM never directly authors an accepted box. |
| Identity and cache | Include schema version, request ID, image/crop digest, model revision and inference-config hash. Bind cache keys to these inputs and verify the response revision; do not trust a mutable endpoint alias as a model version. Mark unknown hosted-model provenance explicitly. |
| Transport and budgets | Configure endpoint, timeout, concurrency and credentials by environment reference. Bound payloads and batch sizes. Count network time, calls and service charges separately from optional measured GPU time. Propagate failures to the existing error/review paths. Retry only under a defined idempotency policy; preserve the VLM interrupted-request behavior. |
| Packaging and tests | Root Pixi remains lightweight; service dependencies stay in their own environment/container. Keep contract models authoritative in one place and export schemas for server validation. Core tests run without servers, torch or network. Integration tests check timeouts, malformed responses, model mismatch, empty detections, crop offsets and deterministic replay. |

**Incremental implementation:** first add the wire contracts and `RemoteDetector` with mocked contract tests; connect one separately served detector and run shadow mode; then add `RemoteRelevanceScorer`. Retain `LMStudioVLM` and replay adapters. Move the optional local CLIP runtime into a service only after behavior parity is demonstrated. Add multiple relevance sources and their calibration after this single-service path works. TAO Grounding DINO below is one possible backend, not a dependency of the routing package.

### Candidate for T-1: NVIDIA TAO Grounding DINO

**Recommendation:** trial pretrained Grounding DINO as the primary detector before adopting TAO for training. Its text-conditioned boxes suit the custom class registry; Qwen remains the verifier. Use descriptive prompts rather than opaque class IDs, and map results back to registry IDs. Recognition of unfamiliar parts still needs measurement on real images. TAO supports training, evaluation and export as later integration options ([model documentation](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/object_detection/grounding_dino.html)).

| Integration requirement | Proposed approach |
|---|---|
| Environment and hardware | Keep the router in Pixi and TAO in a separate Linux GPU container environment. NVIDIA documents a Linux host or remote Docker daemon. Validate the chosen release on the RTX 5070 (12 GB); compatibility and memory use alongside CLIP/Qwen are untested here ([TAO prerequisites](https://docs.nvidia.com/tao/tao-toolkit/latest/text/getting_started.html)). |
| Data formats | TAO Grounding DINO uses ODVG JSONL for training and COCO validation with contiguous category IDs starting at 0. Our COCO-style export starts at 1: add an explicit mapping/conversion. Its KITTI inference labels can first be converted to our prediction JSONL, including empty rows for processed images with no boxes. |
| Live adapter and slicing | Implement `Detector` with original-image xyxy boxes, raw scores, class mapping, weights digest and measured GPU time. Enable crops only after tile-edge and overlap tests pass; apply tile offsets exactly once. |
| Deployment optimization | Consider FP16 TensorRT only after establishing accuracy and resource baselines; TAO Deploy provides engine generation and inference ([deployment documentation](https://docs.nvidia.com/tao/tao-toolkit/latest/text/tao_deploy/grounding_dino.html)). |

**Trial sequence:** use a pretrained Swin-Tiny checkpoint on 30–50 representative development images; convert predictions and run shadow mode; compare with human labels for missed objects, false positives, latency and peak VRAM. If useful, build the live crop-capable adapter, then evaluate fine-tuning and TensorRT. This small trial is a feasibility check, not a substitute for the frozen benchmark or promotion gates. A later faster primary could use Grounding DINO as a secondary detector if the measured recovery justifies its cost.

### Other remaining work

1. **Real detector and real-image shadow:** choose a primary detector, add a crop-capable adapter with a weights digest and measured GPU time, and replay or run predictions on representative images. Add an independent secondary detector only if it provides useful recovery.
2. **Human data and calibration:** label a sample across classes, domains, sizes and negatives; split by source/near-duplicate group. Fit relevance and difficulty calibrators on calibration data, choose thresholds on development data, and record data/model hashes. Decide CLIP score bands from that sample; replayed JSONL scores currently have no scale guard.
3. **Frozen benchmark and promotion gates:** lock a held-out set; set numeric recall, false-negative, precision and cost targets before evaluation. Compare primary-only, always-expensive and adaptive routes on the same images, including low-relevance false-negative audits. Integrate split checks into the run workflow rather than relying only on helper functions.
4. **Trainer and rollout:** connect the annotation export to an actual training entry point, then run a canary by source group with a versioned rollback configuration. Keep automatic discard disabled until the frozen gates pass.
5. **VLM policy:** compare the 4B and 8B models against human labels on real crops before considering `vlm.absent_rejects: true`; keep the safer `false` default in the meantime.

**Suggested order:** verify clean locked environments and model prerequisites; collect real predictions and labels; implement reviewer ingestion, metrics and split enforcement; calibrate and benchmark; then consider canary rollout.
