# Reviewer suggestions — adaptive object detection routing

## Current code review — 2026-11-10, follow-up

**Assessment:** substantial progress on evaluation, sample preparation and human review. The previous two P1 findings are fixed. The new data-handling paths still have reproducible integrity defects; fix these before relying on fitted artifacts or frozen benchmark results. This section supersedes older status claims below.

**Scope and verification:** inspected current source and tests, including calibration code arriving during the review. The initial missing-module collection error was transient: `routing.calibration` and its CLI command subsequently appeared. Full CPU/mock suite results are recorded below; additional reproductions used synthetic labels/images and temporary files only. No live service or collected image was used. Only this review file was edited; no implementation changes or commits were made.

### Findings requiring action

#### R5-1 · P1 — merging reviews can silently change a chosen box's class

**Location:** `src/routing/sample.py:397–419`, `merge_reviews`.

**Evidence / implication:** each input label file correctly resolves its own category IDs, but the merged file copies the second pass's numeric `category_id` under the **first** pass's category table. Reproduced with reversed IDs: the adjudicator chose a `zz-custom.class_7` box; the merged result loaded it as `blue_widget_x9`. Independent labelling exports need not share numeric IDs. Alternative opinions have the same ambiguity.

**Fix / regression:** normalize selected and alternative annotations through source category ID → task class name → output category ID. Validate matching image dimensions before merging and validate the completed document. Add tests for swapped and disjoint numeric category IDs; choosing the second opinion must preserve its semantic class and geometry.

#### R5-2 · P1 — frozen evaluation does not bind predictions to frozen image content

**Location:** `src/routing/sample.py:256–281`, `make_split`; `src/routing/evaluation.py:95–119`; `src/routing/cli.py:179–206`, `_run_evaluate`.

**Evidence / implication:** grouping computes image SHA-256 values, but the split retains only a digest of the groups document; evaluation never compares a route's `image.sha256` with the original image. Replaced a white fixture with a red image of identical dimensions and the same explicit image ID: evaluation exited **0**, reported **`verified frozen_eval`**, and counted the replacement as a result. Locking labels and IDs alone cannot establish that competing runs used the same images.

**Fix / regression:** carry per-image content hashes into the split and bind the frozen subset's hashes/group membership into a versioned lock. Verify route hash, dimensions and task identity before scoring; apply the same content checks to fitting inputs. A changed image with the same ID must fail, while identical bytes at another path should pass. Treat old locks without content hashes as explicitly unverified rather than silently upgrading them.

#### R5-3 · P1 — contradictory human decisions can produce `complete`

**Location:** `src/routing/review.py:103–136`, `_review_one`; `src/routing/completeness.py:50–79`.

**Evidence / implication:** the conflict check covers `class_absent` versus `add_box`, but not accepted or reclassified model candidates. Reproduced `accept(A candidate)` + `class_absent(A)` + resolution of the other class + `complete`: the output was **complete and still exported class A**. The human record asserts both presence and absence, so completeness is not trustworthy.

**Fix / regression:** after applying effective candidate decisions, reject conflicts between `class_absent` and any accepted candidate of that class, including preaccepted and reclassified candidates. Keep the conflict unresolved with actionable IDs; never silently delete a box. Cover accept/absence, reclassify/absence and a valid reject/absence resolution.

#### R5-4 · P2 — duplicate COCO IDs silently remove or mislabel images

**Location:** `src/routing/metrics.py:74–108`, `load_ground_truth`.

**Evidence / implication:** `vad_image_id` uniqueness is checked, but numeric COCO image IDs are used as dictionary keys without a uniqueness check. Two distinct images with the same numeric ID loaded as **one image**, and the first image's annotation moved to the second. This changes evaluation denominators, labels and downstream split/calibration inputs without an error.

**Fix / regression:** validate unique image/category/annotation IDs before constructing lookup maps, and an unambiguous category-name mapping. Raise `LabelError` naming the duplicate. Add the two-image collision fixture and ensure no image or annotation can be silently reassigned. Distinct numeric IDs with valid references must continue to load.

#### R5-5 · P2 — repeated trace rows bypass calibration's sample minimum

**Location:** `src/routing/calibration.py:87–102`, `fit_relevance_rows`; `:117–146`, `fit_relevance`.

**Evidence / implication:** repeating one positive and one negative record ten times each fits successfully and reports **10 positives / 10 negatives from only two unique images**. Concatenated or retried traces therefore change sample weighting and bypass the advertised minimum. `metrics.image_results` also silently chooses the last row for a repeated image ID, while cost reporting counts every row.

**Fix / regression:** require one selected observation per image per fit/evaluation arm. Either reject duplicates, or explicitly deduplicate byte-identical records and reject conflicting records for the same image. Count unique images after that check and retain source groups for later uncertainty analysis. Repeated two-image input must fail the minimum; mixed runs must not be resolved by file order.

#### R5-6 · P2 — calibration artifacts can claim a score scale the trace never used

**Location:** `src/routing/calibration.py:120–129`, `fit_relevance`.

**Evidence / implication:** fitting checks task and scorer name, but takes `scale` from the supplied config without matching the trace's run config. A trace generated with `test_scale_v1` fitted successfully when only the supplied scale was changed to `unrelated_scale_v2`. The artifact then advertised the new scale. A future loader checking that field would trust incorrect provenance.

**Fix / regression:** validate the originating run configuration, accounting explicitly for shadow/CLI overrides, or persist and verify scoring provenance directly in the run. Bind scale, scorer/build, prompts and preprocessing to the observations. Reject changed-scale and incompatible mixed-run inputs; preserve valid fits from the original shadow config.

#### R5-7 · P2 — frozen accuracy and cost reports use different image populations

**Location:** `src/routing/cli.py:200–206`, `_run_evaluate`.

**Evidence / implication:** ground truth is filtered to frozen IDs, but `route_report(records)` receives the entire trace. A trace containing one frozen and one development result reported **one frozen result but two cost images**. Development latency, review decisions and VLM calls can therefore alter a frozen accuracy/cost comparison.

**Fix / regression:** select and validate the evaluation cohort once; use it for detection, annotation and cost metrics. Report out-of-cohort results separately, and retain labelled images without results as operational failures. Adding a slow development record must leave every frozen metric unchanged.

### What is done, and what remains

| Area | Current assessment / next action |
|---|---|
| Previous review fixes | **Verified in source and passing regressions:** dedicated slicer model/digest included in run identity, legacy records still load; VLM URL userinfo rejected without echoing credentials; pipeline docstring now states detector replay and VLM recovery accurately. |
| Service architecture | Three flat integration packages and generic OpenAI-compatible verification remain in place. No further package restructuring is needed for these fixes. Mutable VLM aliases and WeMM without a verified build remain provenance limits. |
| A2 / A3 / A7 tooling | Metrics CLI, grouping/exclusions/splits, optional split-aware exports, second-review merge and persisted adjudication/re-audit now exist. They are **implemented with corrections required above**, not missing wholesale. The mock suite does not establish target-domain accuracy. |
| A5 calibration | Logistic fitting, coherent any-target probabilities, count guards, artifact output and `fit-relevance` now exist. Runtime artifact loading/compatibility validation, CLI routing integration and held-out Brier/ECE/reliability reports remain to be completed; address R5-5/R5-6 first. |
| Real-data and rollout work | Human labels/permissions, enforced transfer policy (D-12), mandatory split provenance for training exports (D-14), difficulty validation, VLM human-label study, frozen ablations/numeric gates and trainer integration remain open. Do not run collected images against services as part of these fixes. |

**Documentation corrections for the implementation agent:** reconcile the board's Phase 4/5 rows, T-7/T-8/T-9/T-11 and checklist with F-49–F-52; replace old `services.*` paths and “CLIP code kept” claims. Update specification statements that reviewer ingestion and detection metrics are absent. The board's promise of `NOT A FINDING` without a verified split is not enforced by `evidence_status(label_doc)`, which checks human provenance only: make evidence eligibility depend on both, while retaining descriptive metrics for unverified runs. Keep one current status table; the research work orders below remain useful acceptance criteria but their presence/missing table is historical.

**Suggested implementation order:** R5-1/R5-4 label integrity → R5-3 adjudication consistency → R5-2 frozen content binding → R5-5/R5-6 fitting provenance → R5-7 cohort reporting. Use small validation helpers at input boundaries and focused regression tests; no new service, framework or broad refactor is necessary. Record verified fixes here and refresh board status after the full CPU/mock suite passes.

**Validation:** `.\.pixi\envs\default\python.exe -B -m pytest -q -p no:cacheprovider` → **436 passed, 7 live tests deselected, 1 Starlette deprecation warning**, in 19.02 seconds. FastAPI mock tests ran with local socket access. All seven findings above were reproduced locally outside the existing suite; a green suite currently does not cover these cases.

## Historical research and cleanup notes

The sections below preserve earlier reasoning and work orders. Use the current review above for implementation status; dated claims that metrics, split enforcement, review ingestion or fitting do not exist are superseded.

## Research evidence and agent action plan — 2026-11-09

**Assessment:** components have research support; the complete workflow and its exact numerical rules are not scientifically validated yet. Use the reference IDs and code mapping in [specification.md](specification.md#research-references-and-implementation-mapping). R1–R9 distinguish published methods, backend documentation and related work. Do not describe the router as an implementation of ALWOD, POPE or learned deferral; do not transfer their reported gains to this project.

**Review boundary:** inspected the current working tree, including ongoing CLIP-removal edits; service packages have not yet moved to the proposed flat layout. These are evidence gaps and work instructions, not newly reproduced runtime defects. No inference, dataset upload, model training or new test run was performed for this documentation task. Previous test counts below remain historical.

### Present versus missing

| Area | Present in source | Missing evidence or implementation / why it matters |
|---|---|---|
| Quality (R1) | Laplacian-based blur proxy, extreme grayscale fraction, resolution; quality-based audit sampling. | Validation against actual detector failures. A sharp textureless scene or white background can trigger poor-quality signals without being a bad detection input. Compression/domain-distance remain absent. |
| Relevance (R2) | WeMM MCP client, data-URI image transport, max-positive-prompt cosine, scale guard, unknown-on-error and incomplete-class guards. | Labelled threshold selection, fitted probabilities, server build provenance; negatives/reference images are not part of the current score. Cosine ranking is not a calibrated probability of target presence. |
| Detection/slicing (R3, R4) | Pinned GDINO model, per-class inference, remote contracts, native slicing, parent provenance, per-tile caps and mocked regressions. | Real-image recall/precision and resource comparisons. Per-class attribution is deterministic bookkeeping, not proof that the class is present. Synthetic detections cannot establish accuracy. |
| Difficulty (R5, R7) | Weighted predicted-box features and optional detector disagreement; unknown when no boxes/confidence; pre/final snapshots. | An explicit labelled failure target, validated ranking, trained calibrator or justified weights. Secondary agreement can reflect correlated mistakes; high confidence does not establish good localization. |
| Calibration (R6, R7) | Relevance protocol, range/consistency guards, placeholder thresholds. | Fitting/loading workflow, reliability metrics, artifact/data manifests. Detector calibration, relevance calibration and difficulty calibration are three different tasks; one scalar calibration method does not solve all of them. |
| VLM (R8) | Structured verifier, request binding, uncertain fallback, durable attempts and human-agreement utility with Wilson bounds. | Human crop/image presence labels, adversarial/confusable negatives, prompt/crop sensitivity and evidence for accepting/rejecting boxes. These checks also do not prove completeness. |
| Review/data (R9) | Human-review reasons, audit helper inputs, split/frozen-lock utilities. | Reviewer ingestion, per-finding adjudication and enforced split-aware run paths. Utilities tested alone do not prevent leakage in an actual workflow. |
| Whole-system outcome | Route/cost reports, current-run exports and regression fixtures. | Detection metrics, controlled ablations, held-out promotion results, actual trainer integration and incomplete-label handling. Passing unit tests establishes behavior, not a measured accuracy gain. |

### Agent work orders and acceptance evidence

**A1 — Pin behavior and provenance before extending scores (implementation agent; T-13/T-14).** Finish the requested service cleanup independently of scoring changes. Preserve feature flags, unknown/empty/error distinctions, coordinate transforms and at-most-once VLM recovery. Keep a manifest of code revision, model/weights/build, prompts, preprocessing dimensions, thresholds and policy. Update research-to-code links after package moves. Output: clean imports/tasks, reviewed identity migration and passing relevant existing regressions. Do not change numeric behavior simply to make a research citation appear closer to the implementation.

**A2 — Prepare ground truth and enforce splits (data owner + implementation agent; T-2/T-9/T-11).** Human experts label all target boxes and class absence, with second review for ambiguous cases and a sample of others. Record occlusion/ignore and non-photo exclusion rules before evaluation. Dedupe/group by source; freeze the evaluation manifest/annotations before tuning. Divide the tuning portion into calibration and development groups, keeping every group in one split. Connect isolation checks to fitting/evaluation/export entry points rather than leaving them only in utility tests. Output: versioned group/split manifests, image permissions, adjudicated labels, lock hashes and a demonstrated rejection of cross-split leakage. Without labels or transfer approval, implement the tooling with fixtures and stop dependent real-image runs.

**A3 — Define and implement metrics first (evaluation agent; T-8).** Fix class-aware one-to-one matching, IoU thresholds, size strata, ignored objects and duplicate treatment before comparing methods. Compute mAP50–95 using a standard evaluator or verify against one; add recall, FP/image, annotation precision, missed-instance counts, review coverage and pipeline errors/cost. Treat a failed image as a reported operational outcome; do not silently drop it to improve metrics. Output: a fixture with known TP/FP/FN outcomes, metric agreement checks and a reproducible evaluation command. Preserve model scores needed for ranking; explicitly document any score-free final-annotation metric rather than inventing confidence values for mAP.

**A4 — Test quality and difficulty as predictors (analysis agent, after A2/A3).** Define primary failure on target-positive images as at least one missed or insufficiently localized instance under the agreed matching criterion. Evaluate negative-image false positives separately. Compare existing difficulty against simple baselines (minimum confidence alone, small-box signal alone, and random selection) at equal review/escalation budgets. Report failure capture versus coverage, precision-recall/ranking metrics, and source/size breakdowns with group-based confidence intervals. Ablate blur, crowding, ambiguity and disagreement separately. Output: analysis showing which terms add predictive value, including failures on plain/bright/dark/blurred scenes. Change weights or remove terms only from calibration/development evidence, with a separately reviewed policy change. Do not equate a normalized weighted sum with a probability.

**A5 — Fit relevance first; add other calibration only with sufficient labels (calibration agent; B-2/B-3).** Use class presence and `any target present` labels from A2. Compare the current max-cosine rule with a simple fitted baseline on calibration groups; choose prompts/bands on development groups. Report Brier score, reliability curves, ECE with stated binning and sample counts, and low-relevance false-negative risk. Explicitly define image-level aggregation and validate coherent per-class/any-target outputs; never sum class probabilities assuming independence. Save scorer scale/build, prompt/preprocessing hashes, fit data hashes and parameters together; refuse mismatched artifacts. Output: a loadable relevance calibrator, guard tests and development report. Temperature scaling in R6 is not automatic justification for transforming arbitrary cosine values. Fitting detector correctness or image difficulty is a separate proposal with its own labels and runtime path.

**A6 — Validate the VLM on human labels (analysis agent; B-6).** Hold detector boxes fixed and label candidate crops plus full-image missing-class questions, including absent/confusable classes. Adapt the object-presence evaluation idea from R8 without claiming POPE reproduction. Run `vlm_agreement` against human labels, reporting present/absent precision, Wilson lower bounds, uncertain/timeout rates and counts by class, size and purpose. Examine crop padding and prompt sensitivity on development data. Output: labelled question set with crop/question identity and model revision, report and error examples. Set numeric acceptance/rejection targets before evaluation; retain `absent_rejects: false` until evidence meets them. Neither `present` nor `absent` can author geometry or clear conflicting/unadjudicated evidence.

**A7 — Close the human-review loop (implementation agent + reviewer; T-7).** Add persisted per-finding accept/reject/class/completeness adjudication, reviewer identity, source candidate references and reasons, then re-audit the image. Keep model suggestions distinct from human labels; prevent frozen-set corrections entering fitting/training. Output: a replayable correction record and an end-to-end example where explicit adjudication closes findings while a bare human `complete` cannot. Evaluate annotation time and residual errors when real reviewers are available. R9 motivates deferral but does not prove that our current queue improves outcomes.

**A8 — Run controlled ablations and promotion gates (evaluation agent; B-4/B-5, after A2–A6).** On identical frozen groups compare primary-only, primary+slicing, primary+VLM, always-expensive and adaptive. Within each arm use fixed model artifacts, prompts and thresholds; explain any inability to run an independent secondary. Report both detector outputs and post-verification accepted annotations; all review, timeout and failed cases remain visible. Measure recall/precision, small-object recovery, negative FP/image, completeness, review workload and latency/GPU/calls. Use paired source-group uncertainty estimates. Predeclare primary metric and numeric recall/FN/precision/cost gates before exposing frozen results. Output: machine-readable results, run manifests, reproducible command and pass/fail decision. Training-benefit claims require a later matched training experiment; do not promote solely because the synthetic pipeline runs.

**Dependency order:** A1 can proceed with fixture-based A2/A3 tooling. Human labels and permissions gate real-image A4–A6; A8 waits for stable metrics and development decisions. A7 can be built separately with fixtures. Keep these as small reviewed changes tied to existing board cards; do not add all research methods or dependencies merely because they are cited. An analysis that rejects a heuristic or leaves the current policy unchanged is a valid result.

## Cleanup plan: three model integrations — 2026-11-09

**Requested target:** flatten the integrations directly under `src/` as `services_embedding`, `service_detector_vlm`, and `service_detector_gdino`, alongside `routing`. Remove the enclosing `services` package, OpenCLIP support and LM Studio-specific client naming. Use a generic OpenAI-compatible VLM call. This section supersedes earlier nested layouts and the recommendation to retain parked CLIP code. Only this review file is changed; preserve concurrent implementation and documentation work.

### Minimal target layout (proposed)

```text
src/services_embedding/
  __init__.py
  client.py                   existing WeMMMCPRelevanceScorer; remote MCP only
src/service_detector_vlm/
  __init__.py
  openai_service_call.py      OpenAICompatibleVerifier; existing httpx transport
src/service_detector_gdino/
  __init__.py
  client.py                   existing RemoteDetector
  contracts.py                existing svc-1.0 detector wire models
  server.py, runtime.py       existing Grounding DINO server + GPU/hash helpers
  smoke.py, README.md
  pixi.toml, pixi.lock         separate GPU server environment
src/routing/
  replay.py                   moved file-based replay adapters
  ...                         existing routing/domain modules
```

These packages sit one level below `src/`; files sit directly inside them. Names follow the user's requested spelling, including plural `services_embedding`. Keep WeMM on the existing external embedding service. `service_detector_vlm` remains a **verifier** implementing `VLMVerifier.verify`; its folder name must not imply that it now generates boxes. Keep request binding, `present/absent/uncertain`, and detector-authored geometry. No new gateway, model registry, base client hierarchy or mandatory OpenAI SDK is needed.

**Packaging/path changes required by flattening:**

- Set Hatch wheel packages to `src/routing`, `src/services_embedding`, `src/service_detector_vlm`, and `src/service_detector_gdino`; remove `src/services` after all callers move. These remain import packages in the existing distribution, not four independently published distributions.
- Use imports such as `from services_embedding.client import WeMMMCPRelevanceScorer`, `from service_detector_vlm.openai_service_call import OpenAICompatibleVerifier`, and `from service_detector_gdino.client import RemoteDetector`. The server imports its contracts/runtime from `service_detector_gdino`.
- Grounding DINO startup becomes `pixi run -m src/service_detector_gdino/pixi.toml serve`. Its editable root dependency changes from `../../..` to `../..`. If tests retain their current location, its test task changes to `pytest -q ../../tests/services/grounding_dino`. Update detached-environment configuration, lockfiles, smoke commands and README paths together.
- Update CI import smoke checks, service-resolution tests and all source links. `tests/services` can remain an organizational test folder; preserve `tests/__init__.py` and relevant package markers. Keep the `vad-route` command and persisted identifiers stable except for the deliberate VLM identity migration below.

### Removal and migration instructions

| Step | Existing code or configuration | Instruction / completion evidence |
|---|---|---|
| 1. Remove OpenCLIP as one coherent change | `src/services/open_clip.py`, `tests/services/test_clip_live.py`; `routing/cli.py` CLIP parser/import/construction branches | Delete the scorer and its dedicated live test. Remove `--clip-weights`, `--clip-arch` and their scale-check branch. Keep the generic pipeline score-scale guard and replay support. Confirm obsolete CLI flags fail clearly instead of being silently ignored. |
| 2. Remove its environment footprint | `pyproject.toml`: `clip` extra, pytest marker/exclusion, Pixi feature/index override, environment, `test-clip`, and root clip-only `gpu-check` | Remove those entries and regenerate locks with Pixi, including the service lock if the editable root dependency metadata changes. Do not hand-delete lock entries: shared transitive dependencies can still be required. Keep torch/transformers/CUDA dependencies in the Grounding DINO workspace. The root development environment still needs FastAPI for fake-server tests. |
| 3. Generalize the VLM client | `services/lmstudio.py:LMStudioVerifier` → `service_detector_vlm/openai_service_call.py:OpenAICompatibleVerifier` | Retain the current multimodal Chat Completions request and response parser. Make base URL/model explicit deployment configuration; keep endpoint/model examples for the current local installation in docs. Support optional authorization via an environment-variable reference; do not put secrets in YAML, traces or cache identity. Update CLI construction, acceptance tests, golden identity tests and all imports. |
| 4. Define the supported compatibility contract | Existing VLM JSON-schema request and LM Studio assumptions | Document that the endpoint must accept image data URIs and the chosen structured-output format. Keep one HTTP attempt and uncertain-on-failure. If another server needs a different response format, select it explicitly in configuration; do not silently retry a rejected schema request. `/models` is not universally available or proof that vision/schema inference works, so use an actual synthetic inference for the live readiness check. |
| 5. Preserve cache correctness during generalization | Current `model_version` returns `lmstudio:{model}`; `test_run_identity.py` asserts that value and its resulting record key | A module/class move alone should preserve identity. Replacing provider-specific identity needs an explicit migration: record an endpoint deployment/revision identifier with the model, intentionally produce new keys, and update the golden expectation with a migration note. Do not reuse an old result merely because two endpoints advertise the same model name. Keep old SQLite runs readable; no automatic cache deletion. |
| 6. Flatten the remaining clients | `wemm_mcp.py`, `http_detector.py`, `contracts.py`, `grounding_dino/` | Move directly to the three top-level integration packages above; remove the enclosing `src/services` once all imports and packaging entries are updated. Apply the relative Pixi path corrections above. Keep `WeMMMCPRelevanceScorer` provider-specific because it validates WeMM metadata. Preserve `wemm_text_cosine_v1`, `svc-1.0`, coordinate conversion and detector artifact checks. |
| 7. Remove single-consumer support modules from the service root | `services/runtime.py`, `services/replay.py` | After CLIP removal, move GPU timing/hashing beside Grounding DINO, updating its server import; retain both helpers because they are used. Move replay adapters to `routing/replay.py`: replay is local file I/O, not a fourth model service. Update test imports. Do not delete replay/legacy behavior during this service cleanup. |
| 8. Remove remaining names and stale instructions | README; board D-5/T-10, F-25/F-39, integration/hardware/run sections; specification layout/CLIP examples; Grounding DINO README; test task names | Remove active CLIP setup, weights path/hash, extra VRAM estimate, score calibration advice and commands. Replace client-level LM Studio wording with OpenAI-compatible VLM, leaving LM Studio as an optional deployment example. Rename `test_lmstudio_live.py`, the `test-lmstudio` task and the `lmstudio` marker to VLM equivalents and update default marker exclusions. Preserve historical results explicitly as historical. |

### Other dangling-code checks

- **No additional unused production imports were confirmed** by a lightweight AST scan. The reported `routing/__init__.py` names are public re-exports; keep them. Do not delete symbols from this scan without checking import/discovery behavior.
- `SliceResult.raw_count`, `RouteStore.records/errors`, frozen-set helpers and the annotation loader have test callers. Frozen-set enforcement and reviewer ingestion are incomplete workflows, not evidence that their helper code is dead. Keep these functions until their requirements are explicitly retired or integrated.
- The new `tests/services/live.py` and `test_three_services_live.py` are active, uncommitted integration work. Rename `require_lmstudio` and update all three-service tests to use endpoint/model configuration. Fix `_port_open`'s default port selection: an HTTPS URL without an explicit port currently probes 80; select 443 for HTTPS. Keep reachability skips distinct from successful inference, and offer a required-services validation mode that fails when a release check cannot run.
- `tests/services/test_clip_live.py`'s weight discovery, CLIP background prompts, text cache, and CLIP-specific GPU/wall timing fields can disappear with the scorer. Do not generalize or retain these as unused scaffolding.
- Empty `__init__.py` files establish package boundaries and avoid the tested `tests.services` shadowing problem; retain them. Keep negative prompts in the task schema during this cleanup: they are part of existing task/config contracts, although WeMM currently ignores them. Removing that field is a separate schema/config migration.

### Implementation order and checks

Remove CLIP and regenerate locks → generalize VLM transport/config and decide its identity migration → move the three integrations/support modules → update tests/tasks/docs. Make each stage importable and testable before the next; avoid mixing this with the larger pipeline refactor proposed below.

Run default CPU/mock tests (including fake Grounding DINO endpoints), schema drift and run-identity tests, and a locked installation/import smoke check. Re-run the service workspace's fake-model suite after paths change. With synthetic images, explicitly exercise embedding, generic VLM and GDINO, then all three together; record passes/skips and the configured models/revisions. Search active source/tests/config/tasks for `open_clip`, `OpenClip`, `VAD_CLIP`, `--clip-`, `test-clip`, `LMStudioVerifier`, and old module paths; remaining matches need a deliberate historical/deployment reason. This plan is based on code/reference inspection, not a new test run.

## Simplicity and readability review — 2026-11-09

**Recommendation:** retain the current `routing` / `services` split and simplify inside the larger functions. Optimize for following one image from input to output and changing one behavior in one place. More folders, base classes, generic runners or compressed expressions would add little value now.

**Scope/evidence:** read pipeline, CLI, policy, completeness, interfaces, persistence, slicing and the WeMM client at `f976203`; counted function spans with Python AST. These are maintainability proposals, not new correctness findings. No source changes or new test runs were needed for this review; the previous verification below remains the functional baseline.

| Order | Location / difficulty | Smallest useful change | Behavior to preserve / check |
|---|---|---|---|
| 1 | `routing/pipeline.py:137` `_route_image` (147 lines): caching, detector normalization, escalation, budgets and record assembly share nested state. | First extract `_load_or_compute_stage_a` and `_run_cached_detector`; reuse the latter for primary/secondary normalization while keeping their different failure handling explicit. Then extract the existing sliced-inference block into `_run_sliced_detection`. Keep helpers private in the same file. Aim for a top-level sequence a reader can scan, without introducing a workflow engine. | Keep step/cache keys, persistence order, crop offsets, raw tile provenance and partial-result review fallback. Primary failures still queue the image; secondary failures still permit review. Use existing resume/budget/coordinate regressions after each extraction. |
| 2 | `routing/cli.py:72` `main` (90 lines): validates arguments, chooses providers, runs legacy/shadow and writes outputs. | Extract `_validate_args`, `_build_relevance_scorer`, `_run_legacy`, `_run_shadow` and, if still useful, `_write_shadow_outputs`. Keep direct `if` branches for provider selection. `main` should read as parse → validate → load → dispatch. Leave parser declarations together. | Preserve flags, defaults, exit codes, environment lookup, lazy CLIP imports, current-run-only exports and stderr scorer warnings. Existing CLI tests should still patch the provider symbol actually used. |
| 3 | `routing/pipeline.py:320` `_verify`: nested `ask` combines external inference with durable attempt/recovery handling; several arguments/results lack types. | Move the existing request operation to `_verify_request(image, key, request, class_description)`. Keep call counting and budget checks visible in `_verify`. Add parameter/return annotations to `_verify`, `_candidates`, and the detector cache callback. Use a private `VerificationOutcome` dataclass for the four returned values if it improves the call site. | Persist the attempt before sending, never resend an interrupted VLM call, validate request binding, preserve cached-call accounting and cross-class review. Keep existing crash/recovery tests. |
| 4 | `services/wemm_mcp.py:61` `_call`: retrying HTTP, SSE/JSON decoding, MCP errors and model metadata checks occur together. | Separate transport/retry from response decoding and metadata validation with two or three private functions. Parse matches in one explicit loop that validates and inserts each index, rather than constructing `pairs`, converting to `dict`, then comparing lengths. Keep a WeMM-specific client. | Preserve one retry on transport failure only, exact row/index checks, finite scores, build checks and real zero scores. Reuse current malformed-response tests; do not add another generic HTTP abstraction. |
| 5 | Dense local expressions across pipeline/policy: `sa`, `g`, `w`, `dis`, chained assignments and nested conditional expressions make changes harder to review. | In touched functions use names such as `stage_a_result`, `gpu_ms`, `detector_warnings`, `disagreement_result`. Replace nested ternaries and assignments of unrelated attributes with ordinary branches/assignments. Keep conventional geometry abbreviations and short loop variables. | This is local clarity work; retain serialized names, reason codes, model identities and score scales. Avoid a repository-wide rename commit. |

**Keep simple code as it is:** `policy.route` is an ordered, pure rule function; keep rule precedence explicit rather than building a registry or generic rule engine. `completeness.audit` has one coherent responsibility; only extract its final status decision if it can become a small pure helper with clear inputs. Keep `RouteStore`, Pydantic boundary validation, provider protocols, explicit error-versus-empty handling and lazy model imports. They protect real requirements even when they add lines. Leave the parked CLIP implementation in place until removing support is an explicit product decision.

**Documentation cleanup:** this file now repeats superseded proposals and test snapshots. Replace active guidance with a short current table and retain prior reviews in a clearly labelled archive section or document when authorized. Correct `pipeline.py`'s opening claim that a restart “never repeats a detector” call: a crash before persistence can repeat one; VLM attempt markers provide the stronger guarantee. Also update the server test docstring that still says the root environment has no FastAPI.

**Suggested implementation sequence:** CLI extraction → one pipeline extraction per change → verification request/result typing → WeMM parsing cleanup. Validate output records, reason/action order, exported annotations and golden run keys against existing fixtures; keep clock-dependent assertions meaningful rather than expecting identical elapsed times. Finish with the existing CPU/mock suite, including server tests with local socket access. Success means fewer responsibilities and less hidden state per function, with the same observable behavior; total line count may stay similar.

## Follow-up review — 2026-11-09 (`f976203`)

**Result:** no new source-code defect confirmed in commits `26dc4d6`–`f976203`. WeMM now rejects duplicate/missing/extra result rows and indices; `services.contracts` no longer imports `routing`; FastAPI is in the root dev environment, so the six Grounding DINO server tests are collected by the default suite. The board records the trial-image rules and the remaining single-wheel decision.

**Verification:** `.pixi\envs\default\python.exe -m pytest -q --ignore=tests/services/grounding_dino/test_server.py` → **162 passed, 4 deselected**. `tests/services/grounding_dino/test_server.py -q` → **6 passed, 1 Starlette deprecation warning** when run with local socket access. Together this matches the board's 168 passing tests; the full suite hung in the review sandbox while `TestClient` created a Windows socketpair, so I did not claim a full in-sandbox pass. Live GPU/LM Studio/WeMM checks were not rerun.

**Resolved from the previous review:** duplicate/extra WeMM rows (26dc4d6), direct contract import from routing (985ff9c), and skipped server contract tests in the default environment (a04a7fe). These rows in the earlier section below are historical, not open findings.

**Still worth deciding:** keep one `vad` wheel for `routing` and `services` while both deploy together (D-10); split out/version `svc-1.0` only when independent upgrades are required. Before promotion, require a server-reported WeMM build and measured thresholds on human-labelled images; the current scorer uses positive prompts only. Keep the image-permission gate from T-2. Update or archive the old `src/vad` / `third_party_service` proposals below to avoid a misleading active architecture.

## Earlier service split review — partially resolved

**Verified:** clean worktree at `045d525`; `.pixi\envs\default\python.exe -m pytest -q` → **152 passed, 1 skipped, 4 deselected**. This is the local CPU/mock suite; the skipped FastAPI server test and four live model tests were not independently run here. The old `RemoteRelevanceScorer` CLI patch concern was checked and is **resolved**: `tests/services/test_wemm_mcp.py` patches `WeMMMCPRelevanceScorer`.

**Implemented since the earlier review:** `routing` now owns orchestration, inputs, store, interfaces and box normalization; `services` holds replay, HTTP detector, WeMM MCP, LM Studio, OpenCLIP, shared runtime helpers and the Grounding DINO server. The CLI, root Pixi package, service Pixi workspace, imports and tests were updated. A golden record-key test guards the rename. Earlier proposed paths and compatibility aliases below are historical; the implementer deliberately removed `vad.routing` and old aliases rather than keeping shims.

| Priority | Finding | Suggested change and evidence |
|---|---|---|
| P1 before external adoption | `src/services` is installed in the **same** `vad` wheel as `src/routing` (`pyproject.toml`), and `services.contracts` imports `routing.schemas`. The Grounding DINO service has its own Pixi environment, but shares the root package and wire models. This is process/dependency separation, not an independently versioned service contract; the generic top-level `services` import also risks collision in a larger host environment. | Decide whether this coupling is acceptable for one repository. If independent deployment is required, put the `svc-1.0` contract in a small versioned shared package, give client/server explicit package names, and test a server install against that wheel. Do this before promising independently upgradable services. |
| P2 | CI installs the default environment; `tests/services/grounding_dino/test_server.py` skips without FastAPI. The green default suite therefore does not execute the server's request/response contract after these moves. | Add a separate CI job using `src/services/grounding_dino/pixi.toml` that runs its fake-model tests with a locked install. Keep live GPU checks optional. |
| P2 | `services.wemm_mcp.score` uses only positive prompts, although Stage A passes positive **and negative** prompts. Its `max(cosine)` scale is documented, but changing prompt composition could materially change scores. The scorer accepts `@unverified` when no server build is reported. | Keep the current score definition until measured on labelled data; record this limitation in active docs. Add a verified build/revision to the server response and require it for promotion or long-lived cache reuse. Recalibrate thresholds if negatives, instruction strings, or pooling change. |
| P2 | `services.wemm_mcp.score` builds a dictionary by match index, so duplicate indices overwrite earlier scores silently; it reads only the first result row. A malformed or future multi-row MCP reply can be treated as a valid score set if all expected indices appear. | Require exactly one result row and exactly one finite match per expected index, and verify query identity or cardinality where the MCP contract exposes it. Add malformed-response tests for duplicates/extra rows. |
| P2 documentation | The historical sections below still describe `src/vad`, `third_party_service`, and remote clients as proposals. `docs/project_board.md` now records F-39 correctly, but readers of this review can mistake the older tables for current architecture. | Archive or condense the obsolete proposed tree after this review is accepted; keep this section and the board as the current status. Refresh the live-test and deployment claims with dated evidence when reproduced. |

**Keep:** per-class Grounding DINO passes avoid ambiguous phrase-to-class mapping; explicit zero-detection versus failure responses; data-URI transport for WeMM; scorer failures become unknown relevance/review; response identity checks and bounded image size. Real-image calibration, detector false positives, and frozen-set benchmarks remain the main release gates. No application source was changed in this review.

## Earlier naming and structure review — superseded by F-39

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
