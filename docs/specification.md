# Specification: adaptive routing for custom object detection

Requirements, acceptance criteria and the implemented execution flow for this project. Status of each item is tracked on the [project board](project_board.md). Keep the execution diagrams aligned with runtime behavior; the later calibration and rollout requirements describe the intended release gates.

Source: the original plan (Part B of `git show 208713e:docs/project_board.md`), without the Codex execution prompt. The repository layout below is as built.

## Goal and integration boundary

Add an image-level routing layer to the existing self-hosted Python detector pipeline. It decides how much inference and review each image needs, then emits traceable candidate annotations for the current dataset writer and trainer. It must support arbitrary project-defined classes, including classes absent from public detection taxonomies. Python owns routing decisions; Claude Code or Codex can implement and inspect the code, but neither is a runtime decision maker.

**First repository task:** locate the current image manifest, class registry, primary detector call, prediction format, annotation writer, training entry point and evaluation split. Wrap those interfaces. Preserve current detector and annotation behavior behind a feature flag until the new path passes the acceptance tests. This plan does not assume a particular detector family, labeling format or VLM. *(Greenfield here: see "Integration notes" on the [project board](project_board.md).)*

Each image has three distinct questions:

1. **Relevance:** How likely is at least one configured target class present? Estimate per class and aggregate to image level. Relevance is not image quality.
2. **Difficulty:** How likely is the primary detector to miss or mislocalize a target **if one is present**? Estimate from visual and prediction signals. Record `unknown` when evidence is insufficient.
3. **Annotation completeness:** After candidate fusion, is there evidence of a target instance with no accepted annotation? This is an audit result, not another synonym for difficulty.

Keep raw embedding similarity, calibrated probability, quality signals and difficulty score in separate fields. In particular, `zero primary detections` never implies `easy` or `irrelevant`.

## Processing graph

The following describes the implementation reviewed at `be8be70` on 2026-11-10. The runtime is ordinary Python orchestration, not a state-machine framework. State names in the diagrams explain control flow; only the schemas, cached step payloads, records and errors described below are persisted. Source links are relative to this document. Known comparison defects are recorded in [reviewer_suggestion.md](reviewer_suggestion.md); diagrams describe current behavior, not proof that every boundary is enforced.

### Commands and artifact flow

`vad-route` enters [cli.main](../src/routing/cli.py). These are separate invocations, not an automatically scheduled training loop. Dashed edges below denote user-supplied inputs or subsequent invocations.

```mermaid
flowchart TD
  Images["Image manifest and task config"] --> Group["sample group: source/duplicate groups and content hashes"]
  Group --> Split["sample split + labels: calibration, development, frozen_eval"]
  Split --> SplitFiles["split-v2 manifest and frozen-lock-v2"]
  Images --> Shadow["shadow: routed inference"]
  SplitFiles -. optional split membership and export tags .-> Shadow
  Shadow --> Trace["RouteRecord trace, accepted annotations, cost/error report"]
  Trace --> Queue["sample review-queue: prioritize annotation work"]
  Queue -. human labels .-> Merge["sample merge: combine supplied labels"]
  Merge -. labels .-> Fit["fit-relevance: calibration-only fit"]
  Trace --> Fit
  SplitFiles --> Fit
  Fit --> Calibrator["Versioned calibrator with input identities"]
  Calibrator -. next shadow invocation .-> Shadow
  Trace --> Findings["review findings: open model/audit findings"]
  Findings -. adjudication log .-> Apply["review apply: validate decisions and re-audit"]
  Apply --> Reviewed["Separate ReviewedRecords and annotation export"]
  Trace --> Eval["evaluate + ground truth: development or frozen cohort"]
  SplitFiles -. cohort and content verification .-> Eval
  Eval --> Metrics["New run-metrics-v1 file"]
  Metrics --> Compare["compare: previous or best eligible earlier run"]
  Compare -. development decisions .-> Images
  Images --> Legacy["legacy: replay primary only, export annotations"]
  Schema["export-schema"] --> SchemaFiles["Pydantic JSON Schemas"]
```

`sample merge` combines supplied labels; it does not label images. `review apply` consumes human decisions and preserves the original model trace. `evaluate --trace` currently expects original `RouteRecord` entries, not `reviewed.jsonl`; the reviewed-export/trainer evaluation loop is not automatically connected. Supplying no split to evaluation produces a descriptive, content-unverified report over all labelled images. Development evaluation requires a v2 split and frozen lock.

### Shadow inference flow

```mermaid
flowchart TD
  CLI["vad-route shadow: validate args, config and manifest; build providers"] --> Lock["Enable routing for this run; acquire output lock"]
  Lock --> Image["Next manifest image: load, validate and hash"]
  Image --> Cache{"Final record exists for this run key?"}
  Image -->|unreadable or corrupt| Error["Persist per-image ErrorRecord"]
  Cache -->|yes| Reuse["Reuse completed RouteRecord"]
  Cache -->|no| A["Stage A: quality and relevance; reuse or persist"]
  A --> Primary["Primary detection; reuse or persist normalized boxes"]
  Primary --> B["Stage B: box features and pre-secondary difficulty"]
  B --> Route["Compute or reuse persisted route decision"]
  Route --> Secondary["Optional secondary: action, capability and budget checks"]
  Secondary --> Slicing["Optional slicing: action, capability and per-tile budget checks"]
  Slicing --> Fusion["Match and fuse; retain provenance; assign candidate statuses"]
  Fusion --> Verify["Optional bounded VLM: candidates, then missing classes"]
  Verify --> Audit["Completeness audit: complete, needs_review or unresolved"]
  Audit --> Save["Persist RouteRecord and trace"]
  Primary -->|failure| Error
  Reuse --> Next{"More images?"}
  Save --> Next
  Error --> Next
  Next -->|yes| Image
  Next -->|no| Export["Current-run annotations, route_trace.jsonl and report.json"]
```

`standard`, `secondary`, `sahi`, `vlm_verify`, `human_review` and `audit_sample` are **combinable actions**, not mutually exclusive processing states. The pipeline checks membership in the action list and executes secondary before slicing, regardless of the list's order. `human_review` records review requirements; it does not pause the process or launch a review UI. Training and frozen benchmarking remain downstream workflows, not steps performed by `vad-route shadow`.

### Command and batch boundaries

[cli.main](../src/routing/cli.py) constructs providers and loads the manifest before entering the shadow output lock. Invalid command/configuration, manifest-loading errors and setup failures stop the command; they are outside the per-image error queue. `StoreLockedError` returns exit code 2. Shadow mode explicitly sets `routing_enabled=True` for that invocation; the default configuration remains disabled. The legacy branch calls `legacy_predict` directly and does not use routed per-image recovery.

[RoutingPipeline.run](../src/routing/pipeline.py) processes manifest entries sequentially. Each `process` returns either a `RouteRecord` or `ErrorRecord`, so an unreadable image or primary inference failure does not stop later valid entries. This assumes error persistence itself succeeds. Unexpected uncaught errors during scoring, fusion, audit or record construction also become per-image errors, tagged with the last `_step`.

### Per-image execution state machine

```mermaid
stateDiagram-v2
  [*] --> Loading
  Loading --> ErrorQueued: invalid image
  Loading --> Lookup: valid image and run key
  state lookup_result <<choice>>
  Lookup --> lookup_result
  lookup_result --> Reused: final record found
  lookup_result --> StageA: no final record
  StageA --> Primary: relevance result or unavailable evidence
  Primary --> ErrorQueued: detector failure
  Primary --> StageB: valid result including zero boxes
  StageB --> RouteReady: compute or reuse decision
  RouteReady --> Secondary: optional action gate
  Secondary --> Slicing: success, skipped or failure recorded
  Slicing --> Fusion: success, skipped or partial result recorded
  Fusion --> Verification: candidates and disagreement ready
  Verification --> Audit: verified, skipped or uncertain
  Audit --> Persisting: completeness assigned
  Persisting --> Recorded: store write succeeds
  Reused --> [*]
  Recorded --> [*]
  ErrorQueued --> [*]
```

| Stage / source | What happens | State retained for explanation or recovery |
|---|---|---|
| Loading / `image_inputs.load_image` | Read bytes, verify/decode, convert to RGB and build an `ImageRef` with byte digest. | An invalid image produces an error; no detection is attempted. |
| Lookup / `process`, `record_key` | Combine image identity with `RunInfo`, including configuration/model identities and available artifact digests. | A final-record hit returns immediately after image validation. |
| Stage A / `scoring.stage_a` | Quality signals plus optional per-class relevance; a scorer exception yields unavailable relevance and a warning. | `stage_a` cache includes failures represented as unavailable evidence; resume reuses it. |
| Primary / `_run_cached_detector` | Run or replay primary predictions and map valid boxes to original-image coordinates. An explicit empty prediction row is valid; an omitted replay row raises. | `primary` cache stores detections, warnings and measured GPU time. |
| Stage B / `box_features`, `estimate_difficulty` | Calculate features and `difficulty_pre_secondary`; zero boxes imply unknown difficulty. | Derived again from cached inputs until the final record is saved. |
| Route / `policy.route` | Apply ordered rules and available capabilities; persist before optional escalation. | `route` contains actions, reasons, priority and expected budget. |
| Secondary and slicing | Check the action, adapter and budget; normalize/fuse tile results. Optional failures accumulate review reasons. | `secondary` and `sahi` caches; slicing retains raw predictions, merged predictions, count and stop reason. |
| Fusion / `_candidates` | Match primary/secondary detections, derive disagreement, fuse all detector sources and assign initial statuses. | Preserve both difficulty snapshots. `difficulty_final` is recomputed only when secondary ran; it does not trigger a second routing pass. |
| Verification / `_verify` | Check reviewable candidates in stable order, then high-relevance classes without an accepted box, using one shared call cap. | Per-request result/attempt markers; VLM outcomes cannot create geometry. |
| Audit and persistence / `completeness.audit`, `RouteStore` | Evaluate coverage and open evidence; save the final trace and record. | Final completeness and all source/candidate/verification records. |

Exceptions not drawn individually in the state diagram follow `process`'s outer error handler. A handled relevance, secondary or slicing failure stays on the normal path with unknown evidence or review reasons. A VLM failure becomes `uncertain` when a request was attempted. Skipping a request on a budget leaves the candidate unchanged and adds a failure reason.

### Route selection and budget guards

[policy.route](../src/routing/policy.py) applies the following main branches, then adds calibrated-missing-class and sampled-quality review requirements. Availability changes the selected actions.

| Evidence, in branch order | Route behavior |
|---|---|
| Low relevance, zero primary boxes | Standard path plus a deterministic sampled audit; no automatic discard. |
| Low relevance with boxes | Standard plus VLM verification, or human review when VLM is unavailable. |
| Non-low relevance, zero boxes | High relevance escalates; moderate relevance takes the moderate path. Unknown relevance uses the configured medium/high interpretation. |
| Non-low relevance with boxes and hard/small/crowded evidence | Escalate: plan slicing if available and within slice count, add secondary and VLM when configured; otherwise record unavailable capabilities/review. |
| Remaining medium/unknown difficulty or moderate relevance | Secondary if available, plus VLM when available. Without secondary, use the configured VLM-then-human or human fallback. |
| Remaining high relevance and low difficulty | Standard path. |

Unknown relevance is never treated as low. An omitted calibrated class adds `human_review` and never falls back to raw probability thresholds. Poor-quality sampling can add review to any branch.

Budgets are cooperative guards: Stage A and primary still run for every valid uncached image. Elapsed time is checked before optional work and before each slice; an in-flight call can exceed that budget. GPU limits use adapter-measured time only, and become unenforceable when the aggregate is unknown (`None`). Slicing returns partial results with a stop reason; it does not silently claim a complete pass. VLM call counts include cached answers reached through `_verify_request`, preserving the logical per-image cap on resume. Provider timeouts are separate from the pipeline elapsed budget; the WeMM client currently retries a transport failure once.

### Candidate status state machine

[RoutingPipeline._candidates](../src/routing/pipeline.py) initially accepts an unflagged primary-backed fused box. A non-primary proposal begins as `candidate`; flagged primary-backed boxes begin as `needs_review`. Examples of flags are cross-class conflict, mislocalization, primary-only disagreement, low relevance and missing planned confirmation.

```mermaid
stateDiagram-v2
  [*] --> Fused
  Fused --> candidate: no primary source
  Fused --> needs_review: primary source with flags
  Fused --> accepted: primary source without flags
  candidate --> Checking: VLM selected and budget permits
  needs_review --> Checking: VLM selected and budget permits
  Checking --> needs_review: guarded cross-class conflict
  Checking --> accepted: present and acceptance permitted
  Checking --> needs_review: present proposal but acceptance disabled
  Checking --> needs_review: uncertain
  Checking --> needs_review: absent and absent_rejects is false
  Checking --> rejected: absent and absent_rejects is true
```

`Checking` and `Fused` are explanatory states, not schema values. If verification is skipped, `candidate`/`needs_review` remains unchanged. `_candidate_status_after_verification` checks the cross-class guard **before** interpreting the verdict. A `present` verdict can accept an existing detector proposal only when `accept_confirmed_detector_proposals` permits it; the VLM never authors or moves boxes. Accepted and rejected candidates are not re-verified during this pass.

### Image completeness is a separate decision

[completeness.audit](../src/routing/completeness.py) evaluates the entire image after candidate verification. Its statuses do not replace box statuses:

1. Collect unmatched `candidate`/`needs_review` boxes, failures and suspected missing classes. A high-relevance class without an accepted box is unresolved when its check is missing, uncertain or reports an unannotated instance. An uncertain candidate verification also creates unresolved evidence.
2. A calibrated class with missing probability and no accepted box remains unresolved. A VLM answer cannot resolve that missing relevance evidence.
3. Establish authority from `human_outcome`, otherwise a configured calibrated audit policy, otherwise `none`.
4. Apply the precedence below. A human `complete` cannot clear unmatched candidates, unresolved evidence or failed escalation without adjudicating those findings.

```mermaid
flowchart TD
  Evidence["Collect open findings and authority"] --> Human{"Human complete and no open findings?"}
  Human -->|yes| Complete["complete"]
  Human -->|no| Unknown{"Unresolved evidence?"}
  Unknown -->|yes| Unresolved["unresolved"]
  Unknown -->|no| Review{"Unmatched boxes, failure, no authority, or human_review action?"}
  Review -->|yes| NeedsReview["needs_review"]
  Review -->|no| Complete
```

The initial shadow pass has no human adjudications. With no human outcome and no calibrated audit policy, even an image whose boxes are all accepted remains `needs_review` unless unresolved evidence requires the stronger `unresolved` status. The separate [review](../src/routing/review.py) workflow now ingests adjudications: `review findings` lists open findings and `review apply` checks record/candidate/class/config identity, applies decisions and re-runs the completeness audit. Later decisions supersede earlier decisions for a candidate; contradictory class absence and accepted geometry are refused. Human-added boxes are validated and retain their provenance. A complete decision must still satisfy the audit's open-evidence checks. Reviewed records and exports are separate artifacts; the source trace is unchanged.

### Restart and VLM request recovery

[RouteStore](../src/routing/store.py) persists independent step payloads; it does not save a single execution-state enum. Restart begins at image loading, returns a saved final record if present, otherwise traverses the same stages and reuses the available step results.

```mermaid
stateDiagram-v2
  [*] --> ResultLookup
  state result_available <<choice>>
  ResultLookup --> result_available
  result_available --> CachedAnswer: result exists
  result_available --> AttemptLookup: no result
  state attempt_available <<choice>>
  AttemptLookup --> attempt_available
  attempt_available --> Interrupted: attempt marker exists
  attempt_available --> MarkAttempt: no attempt marker
  MarkAttempt --> Sending: persist marker before VLM call
  Sending --> Validating: response received
  Sending --> Uncertain: exception or timeout
  Validating --> Answer: valid request binding
  Validating --> Uncertain: contradictory binding
  Interrupted --> Uncertain: never resend this request
  Answer --> SaveAnswer
  Uncertain --> SaveAnswer
  CachedAnswer --> [*]
  SaveAnswer --> [*]
```

This request machine runs only when `_verify` reaches the request within its budget. Cache keys are `vlm:<request-id>` and `vlm_attempt:<request-id>`. Request binding includes class, candidate IDs and region as well as request ID. Malformed provider responses are normalized to uncertain by the adapter. If the process dies after the marker but before the answer is persisted, resume records `vlm_interrupted` as uncertain instead of repeating inference—even if the original call was never sent. This is the deliberate at-most-once tradeoff.

Detector steps have no equivalent attempt marker. A crash after inference but before its cache write can repeat that detector call. Sliced inference is cached after the sliced operation returns, not after every tile, so interruption can repeat tiles. A saved final record is reused; its historical timings/warnings are not recomputed. Changing configuration or recorded artifact identity creates a different run key; changing an unverified remote model behind the same identity does not automatically invalidate SQLite results.

### Output and downstream boundary

Shadow mode writes only records returned for the current invocation:

- `annotations.json`: all valid routed images with completeness metadata, and only `accepted` candidate boxes, converted from xyxy to COCO-style xywh.
- `route_trace.jsonl`: complete `RouteRecord` values, including proposals, provenance, verifications and completeness.
- `report.json`: route/cost summaries, relevance-scorer failure counts and the current invocation's per-image errors.

An image may export accepted boxes while still being `needs_review` or `unresolved`; accepted geometry does not prove full coverage. The current export does **not** enforce a complete-images-only training gate. Optional split manifests validate membership and tag exports; the training-label loader rejects known frozen tags/IDs/groups when the corresponding provenance is supplied. It cannot identify renamed copies by image bytes because annotation exports lack content hashes. Actual trainer integration and measured promotion remain downstream requirements.

### Evaluation and comparison state machine

```mermaid
stateDiagram-v2
  [*] --> ValidateInputs
  ValidateInputs --> Refused: duplicate image IDs or originating config mismatch
  ValidateInputs --> SelectCohort: labels and trace valid
  SelectCohort --> Descriptive: no split, or legacy frozen lock
  SelectCohort --> CheckContent: v2 split and lock supplied
  CheckContent --> Refused: frozen label lock or image identity mismatch
  CheckContent --> Development: development selected and frozen ID/group excluded
  CheckContent --> Frozen: frozen selected and content verified
  Descriptive --> Score
  Development --> Score
  Frozen --> Score
  Score --> WriteMetrics: official COCO, custom diagnostics, annotation P/R, costs
  WriteMetrics --> CompareInputs: separate compare invocation
  CompareInputs --> Refused: content ID invalid, mixed runs, or series differs
  CompareInputs --> DescriptiveDelta: image content unverified, previous comparison
  CompareInputs --> SelectBaseline: image content verified
  SelectBaseline --> CustomInterval: previous, or best official AP with guardrails
  CustomInterval --> ComparisonReport: custom interval available or explicitly unavailable
  DescriptiveDelta --> ComparisonReport
  ComparisonReport --> [*]
  Refused --> [*]
```

- **Input identity:** evaluation binds supplied configuration to trace run hashes, allowing supported shadow overrides. Frozen labels must match their lock; v2 image hashes and dimensions must match the split and trace. Genuine legacy locks/manifests remain loadable as unverified content, with no fabricated hashes or rewriting. Development labels are not frozen, and their digest distinguishes comparison series.
- **One cohort:** accuracy and cost use the selected labelled cohort. A labelled image without a routed result stays in the accuracy denominator as an operational failure; out-of-cohort records are listed separately. Cost summaries describe available cohort records, so consult failure counts alongside costs.
- **Prediction sources:** official `coco.map50_95` and `coco.map50` are reported for primary, routed all-unrejected and routed accepted-only boxes. Routed ranking uses the maximum confidence among source detections (`routed-score-v1`); unscored routed boxes are counted and excluded from AP. Primary AP becomes unavailable if any primary prediction is unscored. Score-free accepted annotations receive precision/recall, not AP.
- **Metrics identity:** evaluation refuses an existing output path and writes a content-identified `run-metrics-v1`, with originating runs, cohort/image/label digests, evaluator settings, and per-image custom match outcomes. A parent ID provides lineage; it does not authorize comparison. Evidence status distinguishes descriptive/synthetic/development results from human-labelled verified frozen evidence; content verification alone is not a promotion result.
- **Comparison:** `compare` checks input content IDs, rejects mixed-run arms and differing series, and requires guardrails for `--against best`. It reports official COCO deltas and separately named custom deltas. Seeded paired source-group bootstrap recomputes dataset AP from custom outcomes; it does not average per-image AP. Its interval and `verdict` concern **custom-vad-1**, not COCO. Opposite-sign changes produce a warning. Unverified comparisons are descriptive; best-run selection is refused for them. Current group-compatibility, numeric-validation and unavailable-AP defects are in R7-1–R7-3 of the reviewer feedback.
- **Iteration:** use development data for repeated previous/best comparisons, and frozen evaluation for predeclared promotion checks. Optional meaningful-gain/patience settings report an operational plateau on official AP; they do not prove statistical convergence. Guardrail results and the custom accuracy verdict are separate outputs: an `improved` verdict is not automatic permission to promote.

### Two-stage scoring

**Stage A, before detection:** validate the image; compute resolution, blur/exposure indicators and per-class relevance. The current remote scorer uses positive text prompts only; compression, reference-distribution distance and negative-prompt contrasts remain proposed signals. The stage outputs raw features plus calibrated relevance probabilities when a compatible calibrator is loaded. It may flag a likely hard image but must not claim an object-size or occlusion measurement without an object proposal.

`fit-relevance` fits per-class and any-target logistic models with ridge regularization using calibration-only rows. It checks score scale, scorer identity, task/prompts, originating config and split-v2 image content, and refuses frozen bytes even under another ID. Insufficient class support is recorded rather than silently filled by raw-threshold fallback. Loading the artifact checks scale/scorer/prompts and records its file digest. A fitted artifact's existence does not demonstrate target-domain reliability or calibrated difficulty.

**Stage B, after primary detection:** calculate normalized box area, predicted crowding/overlap, confidence or margins when supported, class ambiguity, count, and the contradiction between high relevance and zero boxes. Use Stage A and these features to estimate difficulty. If a secondary detector is run, recompute with class-aware matching and disagreement. Store `difficulty_pre_secondary` and `difficulty_final`; never silently replace the former, because its value explains why the secondary path was selected.

For matched boxes, use an explicitly configured IoU threshold and class mapping. Disagreement includes unmatched candidates from either detector and localization differences among matched boxes; record direction and count, not just a Boolean. A secondary model trained on identical data may have correlated errors, so its disagreement is evidence for review, not ground truth. If no secondary model exists, mark disagreement `unavailable` and allow routing from other signals.

## Contracts and repository layout

As built at the reviewed revision:

```text
src/routing/
  schemas.py       Pydantic v2 records + JSON Schema export
  task_config.py   class registry, prompts, thresholds, budgets, policy knobs, versions
  scoring.py       Stage A and Stage B feature extraction
  interfaces.py    detector / relevance / VLM protocols and result types
  detection_mapping.py, verification_results.py  box mapping; VLM result parsing
  slicing.py       native SAHI-style tiling and original-coordinate conversion
  matching.py      class-aware matching, deduplication, provenance-preserving fusion
  policy.py        pure deterministic route function
  completeness.py  unmatched proposals and coverage audit
  pipeline.py      bounded orchestration and resumability
  store.py, image_inputs.py  persistence + job lock; manifest/image loading
  evaluation.py    isolation, frozen lock, route/cost report, VLM agreement
  sample.py        source/duplicate grouping, content-bound splits, label queues/merge
  calibration.py   relevance fitting, artifact validation and calibrated scores
  review.py        human findings, adjudication validation and re-audit
  metrics.py       ground truth, custom matching/AP, annotation metrics, per-image outcomes
  coco_eval.py     official pycocotools bbox AP with explicit settings
  compare.py       compatible-run deltas, custom group bootstrap, guardrails and plateau
  export.py        COCO-style versioned export + loader
  cli.py           vad-route
src/services_embedding/      client.py: WeMM MCP relevance scorer
src/service_detector_vlm/      openai_service_call.py: OpenAI-compatible VLM verifier
src/service_detector_gdino/    client.py (RemoteDetector), contracts.py (svc-1.0), server.py, runtime.py, smoke.py
                               own pixi workspace for the GPU server
(routing/replay.py holds the file-based replay adapters)
configs/routing/example.yaml
tests/routing/, tests/services/
```

The 34-file source inventory also includes each package's `__init__.py`. CLI functions own file/argument boundaries; `pipeline.py` owns per-image sequencing; policy/scoring/matching/completeness own domain decisions. Service clients implement [interfaces.py](../src/routing/interfaces.py), while `schemas.py` and `task_config.py` define shared contracts. Replay stays in routing and the GPU server has its own environment.

| Service path | Runtime contract and failure behavior |
|---|---|
| [Embedding client](../src/services_embedding/client.py) | Remote MCP `compute_similarity`, image data URI plus positive prompts; validate backend/model/dimension and optional build identity, require finite complete scores, retry a transport failure once. Stage A records unavailable relevance if the scorer fails. |
| [VLM client](../src/service_detector_vlm/openai_service_call.py) and [result parser](../src/routing/verification_results.py) | OpenAI-compatible HTTP transport, bounded crop/context and strict presence-answer parsing. No geometry authoring; failed/malformed calls become uncertain. Pipeline checks request/class/candidate/region binding and handles durable attempt markers. |
| [Detector client](../src/service_detector_gdino/client.py) and [wire contracts](../src/service_detector_gdino/contracts.py) | POST `/detect` with encoded image content and identity; check response request, image size and model/revision/optional weight digest before mapping boxes. Primary failure queues the image; optional escalation failure creates review evidence. |
| [Detector server](../src/service_detector_gdino/server.py), [runtime](../src/service_detector_gdino/runtime.py), [smoke](../src/service_detector_gdino/smoke.py) | Separate GPU process: validate image bytes/dimensions, serialize inference, run requested classes with a pinned model and verified weights, report predictions/timing or failure. Smoke checks transport/model behavior, not real-data accuracy. |

**Review coverage:** source inventory/AST analysis plus manual tracing of the paths above and the CPU/mock suite (592 passed, seven live tests deselected). Static reachability alone cannot prove code unused: protocol implementations, callbacks and framework entry points are invoked dynamically. Live GPU inference, deployment behavior, real-image accuracy and trainer integration were not verified by this audit.

Define Pydantic v2 models and export their JSON Schemas. Reject unknown class IDs, nonfinite scores, invalid boxes, missing model versions and unsupported coordinate systems. Store a `schema_version` and content/hash IDs on every persisted record. Use canonical `xyxy` **pixel coordinates in the original image**, with `0 <= x1 < x2 <= width` and the analogous y rule. Convert sliced boxes before matching; keep slice origin and model provenance. Keep model confidences as raw model outputs rather than calling them calibrated probabilities.

Minimum contract example (illustrative values, not tuned thresholds):

```json
{
  "schema_version": "1.0",
  "image": {"id": "sha256-or-stable-id", "uri": "images/001.jpg", "width": 1920, "height": 1080, "sha256": "..."},
  "task": {"task_id": "site-safety-v1", "class_ids": ["helmet", "vest"], "prompt_set_version": "p1"},
  "run": {"config_hash": "...", "primary_model": "model@hash", "secondary_model": null, "relevance_calibrator": "cal@hash", "difficulty_calibrator": "cal@hash"},
  "stage_a": {"per_class": {"helmet": {"raw_similarity": 0.42, "relevance_probability": 0.71}, "vest": {"raw_similarity": 0.27, "relevance_probability": 0.19}}, "image_relevance_probability": 0.76, "quality": {"blur": 0.18, "exposure": 0.05}, "warnings": []},
  "primary_candidates": [{"id": "p1", "class_id": "helmet", "box_xyxy": [100, 80, 135, 115], "raw_confidence": 0.61, "source": "primary", "model_version": "model@hash"}],
  "stage_b": {"difficulty_pre_secondary": 0.64, "difficulty_final": 0.64, "bucket": "hard", "features": {"min_box_area_fraction": 0.00059, "crowding": 0.0}, "disagreement": {"status": "unavailable"}},
  "route": {"actions": ["secondary", "sahi", "vlm_verify"], "reason_codes": ["small_candidate", "high_miss_risk"], "policy_version": "route-v1", "budget_used": {"slices": 12, "vlm_calls": 1}},
  "final_candidates": [{"id": "f1", "class_id": "helmet", "box_xyxy": [100, 80, 135, 115], "status": "accepted", "source_ids": ["p1"]}],
  "completeness": {"status": "needs_review", "unmatched_candidate_ids": [], "suspected_missing_classes": ["vest"], "reason_codes": ["high_relevance_zero_boxes_for_class"]},
  "trace": {"elapsed_ms": 420, "gpu_ms": 310, "warnings": []}
}
```

Separate request/config types: `TaskSpec(class_id, display_name, positive_prompts, negative_prompts, aliases)`, `ImageRef`, `Detection`, `ScoreRecord`, `RouteDecision`, `VerificationRequest`, `VerificationResult` and `AnnotationAudit`. `VerificationResult` is constrained to `present | absent | uncertain`, the requested class, cited candidate IDs or image regions, and a reason code; invalid or unparseable VLM output becomes `uncertain`. The VLM cannot directly create a final accepted box. Human approval or a configured detector-supported proposal is required for a new box.

## Deterministic policy, version 1

All numbers below are **configuration placeholders**. Existing raw similarity cutoffs `0.3` and `0.5` from the prior discussion can seed relevance bands only if the current embedding model uses that same scale; they are not difficulty cutoffs or calibrated probabilities. Run the shadow phase before enabling automatic discard.

1. Run Stage A and primary inference on every valid image in the initial release. Invalid/corrupt images go to an explicit error queue.
2. Compute image relevance as the calibrated probability of any target being present, accounting for class correlations during calibration rather than simply summing per-class values. If unavailable, use raw similarity bands and label the result `uncalibrated`.
3. Set `difficulty=unknown` if key signals are missing or the primary detector has zero boxes. Zero boxes with high relevance triggers at least a secondary pass or a bounded spot check.
4. For high-relevance, low-difficulty images with coherent primary detections: accept the standard path, then audit completeness.
5. For medium difficulty or moderate relevance: run the independent secondary detector if configured, match/fuse candidates, and verify only unresolved discrepancies.
6. For high difficulty, a small-object risk, severe crowding, or high relevance with zero boxes: run SAHI if an adapter and remaining slice budget exist; add secondary inference when configured. Run VLM verification for unresolved candidates or suspected missing classes. SAHI tile size, overlap, merge strategy and full-image pass are task configuration, tuned on calibration data.
7. For low relevance: keep the image in a sampled audit pool, including a fixed share of zero-box images and images with poor quality/domain shift. Do not silently discard until the measured false-negative risk of this gate meets the target. A low-relevance image with detector evidence is routed for review, not discarded.
8. Enforce caps on slices, VLM calls, elapsed time and GPU time per image. On cap or model failure, emit `needs_review` with the trace; do not convert an unverified image to a clean negative. No recursive retries. Persist route decisions before expensive calls so restarts are idempotent.

Represent the policy as ordered pure rules over typed scores. Return the action list, priority, reason codes and expected budget. Fixed config, model artifacts, image bytes and random seed must reproduce the route. For randomized low-relevance audits, select by a stable hash of image ID and policy seed, not process order.

**Candidate fusion:** class-aware matching in original coordinates; preserve all source predictions, then choose one canonical box under a documented tie rule. Suppression may remove duplicate display boxes but must not erase disagreement or a candidate requiring audit. VLM output may confirm/reject a candidate or request review; it is not a localization oracle.

**Completeness audit:** compare fused candidates against accepted annotations by class and IoU; collect unmatched candidates, high-relevance classes with no accepted box, and audit-sampled negatives. Status is `complete | needs_review | unresolved` with reason codes and reviewer outcome. Do not claim complete solely because all model boxes were accepted. Record whether a human or a calibrated audit policy established completeness.

## Calibration and frozen evaluation

Create a task-specific, human-checked sample across class, source/domain, image resolution, object size, crowding, positive/negative images and low similarity. Split by video/source/near-duplicate group to avoid leakage. Keep three disjoint sets: **calibration** for fitting probabilities and thresholds; **development** for implementation choices; **frozen evaluation** for final comparison. Version and lock the frozen image IDs and annotations before tuning. Never feed frozen images, or reviewer corrections from them, into model training, prompt selection or threshold fitting.

Calibrate relevance against `any target present`, and difficulty against a clearly defined primary-detector failure event, e.g. at least one missed ground-truth instance at the agreed IoU/class criterion. Report reliability by class/domain; use a global fallback only where samples are sparse. Select thresholds under explicit constraints: maximum allowed missed relevant images at the relevance gate, minimum recall on hard/small objects, maximum false positives per image, and processing budget. Use confidence intervals or a bootstrap by source group; defer automatic rejection when evidence is thin. Record calibrator versions and the exact calibration data hash.

Compare **baseline primary-only**, **always-expensive** and **adaptive** pipelines on the same frozen set. Report mAP50–95, class- and size-stratified recall, missed-instance recovery, false positives/image, annotation precision and completeness, review rate, latency percentiles, GPU seconds/image, VLM calls/image, and route confusion by domain. Include a low-relevance audit estimate of the false-negative rate. Analyze VLM agreement against human labels; never score a VLM self-judgment as ground truth. Evaluate annotations separately from the detector trained on them. For training benefit, train identical detector configurations on versioned baseline versus routed annotation sets, then evaluate both on the frozen set.

Promotion gate: adaptive routing must meet project-set recall and false-negative constraints, show a useful cost/latency reduction versus always-expensive processing, and avoid a material precision regression versus primary-only. Record numeric targets in the repository config before the frozen run; the plan intentionally does not invent task-specific targets.

## Acceptance tests

| Test | Required result |
|---|---|
| Arbitrary class registry | A class absent from public datasets flows through prompts, adapters, route, VLM contract and annotation export without hard-coded mappings. |
| Independence | Low relevance/high difficulty and high relevance/low difficulty keep distinct scores and take the configured routes. |
| Zero detections | High relevance plus zero boxes never becomes `easy` or an accepted negative; missing secondary/VLM capacity creates review. |
| Stable routing | Same image, artifact hashes, config and seed give identical actions, reasons and audit sampling across process restarts. |
| Disagreement | Matched, unmatched, duplicate and cross-class boxes give correct directional disagreement and keep provenance. |
| SAHI coordinates | Tile-edge and overlap detections map to original pixels once, deduplicate correctly and stay in bounds. |
| VLM failures | Malformed, contradictory, timed-out or unavailable responses become `uncertain`; retry/call caps hold. |
| Completeness | A missing ground-truth-like proposal, a high-relevance class with no box, and an audited zero-box image create explicit unresolved/review states. |
| Budget and recovery | Slice/time/GPU limits stop escalation; a repeated job resumes without duplicate calls or annotations. |
| Data isolation | Calibration and frozen-eval group IDs don't overlap training; changing thresholds never rewrites frozen annotations. |
| End-to-end | On a tiny checked fixture, exported annotations load in the existing training pipeline and the route trace explains each decision. |

## Research references and implementation mapping

**Evidence boundary (reviewed 2026-11-10):** the architecture combines established techniques with project-specific heuristics. The cited work supports individual components or related ideas; none evaluates this repository's complete router, its weights, thresholds or completeness guarantee. Source mappings below reflect the implemented flat service packages. OpenCLIP is not a required dependency for these references. This code review updates implementation mappings, not the literature search.

### Reference documents

| ID | Primary reference | Supported scope and limits |
|---|---|---|
| R1 | S. Pertuz, D. Puig and M. A. Garcia, **Analysis of focus measure operators for shape-from-focus**, *Pattern Recognition*, 46(5), 1415–1432, 2013. [Publication and DOI](https://researchportal.tuni.fi/en/publications/analysis-of-focus-measure-operators-for-shape-from-focus/); [author's Focus Measure implementation](https://www.mathworks.com/matlabcentral/fileexchange/27314-focus-measure). | Focus measures provide sharpness cues. This is not a validation of our whole-image quality assessment, exposure cutoffs or inverse blur normalization. |
| R2 | Tencent, **WeMM-Embedding-4B model card**, technical documentation. [Model card, inference examples and linked technical report](https://huggingface.co/tencent/WeMM-Embedding-4B). | Documents multimodal embeddings and model usage. Vendor documentation supports the selected backend, not our prompt aggregation, thresholds or probability interpretation. |
| R3 | S. Liu et al., **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**, 2023 preprint / ECCV 2024. [Paper](https://arxiv.org/abs/2303.05499); [official implementation](https://github.com/IDEA-Research/GroundingDINO). | Supports language-conditioned detection. Our per-class passes, thresholds, service wrapper and target-domain performance require separate evaluation. The repository uses a Transformers implementation, not the paper's training pipeline or NVIDIA TAO. |
| R4 | F. C. Akyon, S. O. Altinuc and A. Temizel, **Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection**, ICIP 2022. [Paper](https://arxiv.org/abs/2202.06934); [official SAHI implementation](https://github.com/obss/sahi). | Supports sliced inference for small objects. Published gains on other datasets are not predicted gains here; native merging and conditional invocation are our implementation choices. |
| R5 | Y. Wang et al., **ALWOD: Active Learning for Weakly-Supervised Object Detection**, ICCV 2023. [Paper](https://arxiv.org/abs/2309.07914). | Uses disagreement/uncertainty to select annotation work. This is related evidence for review signals; we do not implement its student–teacher training or acquisition algorithm. |
| R6 | C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, **On Calibration of Modern Neural Networks**, ICML 2017. [Paper](https://proceedings.mlr.press/v70/guo17a.html). | Establishes the need to assess probability calibration; studies classification and post-processing, including temperature scaling. It does not justify applying temperature scaling directly to our cosine scores or heuristic difficulty. |
| R7 | F. Küppers, J. Kronenberger, A. Shantia and A. Haselhoff, **Multivariate Confidence Calibration for Object Detection**, CVPR Workshops 2020. [Paper](https://arxiv.org/abs/2004.13546). | Extends calibration to detector outputs using box information, including position/scale. Detector correctness calibration is a different target from image relevance or image-level miss risk. |
| R8 | Y. Li et al., **Evaluating Object Hallucination in Large Vision-Language Models**, EMNLP 2023. [Paper and linked POPE code/data](https://arxiv.org/abs/2305.10355). | Motivates object-presence testing and checking hallucination against labels. Our verifier is not an implementation of POPE, and this paper does not establish our crop-verification accuracy. |
| R9 | D. Madras, T. Pitassi and R. Zemel, **Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer**, NeurIPS 2018. [Paper](https://arxiv.org/abs/1711.06664). | Related work for passing decisions to another decision-maker. Our deterministic human-review fallback is not a learned deferral policy, and no accuracy/fairness improvement is established here. |

### Usage in this code base

| Area / references | Current code and behavior | What is not implemented or established |
|---|---|---|
| Quality / R1 | [scoring.quality_signals](../src/routing/scoring.py) computes grayscale Laplacian variance, `blur = 1 / (1 + variance / 100)`, extreme-pixel fraction at ≤5 or ≥250, and shortest-side resolution. [policy.route](../src/routing/policy.py) uses quality for sampled auditing; blur also enters difficulty. | These are diagnostic proxies. Textureless scenes can look blurry; legitimate black/white content can look poorly exposed. No learned quality estimator, compression score, domain-distance population or measured relation to detector failure exists. |
| Relevance / R2 | [WeMMMCPRelevanceScorer.score](../src/services_embedding/client.py) sends an image data URI and positive prompts; takes maximum cosine per class. [stage_a](../src/routing/scoring.py) takes the maximum class score for raw image bands and supports a fitted relevance artifact. Missing/error evidence cannot silently become low relevance. | Negative prompts are not used by this scorer; image–image reference matching is planned. Prompt pooling, raw bands and fitted probabilities need target-domain validation. |
| Detection / R3 | [Grounding DINO server](../src/service_detector_gdino/server.py) loads a pinned model, checks the weights digest and runs once per requested class. [RemoteDetector](../src/service_detector_gdino/client.py) checks response identity. | Synthetic checks establish transport/behavior only. Real class confusion, false positives, prompt robustness and small-object recall need human-labelled evaluation. No independent secondary backend is supplied by this implementation. |
| Small-object recovery / R4 | [slicing.run_sliced](../src/routing/slicing.py) tiles, checks budgets, maps coordinates and merges by IoS while retaining raw parents. Acceptance tests cover offsets, overlap, recovery and caps. | This is a native SAHI-style implementation; it does not import the SAHI library. Paper-level parity, slicing fine-tuning and target-domain accuracy/cost gains have not been demonstrated. |
| Difficulty and disagreement / R5, R7 | [box_features / estimate_difficulty](../src/routing/scoring.py), [matching.disagreement](../src/routing/matching.py) and [pipeline](../src/routing/pipeline.py) combine small boxes, overlap, minimum confidence, blur, class ambiguity and optional secondary disagreement. Zero boxes/confidence absence produce unknown difficulty. | The weighted score is heuristic and always labelled uncalibrated when numeric. No fitted difficulty model or demonstrated ranking of actual missed/mislocalized instances exists. ALWOD is related work rather than the implemented algorithm. |
| Calibration / R6, R7 | [calibration.py](../src/routing/calibration.py) fits per-class and any-target logistic relevance models, binds input/scale/scorer/prompt identity and checks calibration split content. [Stage A guards](../src/routing/scoring.py) reject incomplete/inconsistent evidence; omitted calibrated classes remain unknown. | Fitting machinery and synthetic checks are not calibration experiments. Target-domain reliability diagrams, Brier/ECE studies and box-sensitive detector calibration remain missing. Difficulty has no fitted calibrator path. |
| Verification / R8 | [VLM client](../src/service_detector_vlm/openai_service_call.py), [answer parser](../src/routing/verification_results.py) and [_verify](../src/routing/pipeline.py) constrain answers and bind them to requests. [vlm_agreement](../src/routing/evaluation.py) reports human agreement, decision coverage and present/absent precision with Wilson bounds. | Agreement tooling has no target-domain human label study attached. It does not measure localization or discover every missing instance. POPE-style negative sampling and prompt/crop sensitivity studies remain to be run. |
| Human review / R9 | [review.py](../src/routing/review.py) and CLI findings/apply ingest per-finding adjudications, validate added boxes, reject contradictions, re-audit and preserve the original trace. [completeness.audit](../src/routing/completeness.py) separates box acceptance from image completeness. | No reviewer workload/accuracy study or learned deferral policy is established. A review flag is not a completed review; reviewed outputs are not automatically trained/evaluated. |
| Whole-system validation / all references | [sample](../src/routing/sample.py) and [evaluation](../src/routing/evaluation.py) provide content-bound split/frozen-lock utilities. [metrics](../src/routing/metrics.py), [COCO evaluator](../src/routing/coco_eval.py) and [compare](../src/routing/compare.py) provide detection/annotation metrics, run identities and comparison; [tests](../tests/routing/test_acceptance.py) establish selected behavioral invariants. | Real-image ablations, numeric promotion evidence and training-benefit comparisons remain missing. Custom bootstrap intervals do not establish uncertainty on official COCO AP. Exports may contain accepted boxes from incomplete images and lack byte-based training isolation; training must handle these boundaries. |

### Evidence required before claiming improvement

Use a human-checked dataset, separated by source/near-duplicate group. If the trial starts with roughly 70% tuning and 30% frozen evaluation, further separate the tuning portion into calibration and development groups; the frozen portion must not choose prompts, weights or thresholds. Record image-transfer permission before calling external services on collected images.

Define three different targets: **relevance** = any configured target instance present; **difficulty** = a defined primary miss/localization failure conditional on a target being present; **candidate correctness** = correct class and localization under a fixed one-to-one matching criterion. Negatives additionally measure false positives. Never substitute detector confidence or VLM self-agreement for those labels.

On the same frozen set compare primary-only, primary+slicing, primary+VLM, always-expensive and adaptive runs. Keep model/prompt identities and thresholds fixed; report detection precision/recall, mAP, small-object recovery, review rate, errors/timeouts, latency and GPU/call cost with source-group uncertainty estimates. Report candidate acceptance separately from annotation completeness and any downstream training result. This is a proposed evaluation protocol, not an assertion that these experiments have been performed. Detailed action cards are in [reviewer_suggestion.md](reviewer_suggestion.md).

Pydantic's [model documentation](https://docs.pydantic.dev/latest/concepts/models/) is an implementation reference for typed validation, not research evidence of detection or routing accuracy.
