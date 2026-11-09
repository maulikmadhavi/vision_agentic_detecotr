# Adaptive routing for custom object detection: project board

**Updated:** 2026-11-09 · **Owner:** user · **Implementer:** Claude · **Review log:** [reviewer_suggestion.md](reviewer_suggestion.md)

**Status:** phases 0–4 done, Phase 5 started, Phase 6 not started. Routing is off by default (`routing_enabled: false`). `pixi run test`: 452 passed, 0 skipped, 1 warning (starlette testclient deprecation; harmless) (re-run by the analysis agent); the Grounding DINO server tests now run in the default suite (fastapi in the dev extra); live `test-vlm`, `test-gdino`, `test-wemm`: 1 passed each, `test-e2e` (all three services at once): 4 passed, and `test-live-required` (fails instead of skipping): 7 passed (reported by main_agent, synthetic images only; I re-ran `test-gdino` only, the WeMM URL is not in my environment). Environments: pixi only (the `clip` environment is gone). Git: tags `v0.1-phase4-baseline` (208713e) and `v0.2-r3-blockers-fixed` (0435d99); `backup/pre-author-rewrite` holds the pre-rewrite history.

The requirements and acceptance criteria are in [specification.md](specification.md); this board tracks their status only.

## Phases

| Phase | Status | What's left |
|---|---|---|
| 0. Map pipeline + fixture | **Done** (greenfield) | Baseline metrics need a real detector (T-1). |
| 1. Contracts + trace | **Done** | — |
| 2. Stage A + Stage B | **Done**, gaps | Compression and domain-distance signals (T-4); one scorer per run (T-10 INC 2); shadow only on the fixture and synthetic images so far (B-1a). |
| 3. Router + secondary + SAHI | **Done**, gaps | Remote detector client and Grounding DINO server exist (F-35, F-36); SAHI through the server ran on the fixture only; no independent secondary (B-8). |
| 4. VLM + completeness | **Done**, gaps | Reviewer loop built on fixtures (F-51, F-52); needs real reviewers and labels (T-2). |
| 5. Calibration + evaluation | **Started** | Tooling built on fixtures: metrics (F-49), split/lock/sample (F-50), relevance calibrator fit and load (F-53, F-54). Seven integrity fixes first (T-17). Then needs labelled data (T-2), difficulty and VLM studies (A4, A6), benchmark (B-4…B-6). |
| 6. Rollout | **Not started** | B-9. |

## Needs your decision

| ID | Decision | Default now | Recommendation |
|---|---|---|---|
| D-1 | `policy.unknown_relevance_as`: route unknown relevance like `medium` or `high` | `medium` | Keep for shadow; `high` only if relevance is often missing in production. |
| D-2 | `policy.moderate_without_secondary`: `vlm_then_human` or `human` | `vlm_then_human` | Keep. |
| D-3 | `policy.cross_class_always_review` | `true` | Keep until a secondary detector exists. |
| D-4 | `vlm.absent_rejects`: let a VLM "absent" reject a box | `false` | Set a target for `absent_precision_wilson_lower95` before measuring (B-6). |
| D-5 | Relevance bands on the `wemm_text_cosine_v1` scale (CLIP was removed) | none (`raw_scale: unspecified`) | Choose on the calibration split, or fit a calibrator (B-2, B-3). |
| D-6 | GPU budget counts only adapter-measured device time | measured only | Confirm. |
| D-7 | VLM model | `qwen3-vl-4b-instruct` (8B not loaded) | Load 8B and compare on real crops (B-6). |
| D-9 | Delete `backup/pre-author-rewrite` and `refs/original/*` (old NCS-email history), and the unused `.venv/` | kept | Delete once you're sure. |
| D-10 | The four packages (`routing`, `services_embedding`, `service_detector_vlm`, `service_detector_gdino`) ship in one wheel; `service_detector_gdino.contracts` is the `svc-1.0` wire contract. Keep one repo/wheel, or publish the contract as its own versioned package? (reviewer P1) | one wheel | Keep for now; make contracts self-contained (no import of `routing`). Split only if services must be deployed and upgraded independently. |
| D-14 | **Untagged exports can still reach training.** An export written without `--split-manifest` carries no `vad_split` tag, and a training reader that is not given the manifest cannot tell frozen images from others. Options: (a) require `--split-manifest` for `shadow` and `review apply` as soon as a split exists; (b) make the training reader refuse untagged exports; (c) accept the documented limit. Recommendation: (a). | Closes the last known path for frozen images into training. |
| D-12 | **Image-transfer gate for the WeMM server.** Plan: manifest rows may carry `transfer_allowed`; an explicit `false` blocks `--wemm` (exit 2) before any request, and the sample tool writes `false` by default. Should a MISSING field also block? Missing = block is the safer rule for the collected web images (their manifests have no such field), but breaks existing manifests; missing = allow keeps old runs working and relies on the field being written. Recommendation: missing blocks only when the manifest comes from the sample tool (it always writes the field); old manifests need an explicit opt-in flag. | Decides whether collected images can reach the remote server by accident. Needed before A2 stage 3. |
| D-13 | **A6 acceptance targets.** The absent-precision Wilson lower-bound target (and uncertain/timeout limits) must be set by you before any VLM label study; there is no default. | Predeclared gate for `absent_rejects`. Needed before the A6 study, not before the tooling. |

## To do — ready now

| ID | Card | Why |
|---|---|---|
| T-1 | **Evaluate the Grounding DINO service on real images.** The server exists (F-36) and `--primary-url` works; what is missing is the user's 30–50 trial images, human labels, and choosing prompts/thresholds on them (it boxes something for every prompt: absent class 0.71, blank image 0.83 on synthetic tests). Hosted APIs remain unbuilt: they need a normalizing adapter, terms/data-handling check, env-var credentials. | Unblocks real shadow runs, SAHI, the benchmark and rollout. |
| T-2 | **Build the human-checked sample.** Candidates fetched 2026-11-09 into git-ignored `data/`: 100 images (50 "wild boar", 50 "monkey") in `trial_images/`, 48 negatives (people indoors/outdoors, cats, dogs) in `trial_negatives/`, each with provenance in its `manifest.jsonl`. **Queries and folder names are not labels.** Rules decided by the user: (1) a human who knows the classes draws the boxes and marks absent classes; a second person reviews ambiguous images plus a sample of the rest; model predictions never become ground truth automatically. (2) Dedupe first, then group near-duplicates and same-source images and assign whole groups: about 70% tuning, 30% frozen, positives and negatives in both; no looking at frozen results while choosing prompts, thresholds or calibrators. (3) Exclude illustrations, collages and screenshots from the photo benchmark; keep them as a separate out-of-domain set only if production could see them, else discard; record exclusions before the split. (4) Do not send the collected images to the embedding server until their source terms and the operator's retention/access policy are confirmed; synthetic or explicitly cleared images only (the synthetic smoke test grants nothing for them). Tools needed: dedupe/grouping, a box-labelling workflow, split + lock (T-11). | Every Phase 5 card needs it. |
| T-11 | **Sample tooling for T-2:** near-duplicate grouping (the current filter only catches exact copies), exclusion tagging recorded before the split, group-wise 70/30 split writing a frozen lock (`evaluation.lock_frozen_set`), and a box-labelling workflow with a second-reviewer queue. Plan first. | Makes T-2 reproducible and leak-free. |
| T-12 | **Findings from the first three-service run (synthetic images):** (a) a primary-detector outage exits 0 with nothing on stderr (only `report.json` shows it); the WeMM failure does print a warning. Add a one-line stderr count of per-image errors, like the WeMM one. (b) The VLM identity is a model name only (LM Studio exposes no build), like WeMM `@unverified`; record that limit. (c) Grounding DINO draws boxes for both classes on a blank image; routing sent them to review, but the review rate was 1.0, so measure this on real images (T-1). | Makes a failed run visible and states the identity limits. |
| T-14 | **Open part of the pin-tests card:** under the VLM call cap, which candidates are checked is decided by candidate-id hash order, not by risk or confidence. Not changed; decide whether to rank by risk (new decision, not requested). | Pins and the cleanup (C1–C4) are done: F-41–F-46. |
| T-16 | **Reviewer work orders A2–A7, tooling on synthetic fixtures only (user approved).** Order: A3 metrics and the ground-truth label format → A2 grouping, split, frozen lock, enforcement, review queue → A7 adjudication loop → A5 relevance calibrator → A4 difficulty-vs-baselines analysis → A6 VLM label harness. Technical defaults taken from main_agent's plan: own mAP checked once against pycocotools (no new dependency), IoU 0.50 for recall/FP, COCO size strata, labels are COCO JSON plus `vad_*` fields, calibrator is logistic and refuses to fit below 10 positives or negatives per class. Every report carries `evidence_status` = NOT A FINDING unless human labels and a verified split exist. Real-image A4–A6 runs are blocked on labels (T-2/T-11) and D-12. A8 not started. | `reviewer_suggestion.md` A2–A7; stages tracked in F-cards as they land. |
| T-17 | **Reviewer round R5 (seven integrity defects in the new data paths; fix before relying on fitted artifacts or frozen results).** S1 label integrity: merged reviews map categories by name and check image sizes; duplicate COCO image/category/annotation ids rejected (R5-1, R5-4). S2 contradictory decisions: accept/reclassify plus `class_absent` on the same class is a conflict (R5-3). S3 frozen content binding: per-image hash and size in the split and lock, verified by `evaluate` and fitting; old locks reported unverified (R5-2). S4 fitting provenance: repeated image ids rejected; scale bound to the run (R5-5, R5-6). S5 one cohort for all metrics and cost (R5-7). S6 `evidence_status` needs human labels AND a verified split. A5 stage 3 (development report) waits for these. | Reviewer, `reviewer_suggestion.md` follow-up review. |
| T-3 | **Independent review** (human or Codex) of `routing/pipeline.py` `_candidates`, `_verify`, `_candidate_status_after_verification`, `policy.route`. | Same model family wrote and reviewed the code. |
| T-4 | **Stage A gaps:** compression indicator; domain-distance when a reference distribution exists (`StageA.domain_distance` is never filled). | Spec Stage A list; domain-shift audit sampling (rule 7). |
| T-10 | **Remote embedding scorer, remaining work.** INC 1 (text-image relevance) is done (F-37). Decision: the user's embedding service is the relevance scorer and CLIP is not used for now (code kept, unused), so the multi-scorer Stage A (old INC 2) is dropped. **INC 3:** image-image matching against per-class reference images (max cosine, references in a separate manifest, group isolation from frozen images); needs a spec line and the user's reference images. The server reports no build yet: runs record `@unverified` until `--wemm-build` is passed. **Limits to keep visible (reviewer):** the score is max cosine over *positive* prompts only (negatives are not used); changing prompt composition, instruction strings or pooling changes the scale, so recalibrate thresholds. Runs record `@unverified` until the server reports a build; require a verified build before promotion or long-lived cache reuse. | Image-image matching is new (not in the original spec). |
| T-7 | **Reviewer-outcome ingestion:** built on fixtures (F-51, F-52): `Adjudication` log, `review findings` / `review apply`, re-audit, reviewed export. Open: real reviewers, annotation-time measurement, and the S2 consistency fix (T-17). | Spec: human review closes findings explicitly. |
| T-8 | **Detection metrics:** built on fixtures (F-49): `vad-route evaluate`. Open: the S4–S6 fixes (T-17) and real human labels before any number counts. | Spec: mAP50–95, stratified recall, FP/image. |
| T-9 | **Enforce split isolation in runs:** built on fixtures (F-50, F-52): frozen lock verified by `evaluate`, `vad_split` tag on exports, `assert_split_allows` for fitting, training reader refuses frozen images. Open: image content is not yet bound to the lock (T-17 S3), and untagged exports (D-14). | Spec: frozen images never reach training or threshold fitting. |

## Backlog — blocked

| ID | Card | Blocked by |
|---|---|---|
| B-1a | Real-image shadow run from saved predictions (`PrecomputedDetector`) | real predictions + labels |
| B-1b | Live primary and real SAHI shadow | T-1 |
| B-2 | Relevance calibrator (P(any target), class correlation) and difficulty calibrator (defined failure event) | T-2 |
| B-3 | Threshold fitting under explicit constraints, bootstrap by source group; record calibrator versions + data hash | T-2, B-2 |
| B-4 | Numeric promotion targets in config **before** the frozen run: max missed relevant images at the relevance gate, min recall on hard/small objects, max FP/image, cost budget. Gate: meet recall/FN constraints, cut cost vs always-expensive, no material precision loss vs primary-only | user, T-2 |
| B-5 | Frozen benchmark: primary-only vs always-expensive vs adaptive. Report mAP50–95, stratified recall, missed-instance recovery, FP/image, annotation precision/completeness, review rate, latency percentiles, GPU s/image, VLM calls/image, route confusion by domain, low-relevance FN audit | T-1, T-2, T-8, B-3, B-4 |
| B-6 | VLM agreement on real human labels, 4B vs 8B; decides D-4 | T-2, D-7 |
| B-7 | Trainer integration (replace substitute loader `export.load_annotations`); train on baseline vs routed annotations | a trainer |
| B-8 | Secondary detector adapter (independent model) | T-1 |
| B-9 | Rollout: shadow → canary by source group, rollback config, gated promotion; no automatic discard before gates pass | B-5 |
| B-10 | Minor: `expected_budget.vlm_calls` reports the cap; with `absent_rejects: true` a rejected primary box stops counting as unmatched; `legacy` stops on a missing prediction row | — |

## Spec checklist (what's done, what's missing)

✅ done · ⚠️ partial · ❌ missing

| Area | Requirement | Status | Card / evidence |
|---|---|---|---|
| Contracts | Pydantic v2 records, exported JSON Schemas, drift test | ✅ | F-2 |
| | Reject unknown classes, NaN/inf, bad boxes, missing versions, other coordinate systems | ✅ | F-3 |
| | `schema_version` + hash IDs on persisted records | ✅ | F-4 |
| | Feature flag off by default, legacy path unchanged | ✅ | F-5 |
| Stage A | Image validity, resolution, blur, exposure | ✅ | F-6 |
| | Compression indicator | ❌ | T-4 |
| | Domain-distance indicator | ❌ | T-4 |
| | Per-class text–image similarity with positive/negative prompts | ✅ CLIP (live-tested); ⚠️ WeMM server scorer uses positive prompts only | F-25, F-37 |
| | Calibrated relevance probabilities | ⚠️ protocol + guards, no calibrator | B-2; F-31–F-33 |
| Stage B | Box area, crowding, confidence, ambiguity, count, high-relevance/zero-box contradiction | ✅ | F-7 |
| | `difficulty_pre_secondary` and `difficulty_final` both kept | ✅ | F-7 |
| | Calibrated difficulty | ❌ heuristic only | B-2 |
| Detectors | Primary detector through an adapter | ⚠️ replay; remote client + Grounding DINO server built and live-tested on synthetic images, not on real images | F-35, F-36, T-1 |
| | Optional independent secondary | ⚠️ protocol, no real adapter | B-8 |
| | SAHI with original-coordinate mapping | ⚠️ native slicer; real crop-capable client exists, not exercised on real images | F-10, F-35, T-1 |
| Policy | Pure deterministic rules, hash-sampled audits, replay across processes | ✅ | F-8 |
| | Rules 1–6, 8 (error queue, zero boxes ≠ easy, escalation, caps) | ✅ | F-8, F-14 |
| | Rule 7: audit pool incl. poor quality (done) and domain shift (not done) | ⚠️ | T-4 |
| | No automatic discard | ✅ by design | — |
| Fusion | Class-aware matching, directional disagreement, provenance kept | ✅ | F-9 |
| VLM | Verifier only; failures → `uncertain`; call cap; never authors boxes | ✅ | F-11, F-21, F-28 |
| Completeness | `complete` only by human or calibrated policy; open findings block it | ✅ | F-12, F-30 |
| | Human queue that records reviewer outcomes | ❌ | T-7 |
| Persistence | Route and detector results persisted; resume reuses them. VLM calls are at-most-once (attempt marker). A crash between a detector call and its persist re-runs the detector | ✅ | F-13, F-28 |
| Calibration | Human-checked sample, split by source group | ❌ | T-2 |
| | Isolation + frozen lock utilities | ✅ | F-15 |
| | Isolation enforced in actual runs | ❌ | T-9 |
| | Threshold fitting with CIs; numeric promotion targets | ❌ | B-3, B-4 |
| Evaluation | Detection metrics (mAP, stratified recall, FP/image) | ❌ | T-8 |
| | Primary-only vs always-expensive vs adaptive on a frozen set | ❌ | B-5 |
| | VLM agreement vs human labels | ⚠️ code, no labels | F-24, B-6 |
| | Trainer integration; train-benefit comparison | ❌ substitute loader | B-7 |
| Rollout | Shadow, canary, rollback, gated promotion | ❌ | B-9 |

## Done

| IDs | What | Evidence |
|---|---|---|
| F-1–F-5 | Fixture with non-COCO classes; Pydantic contracts + committed schemas; validators; hash IDs; feature flag | `schemas.py`, `test_schemas.py` |
| F-6–F-10 | Stage A/B features; zero boxes → difficulty `unknown`; pure policy; matching/fusion; native slicer with per-tile budget | acceptance rows 2–6 |
| F-11–F-15 | LM Studio verifier; completeness audit; resumable SQLite cache + error queue; budget caps; frozen lock + isolation utilities | acceptance rows 7–10 |
| F-16–F-17 | Codex review rounds 1–2 fixed; plan path fixed | git history |
| F-18–F-20 | Quality cut-offs in config; policy knobs (route-v1 defaults); per-output-dir job lock | How it works |
| F-21–F-22 | **Behavior changes:** VLM `absent` → review by default; moderate route without a secondary never auto-accepts | How it works |
| F-23–F-25 | `difficulty_uncalibrated` reason; dependency pins; GPU-time rule; VLM-vs-human agreement scoring; CLIP scorer with scale guard | How it works |
| F-26 | R3-1: export uses only the current run | f2dd581 |
| F-27 | R3-2: cross-class conflicts stay in review on VLM `absent` | cd33250 |
| F-28 | R3-3: **at-most-once** VLM calls on resume (behavior change) | f530f98 |
| F-29 | R3-4: missing raw class score never gates low | 5cbbc48 |
| F-30 | R3-5: bare human `complete` can't clear open findings | 0435d99 |
| F-31 | R4-1: incomplete/inconsistent calibrator output never gates low | 770c9d3 |
| F-32 | T-6: calibrator-missing class is unknown, never judged by raw bands | 6a31afd |
| F-33 | P2: calibrator-missing class routes to `human_review` and is listed as suspected | 76455b3 |
| F-34 | Environments managed with pixi (`default`, `clip` with CUDA 12.8 torch) | 2267644 |
| F-35 | Services increment 1: `svc-1.0` wire contract + `RemoteDetector` (existing `Detector` protocol; one POST, hard timeout, no retry; failures raise → error queue/review; model revision and weights digest checked and bound into the run key; pixel cap before any call). 20 tests with mocked HTTP | 7bd7c2e |
| F-36 | Grounding DINO detector service (now `src/services/grounding_dino/`, moved in 4addae9) (own pixi workspace, `svc-1.0`, one forward pass per class so no phrase mapping; weights pinned to HF revision a2bb814 + SHA-256, verified at start; thresholds in the revision string; server-measured `gpu_ms`). Measured on the RTX 5070: ~390 ms/image for 2 classes, 1.76 GiB peak, 9.6 of 12.2 GiB with LM Studio Qwen loaded. Live test behind marker `gdino`; service tests not in CI | e65ff6d |
| F-37 | T-10 INC 1: `WeMMMCPRelevanceScorer` (was `RemoteRelevanceScorer`; the old name is gone; now in `services/wemm_mcp.py`) for the user's WeMM MCP server (`--wemm`, URL from `$VAD_WEMM_URL`, never in git): scale `wemm_text_cosine_v1`, max cosine over positive prompts, longest side ≤ 448 px, one retry on network errors only, errors never become a score of 0, meta checked per response. **Behavior change:** any scorer exception now leaves relevance `unavailable` and the image in review (it was lost to the error queue). 28 tests + live test behind marker `wemm` | 9eeef4c |
| F-38 | `route_report` counts images whose relevance scorer failed (`relevance_scorer_failures`: count + error types); `vad-route shadow` prints a one-line stderr warning when the count is above zero (exit code stays 0). Also: rename commit f86bec0 (WeMM scorer name, clearer verification helper names; run keys unchanged by a golden test) | de1f3ad |
| F-39 | Restructure (user's choice): packages `routing` and `services` under `src/` (no `vad` level, no compatibility shims; console script still `vad-route`); `adapters`/`pipeline`/`embedder` split by responsibility (`interfaces`, `detection_mapping`, `verification_results`, `store`, `image_inputs`, `services/{contracts,http_detector,wemm_mcp,replay,lmstudio,open_clip,runtime}`); `LMStudioVerifier`, `OpenClipRelevanceScorer`; Grounding DINO env stored outside `src/` (pixi detached environments; one config command per fresh clone, see its README). Golden run keys unchanged by tests | 4addae9, 7151324 |
| F-40 | Live three-service test (`pixi run test-e2e`, marker `e2e`, excluded from the default suite): one shadow run with `--primary-url`, `--wemm`, `--vlm` on 5 synthetic images, plus an outage test per service. Live tests now skip with a reason when a service is down (`tests/services/live.py`; `test-lmstudio` used to fail). Run identity recorded all three: Grounding DINO `a2bb814dd30d+bt0.35+tt0.25` with weights digest checked, WeMM `…/d2560@unverified+max448`, VLM `lmstudio:qwen3-vl-4b-instruct`. 0 scorer failures, 11 real VLM verdicts, VLM authored no boxes, only accepted boxes exported. Outages: detector down → image in `report.errors`; WeMM down → relevance unavailable, review; LM Studio down → all uncertain, review | dbc1857 |
| F-41 | C1: OpenCLIP removed (scorer, live test, CLI flags `--clip-*` now fail as unrecognized, `clip` extra/marker/environment/tasks; locks regenerated, default environment packages unchanged). Orphaned `.pixi/envs/clip` and old `.venv/` deleted on the user's instruction | d4738fa |
| F-42 | C2: `OpenAICompatibleVerifier` (was `LMStudioVerifier`): base URL and model explicit, optional bearer token read from an environment variable named by `vlm.api_key_env` (outside `config_hash`, never in repr/traces/identity). Identity still `lmstudio:{model}` (D-11). Golden keys unchanged | d099fbb |
| F-43 | C3: packages flattened to `routing`, `services_embedding`, `service_detector_vlm`, `service_detector_gdino` (replay moved to `routing/replay.py`; `src/services` gone; one wheel). Golden keys unchanged. Older rows below name the old paths: history | 4665884 |
| F-44 | C4: live tests renamed to VLM names; readiness is a real synthetic inference (reachable but failing = FAIL); unreachable = skip, or FAIL with `VAD_LIVE_REQUIRED=1` / `pixi run test-live-required`; https probes port 443. Config from `VAD_GDINO_*`, `VAD_VLM_*`, `VAD_WEMM_URL`. Spec link targets updated. 184 tests | cfac81c |
| F-45 | Stage P: 18 pin tests (`tests/routing/test_pins.py`) for paths the suite never ran (T-14): cached VLM answer on resume and its budget count, `sahi_failed`, sliced-result reuse, accepted candidates not re-verified, missing-class cap, VLM budget skip, errors tagged with their step, missing file, `export-schema` CLI, crowding/hard escalation, completeness status, acceptance-off. `pipeline.py` now 100% line+branch covered; 14 of 15 mutations caught (the miss is an equivalent mutant). No source changes. 202 tests | 3d969dd |
| F-46 | Simplicity review, stages 1 to 4 (behavior identical, each checked by a before/after snapshot of records, cache writes, errors): `cli.main` split into validate / build scorer / run legacy / run shadow (09a7dcd); `_route_image` uses `_load_or_compute_stage_a`, `_run_cached_detector` (62c51cb) and `_run_sliced_detection` (1884661); `_verify` uses `_verify_request` and is annotated (5e96991); WeMM client `_call` split into `_post` / `_decode` / `_check_meta` with one validating score loop, 8 malformed-reply shapes pinned first (6a2c6aa). Stage 5 (clear local names in those functions only, no string literal changed) is the last commit. 218 tests | 09a7dcd–524b92e |
| F-47 | **D-11 resolved (reported by main_agent as the user's decision; user to confirm):** VLM identity is `openai-compatible:{model}@{host[:port]/path}` (URL credentials and scheme left out; the same model on two endpoints never shares results), and `vlm.base_url` / `vlm.model` have no built-in default: enabling the VLM without them exits 2 before any output. **Run keys changed on purpose:** replay+relevance+VLM `rec-bf54146413d69a17` → `rec-876f6df9a5eaad7d`; WeMM `rec-e025bfad645c479b` → `rec-6891a50d24978247` (removing the defaults changes every config hash). Re-supplying the old defaults and identity reproduces the old keys exactly. Old SQLite runs stay readable but are not reused. 223 tests | 504c561 |
| F-48 | Reviewer round 2026-11-10, both P1 fixed: (1) `OpenAICompatibleVerifier` rejects a `base_url` with credentials (error never echoes it; use `vlm.api_key_env`); bearer token/password never reach repr, notes or route traces (401/500/timeout/connection tests); (2) a dedicated slicing detector's model and artifact digest are in `RunInfo` (`sahi_model`, `sahi_artifact_digest`; unset values stay out of the record key, so every run without a dedicated slicer keeps its key; old records load; `RouteRecord` schema regenerated). Also: `pipeline.py` docstring states the real restart guarantee. 244 tests | c873123, bbe3aa1 |
| F-49 | **A3 metrics done (T-8, fixtures only):** `routing/metrics.py` + `vad-route evaluate`. Labels: COCO JSON + `vad_*` fields (`vad_exhaustive` required, `ignore` neutral boxes, `vad_label_provenance: human` declares human labels). Class-aware one-to-one matching, IoU 0.50 working threshold; AP/mAP50–95 (101-point), cross-checked once against pycocotools to 2e-16 (not a dependency); recall by class and COCO size strata; FP per image and per negative image; accepted-box precision/recall (no scores, no AP); missed split into failed images / in review / silent; review coverage; a labelled image without a result is an operational failure counted as missed, never dropped. Output says `NOT A FINDING` unless provenance is declared. 289 tests | 73bbcd0, e93e841, 0de2555 |
| F-50 | **A2 split tooling, stages 1–3a (T-9, T-11; fixtures only):** `vad-route sample group` (exact SHA + dHash Hamming ≤ 6 merge, 7–10 listed for a human, never merged; manifest source ids merge too; unreadable listed, never dropped); exclusions file (`vad_exclusions: 1`, recorded_by, reasons incl. out-of-domain set, digest); `vad-route sample split` (seeded, group-wise, stratified positive/negative, 70/30 then 50/50, refuses without exclusions and labels, refuses a stratum under 3 groups instead of padding, never overwrites a split, writes the frozen lock first); enforcement: `evaluate --split-manifest --frozen-lock` verifies the lock and scores only frozen images, `shadow --split-manifest` tags `vad_split` and refuses unlisted images, `assert_split_allows` for fitting (frozen groups refused by id and group). Stage 3b (`transfer_allowed` guard) waits for D-12; stage 4 (second-review queue) next. 332 tests | 738644f, f1c63e7, 307fc1c |
| F-51 | **A2 stage 4 and A7 stages 1–2 (fixtures only):** second-review queue (`sample review-queue`: all ambiguous + seeded 20% by group) and `sample merge` (second reviewer must differ, model-like handles refused, a disagreement needs an adjudication and the other opinion is kept, provenance becomes `mixed` if the passes differ). A7: `Adjudication` and `ReviewedRecord` schemas (2 new committed files), append-only `adjudications.jsonl`, `vad-route review findings` / `review apply`; a finding closes only through an adjudication naming it, a bare `complete` cannot, failed escalations close only through an explicit `complete` once nothing else is open, a later decision supersedes an earlier one (history kept), rejected boxes not exported, human boxes exported without a score, adjudications on frozen images kept and tagged `vad_split=frozen_eval`; zero adjudications reproduce the original audit exactly. 413 tests | 0862b9a, de38377, 0618d28 |
| F-52 | **A7 complete (T-7), stage 3:** `load_annotations` (stand-in for the training reader) refuses images tagged `frozen_eval` and, given the split manifest, untagged images the manifest freezes (by id or frozen group); one shared `assert_not_frozen_in_split`. End-to-end test: shadow run → `review findings` → full adjudications make every image complete; bare `complete` leaves images with open findings not complete; reviewed export has accepted + human boxes, no rejected box, no scores; replay byte-identical; trace unchanged. Residual gap: D-14. 417 tests | 8f7b07e |
| F-53 | **A5 stage 1 (B-2, fixtures only):** `vad-route fit-relevance`: per-class and any-target logistic on max cosine from human labels + Stage A raw scores, fitted only on calibration-split images (development/frozen images are refused, not filtered), refuses below 10 positives or negatives with the class and counts named, any-target fitted directly and clipped into `[max p_c, min(1, Σ p_c)]`, small ridge on the slope always applied (flagged `separable` when data separate), deterministic artifact with scale, scorer build, prompt hash, label/data/split digests, parameters and counts; never overwrites; says NOT A FINDING without declared human labels. **Real-sample constraint:** a 30–50 image sample split 70/30 then 50/50 leaves roughly 10–18 calibration images, so the 10-per-side minimum will likely refuse some classes or the any-target model until the sample is larger (B-2/B-3). 436 tests | ec61fc5 |
| F-54 | **A5 stage 2 (fixtures only):** `RelevanceCalibratorArtifact` Strict model + committed schema; `FittedRelevanceCalibrator.load` refuses a different scale, scorer (model, build, max_side) or prompt hash; `shadow --relevance-calibrator FILE` (needs a relevance scorer); identity `relevance-calibrator-v1@sha256:<file>` enters the run key only on runs that use a calibrator (golden keys unchanged); a partial missing score leaves status `calibrated`, the class unknown, never `low`, and goes to human review. Exact string match for scorer identity, so an `@unverified` WeMM calibrator loads only for another `@unverified` run; a verified build is a promotion gate. 452 tests | 6461ef1 |
| D-8 | Work committed incrementally; author email rewritten to gmail (user's choice) | — |

## Acceptance tests (from the spec)

| Test | Status | Tests |
|---|---|---|
| Arbitrary class registry | ✅ | `test_arbitrary_class_flows_end_to_end`, `test_class_mapping_for_secondary` |
| Independence of relevance and difficulty | ✅ | `test_relevance_and_difficulty_are_independent` |
| Zero detections never easy / accepted negative | ✅ | `test_high_relevance_zero_boxes_is_never_easy_or_negative` + 2 |
| Stable routing across restarts | ✅ | `test_routing_replays_identically_across_processes` |
| Directional disagreement + provenance | ✅ | `test_directional_disagreement_and_provenance` |
| SAHI coordinates | ✅ fake detector | `test_sahi_*` |
| VLM failures → `uncertain`, caps hold | ✅ | `test_vlm_*`, `test_r3_*` |
| Completeness review states | ✅ | `test_unmatched_proposal_creates_review_state` + 3 |
| Budget and recovery | ✅ | `test_resume_after_crash_*`, `test_r4_*` |
| Data isolation | ✅ utility level (T-9) | `test_split_groups_must_not_overlap`, `test_frozen_set_is_locked_*` |
| End-to-end into the training pipeline | ⚠️ substitute loader (B-7) | `test_end_to_end_shadow_and_legacy` |

## How it works and why

**Decision flow (as built):**

1. **Stage A** (`scoring.stage_a`): image validity, blur/exposure, per-class relevance (`calibrated` / `uncalibrated` / `unavailable`; band `low` / `medium` / `high` / `unknown`). A missing or incomplete class score never produces `low`.
2. **Primary detector**, then **Stage B**: box features and difficulty. Zero boxes give difficulty `unknown`, never `easy`.
3. **`policy.route`** (pure, ordered rules):
   - Low relevance, zero boxes: `standard` plus a hash-sampled audit. Nothing is discarded.
   - Low relevance with detections: `vlm_verify` (or `human_review` without a VLM).
   - High relevance and zero boxes (unknown relevance too if `unknown_relevance_as: high`), or high miss risk (hard, small box, crowding): escalate to `sahi`, `secondary`, `vlm_verify`, `human_review` as available.
   - Moderate or unknown relevance (default), or medium/unknown difficulty: `secondary`, then VLM or human; without a secondary, VLM or human, never auto-accept.
   - High relevance, low difficulty: `standard`.
   - A calibrated class with no probability: `human_review`.
   - Slice, VLM-call, elapsed or GPU caps stop escalation and leave `needs_review`.
4. **Fuse and verify:** class-aware fusion keeps every source ID; the VLM only answers `present` / `absent` / `uncertain` and never creates or moves a box.
5. **Completeness audit:** `complete` needs a human or a calibrated audit policy and nothing left open. Only `accepted` boxes are exported.

**Behavior changes:**

| Change | Why | Knob |
|---|---|---|
| VLM `absent` → review, not reject (F-21) | 4B answers `absent` on blank crops; a wrong reject silently drops a box. | `vlm.absent_rejects: false` |
| Moderate route without a working secondary never auto-accepts (F-22) | Nothing confirmed the boxes; they go to VLM or review. | none |
| Interrupted VLM request → `uncertain`, not re-sent (F-28) | Attempt marker before the call; trades a lost answer for no repeat call. | none |
| Calibrator-missing class is unknown → `human_review` (F-32, F-33) | Raw and calibrated scales must not mix. | none |
| New config fields changed `config_hash` | Old `runs/*/routes.sqlite` caches are not reused. | none |

**Design notes:**

- **Policy knobs** (`policy:`; defaults = route-v1): `unknown_relevance_as`, `moderate_without_secondary`, `cross_class_always_review`. Poor-quality cut-offs: `quality.poor_blur` / `poor_exposure`.
- **Job lock:** `vad-route shadow` holds an OS byte-range lock on `<out>/.vad-route.lock`; a second run exits with code 2. No stale-lock cleanup needed; PID probing is unsafe on Windows (`os.kill(pid, 0)` terminates).
- **GPU cap:** enforced only on adapter-measured device time; the elapsed cap covers wall time.
- **CLIP scorer (removed in F-41; kept as history):** local checkpoint, never downloads. Score `clip_prompt_softmax_v1` in [0, 1], uncalibrated. Raw cosines sit around 0.22–0.34, so a scorer declaring a scale is refused unless `relevance.raw_scale` matches. The guard covers only the live scorer, not replayed JSONL. About 9 ms GPU per 320×240 image.
- **VLM agreement:** confusion table and Wilson 95% lower bounds per model; labels keyed by what was asked, so one label set scores several VLMs. Model-authored or conflicting labels are refused.

## Integration notes (greenfield)

There was no existing detector repository, so each interface is a neutral default behind an adapter: JSONL or directory manifest; `task:` class registry in YAML (free-text IDs); `PrecomputedDetector` replay (explicit row per image, `[]` = zero boxes); COCO-style export of `accepted` boxes; LM Studio `qwen3-vl-4b-instruct` verifier; local CLIP relevance.

| # | Assumption | Status |
|---|---|---|
| 1 | No real image evaluation of any detector: remote client and Grounding DINO server exist, tested on synthetic images only | **Open** → T-1 |
| 2 | Text–image relevance: CLIP and the WeMM server exist; bands still placeholders for both | **Open** → D-5 |
| 3 | No calibrators; nothing auto-marked `complete`; shadow review rate ≈ 100% by design | **Open** → B-2 |
| 4 | GPU time only from adapters that measure it | **Decided** → D-6 |
| 5 | COCO-style export until a trainer exists | **Open** → B-7 |
| 6 | CLI model versions unverified; file digests bound into the run key | **Partly resolved** → T-1 |
| 7 | VLM is 4B | **Open** → D-7 |

**Hardware:** RTX 5070 (12 GB), Python 3.13; CUDA 12.8 torch lives only in the Grounding DINO workspace. **Fixture:** `tests/routing/helpers.make_fixture(dir)`: 3 images + 1 corrupt file, manifest, predictions, relevance; non-COCO classes `blue_widget_x9`, `zz-custom.class_7`.

## How to run

```bash
pixi install
pixi run test                   # 452 tests, no network / GPU (live: test-vlm, test-gdino, test-wemm, test-e2e, test-live-required)
pixi run test-vlm               # live VLM (any OpenAI-compatible endpoint)
pixi run vad-route shadow --config configs/routing/example.yaml --images manifest.jsonl \
  --primary-predictions preds.jsonl --primary-version mymodel@sha --out runs/shadow1
```
