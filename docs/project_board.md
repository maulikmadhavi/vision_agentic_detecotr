# Adaptive routing for custom object detection: project board

**Updated:** 2026-11-09 · **Owner:** user · **Implementer:** Claude · **Review log:** [reviewer_suggestion.md](reviewer_suggestion.md)

**Status:** phases 0–4 done, Phase 5 started, Phase 6 not started. Routing is off by default (`routing_enabled: false`). `pixi run test`: 119 passed; live `test-clip` and `test-lmstudio`: 1 passed each. Environments: pixi only. Git: tags `v0.1-phase4-baseline` (208713e) and `v0.2-r3-blockers-fixed` (0435d99); `backup/pre-author-rewrite` holds the pre-rewrite history.

The requirements and acceptance criteria are in [specification.md](specification.md); this board tracks their status only.

## Phases

| Phase | Status | What's left |
|---|---|---|
| 0. Map pipeline + fixture | **Done** (greenfield) | Baseline metrics need a real detector (T-1). |
| 1. Contracts + trace | **Done** | — |
| 2. Stage A + Stage B | **Done**, gaps | Compression and domain-distance signals (T-4); shadow only on the fixture so far (B-1a). |
| 3. Router + secondary + SAHI | **Done**, gaps | SAHI tested with fake detectors only; no real secondary adapter (T-1, B-8). |
| 4. VLM + completeness | **Done**, gaps | No way to feed reviewer outcomes back in (T-7). |
| 5. Calibration + evaluation | **Started** | Utilities only; needs labelled data (T-2), metrics (T-8), calibrators and benchmark (B-2…B-6). |
| 6. Rollout | **Not started** | B-9. |

## Needs your decision

| ID | Decision | Default now | Recommendation |
|---|---|---|---|
| D-1 | `policy.unknown_relevance_as`: route unknown relevance like `medium` or `high` | `medium` | Keep for shadow; `high` only if relevance is often missing in production. |
| D-2 | `policy.moderate_without_secondary`: `vlm_then_human` or `human` | `vlm_then_human` | Keep. |
| D-3 | `policy.cross_class_always_review` | `true` | Keep until a secondary detector exists. |
| D-4 | `vlm.absent_rejects`: let a VLM "absent" reject a box | `false` | Set a target for `absent_precision_wilson_lower95` before measuring (B-6). |
| D-5 | CLIP relevance bands on the `clip_prompt_softmax_v1` scale | none (`raw_scale: unspecified`) | Choose on the calibration split, or fit a calibrator (B-2, B-3). |
| D-6 | GPU budget counts only adapter-measured device time | measured only | Confirm. |
| D-7 | VLM model | `qwen3-vl-4b-instruct` (8B not loaded) | Load 8B and compare on real crops (B-6). |
| D-9 | Delete `backup/pre-author-rewrite` and `refs/original/*` (old NCS-email history), and the unused `.venv/` | kept | Delete once you're sure. |

## To do — ready now

| ID | Card | Why |
|---|---|---|
| T-1 | **Evaluate the Grounding DINO service on real images.** The server exists (F-36) and `--primary-url` works; what is missing is the user's 30–50 trial images, human labels, and choosing prompts/thresholds on them (it boxes something for every prompt: absent class 0.71, blank image 0.83 on synthetic tests). Hosted APIs remain unbuilt: they need a normalizing adapter, terms/data-handling check, env-var credentials. | Unblocks real shadow runs, SAHI, the benchmark and rollout. |
| T-2 | **Build the human-checked sample** (class, domain, resolution, size, crowding, positives/negatives, low similarity); split calibration / development / frozen by source group; lock the frozen set. | Every Phase 5 card needs it. |
| T-3 | **Independent review** (human or Codex) of `pipeline._candidates`, `_verify`, `_apply`, `policy.route`. | Same model family wrote and reviewed the code. |
| T-4 | **Stage A gaps:** compression indicator; domain-distance when a reference distribution exists (`StageA.domain_distance` is never filled). | Spec Stage A list; domain-shift audit sampling (rule 7). |
| T-7 | **Reviewer-outcome ingestion:** a way to record human decisions per finding (accept/reject box, class adjudication, "complete") and re-audit. `completeness.audit` accepts `human_outcome` / `reference_boxes`, but no pipeline or CLI path passes them. | The human queue is one-way today; R3-5 noted the missing adjudication record. |
| T-8 | **Detection metrics:** mAP50–95, class- and size-stratified recall, FP/image, annotation precision/completeness against ground truth. Only route/cost reports and VLM agreement exist. | Needed for the frozen benchmark (B-5) and baseline metrics. |
| T-9 | **Enforce split isolation in runs:** `vad-route shadow` takes no split or frozen lock, so `check_isolation` / `assert_not_frozen` are never applied outside tests. | Spec: frozen images must never reach training or threshold fitting. |

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
| | Per-class text–image similarity with positive/negative prompts | ✅ | F-25 (CLIP) |
| | Calibrated relevance probabilities | ⚠️ protocol + guards, no calibrator | B-2; F-31–F-33 |
| Stage B | Box area, crowding, confidence, ambiguity, count, high-relevance/zero-box contradiction | ✅ | F-7 |
| | `difficulty_pre_secondary` and `difficulty_final` both kept | ✅ | F-7 |
| | Calibrated difficulty | ❌ heuristic only | B-2 |
| Detectors | Primary detector through an adapter | ⚠️ precomputed replay only | T-1 |
| | Optional independent secondary | ⚠️ protocol, no real adapter | B-8 |
| | SAHI with original-coordinate mapping | ⚠️ native slicer, fake detectors only | F-10, T-1 |
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
| F-36 | Grounding DINO detector service in `third_party_service/` (own pixi workspace, `svc-1.0`, one forward pass per class so no phrase mapping; weights pinned to HF revision a2bb814 + SHA-256, verified at start; thresholds in the revision string; server-measured `gpu_ms`). Measured on the RTX 5070: ~390 ms/image for 2 classes, 1.76 GiB peak, 9.6 of 12.2 GiB with LM Studio Qwen loaded. Live test behind marker `gdino`; service tests not in CI | e65ff6d |
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
- **CLIP scorer:** local checkpoint, never downloads. Score `clip_prompt_softmax_v1` in [0, 1], uncalibrated. Raw cosines sit around 0.22–0.34, so a scorer declaring a scale is refused unless `relevance.raw_scale` matches. The guard covers only the live scorer, not replayed JSONL. About 9 ms GPU per 320×240 image.
- **VLM agreement:** confusion table and Wilson 95% lower bounds per model; labels keyed by what was asked, so one label set scores several VLMs. Model-authored or conflicting labels are refused.

## Integration notes (greenfield)

There was no existing detector repository, so each interface is a neutral default behind an adapter: JSONL or directory manifest; `task:` class registry in YAML (free-text IDs); `PrecomputedDetector` replay (explicit row per image, `[]` = zero boxes); COCO-style export of `accepted` boxes; LM Studio `qwen3-vl-4b-instruct` verifier; local CLIP relevance.

| # | Assumption | Status |
|---|---|---|
| 1 | No real primary detector; shadow replays saved predictions | **Open** → T-1 |
| 2 | No text–image relevance model | **Resolved** (F-25); bands still placeholders → D-5 |
| 3 | No calibrators; nothing auto-marked `complete`; shadow review rate ≈ 100% by design | **Open** → B-2 |
| 4 | GPU time only from adapters that measure it | **Decided** → D-6 |
| 5 | COCO-style export until a trainer exists | **Open** → B-7 |
| 6 | CLI model versions unverified; file digests bound into the run key; CLIP hashes its weights | **Partly resolved** → T-1 |
| 7 | VLM is 4B | **Open** → D-7 |

**Hardware:** RTX 5070 (12 GB), Python 3.13; the pixi `clip` env adds CUDA 12.8 torch + open_clip. **Fixture:** `tests/routing/helpers.make_fixture(dir)`: 3 images + 1 corrupt file, manifest, predictions, relevance; non-COCO classes `blue_widget_x9`, `zz-custom.class_7`.

## How to run

```bash
pixi install                    # add `-e clip` for the CLIP scorer
pixi run test                   # 119 tests, no network / GPU
pixi run -e clip test-clip      # live CLIP (local weights)
pixi run test-lmstudio          # live VLM (LM Studio)
pixi run vad-route shadow --config configs/routing/example.yaml --images manifest.jsonl \
  --primary-predictions preds.jsonl --primary-version mymodel@sha --out runs/shadow1
```
