# Reviewer suggestions: adaptive object detection routing

**Reviewer:** Claude (reviewer agent) · **Date:** 2026-11-08 · **Plan under review:** [project_board.md](project_board.md) Part B (formerly `adaptive_object_detection_plan.md`)

> [!NOTE]
> **Update 2026-11-08 (later the same day):** the user confirmed this is greenfield and that Codex cannot proceed. Claude implemented phases 1–4 instead, with **qwen3-vl-4b-instruct** (LM Studio) as a **verifier-only** VLM. Sections 1–2 describe the state *before* that. Section 5 records the decisions and checks the code against the section 3 checklist. Because the implementer and the reviewer are the same agent here, an independent second review is recommended (see 5.4).

## 1. Status: no source code to review yet

Snapshot of the workspace on 2026-11-08 at about 13:10 SGT:

```
vision_agentic_detector/
└── docs/
    └── adaptive_object_detection_plan.md
```

No Python files, configs, tests, fixtures, or JSON Schemas exist. The Codex session started at 13:07 (`~/.codex/sessions/2026/11/08/rollout-2026-11-08T13-07-09-*.jsonl`) did not write any code. It is **blocked on a `PromptForUserInput` dialog titled "Locate the detector repository"**. Its last message:

> "I'm waiting for the detector repository path so the adapters, fixture, and tests can use the actual pipeline."

Codex was right to stop. The plan's first task is to "locate the current image manifest, class registry, primary detector call, … Wrap those interfaces", and the execution prompt says "Do not assume YOLO, COCO classes…". None of those interfaces exist in this folder, so building the package here would mean inventing them.

**Action needed from the user:** answer the Codex prompt with the path to the existing detector repository. The other option is to tell Codex this is a greenfield project. In that case the plan's Phase 0 has to be rewritten, because there is no "existing path" to keep behind a feature flag.

## 2. Problems in the hand-off itself (fix before Codex resumes)

| # | Severity | Issue | Suggestion |
|---|---|---|---|
| H1 | High | The Codex execution prompt points to `outputs/adaptive_object_detection_routing_plan.md`, but the plan is actually at `docs/adaptive_object_detection_plan.md`. | Change the path in the prompt in the plan (around line 148) so later runs don't get stuck the same way. |
| H2 | High | The prompt says "in my existing … repository", but the workspace has no repository. | Give the repository path explicitly, or open Codex inside that repository with this plan copied in. |
| H3 | Medium | The workspace is not a git repository. A reviewer can't diff Codex's changes, and Codex can't make per-phase commits. | `git init` (or work in the detector repo's git) so that each phase from §6 lands as its own reviewable commit. |
| H4 | Medium | Codex's write sandbox only covers this folder. If the detector repo is somewhere else, Codex can read it but can't write there. | Run Codex with the detector repo as its workspace root. |

## 3. Review checklist for when code arrives

I will check the implementation against these points, taken from the plan. Codex can also use them as a self-check before handing back.

### Contracts (`schemas.py`, `task_config.py`)
- [ ] Pydantic v2 models. JSON Schema is exported to a committed file, and a test fails if the schema drifts.
- [ ] Validators reject: unknown `class_id`, NaN/inf scores, `x1 >= x2` / `y1 >= y2`, boxes outside `[0,width]×[0,height]`, missing `model_version`, any coordinate system other than `xyxy` pixels.
- [ ] `schema_version` and a content-hash ID are on **every** persisted record, not only the top-level one.
- [ ] Separate fields for `raw_similarity`, `relevance_probability`, `raw_confidence`, and difficulty. No field named or documented as "calibrated" holds a raw model output.
- [ ] `stage_b` stores both `difficulty_pre_secondary` and `difficulty_final`. Look for code that overwrites the first one.
- [ ] Difficulty can take the value `unknown`. Check that it's not just `Optional[float]` with `None` quietly treated as 0.
- [ ] `VerificationResult.verdict` is `Literal["present","absent","uncertain"]`.

### Policy (`policy.py`)
- [ ] Pure function with no I/O, clock, or global RNG, and no reads from env or config inside the function. Inputs are only typed scores plus the config.
- [ ] Rules are ordered and data-driven. Every output includes actions, priority, reason codes, and expected budget.
- [ ] Zero boxes leads to `difficulty=unknown`, never `easy` and never an accepted negative. High relevance with zero boxes leads to a secondary pass, SAHI, or review.
- [ ] Low-relevance audit sampling uses `hash(image_id, policy_seed)`. Watch for Python's built-in `hash()`: it is salted per process (PYTHONHASHSEED) and breaks deterministic replay across restarts. Use `hashlib`.
- [ ] No automatic discard path is enabled, or one exists only behind a flag that is off by default.
- [ ] Missing secondary detector, SAHI, or VLM falls back to `needs_review`, not `standard`.

### Matching and fusion (`matching.py`, `slicing.py`)
- [ ] Slice boxes are converted to original coordinates **exactly once**, before matching. Look for double offset or clipping before the offset is added.
- [ ] Converted boxes are clipped to the image bounds and still pass the schema validator. Degenerate boxes that clip to zero area are dropped, and the drop is logged.
- [ ] IoU threshold and class mapping come from config, with no hard-coded `0.5`.
- [ ] Disagreement records direction and count: primary-only, secondary-only, matched-but-mislocalized, cross-class. A single boolean is not enough.
- [ ] Fusion keeps every source prediction (`source_ids`), and the tie rule is documented and tested. NMS does not delete candidates that are under audit.
- [ ] Missing secondary detector gives `disagreement.status = "unavailable"`, not zero disagreement.

### VLM and completeness (`adapters.py`, `completeness.py`)
- [ ] Malformed, timed-out, contradictory, or unavailable VLM responses all map to `uncertain`. There are no recursive retries, and the call cap is enforced.
- [ ] There is no code path where VLM output produces an `accepted` box with new coordinates.
- [ ] Completeness status is one of `complete | needs_review | unresolved` and records who established it (human or audit policy). "All model boxes accepted" never implies `complete`.

### Pipeline (`pipeline.py`)
- [ ] The route decision is persisted **before** expensive calls. On resume it skips calls that already finished, with no duplicate VLM calls or annotation rows. A test kills the job partway through and resumes it.
- [ ] Caps on slices, VLM calls, elapsed time, and GPU time are checked before each escalation. When a cap is hit, the result is `needs_review` with the trace attached.
- [ ] Corrupt images go to an explicit error queue and don't crash the batch.
- [ ] The feature flag is off by default, and the legacy path gives byte-identical output with the flag off (golden test).

### Evaluation and data isolation (`evaluation.py`)
- [ ] Frozen-set image IDs and annotations are hash-locked, and a test asserts no overlap by group ID with training or calibration.
- [ ] Changing a threshold cannot rewrite frozen annotations. Check that frozen data is read-only in code, not just by convention.
- [ ] VLM agreement is scored against human labels, never against itself.

### Tests (acceptance table, plan §Acceptance tests)
- [ ] All 11 rows have at least one named test. For the arbitrary-class row, use a made-up class ID (e.g. `"blue_widget_x9"`) so any hidden COCO or YOLO mapping fails the test.
- [ ] The deterministic-replay test runs in a **subprocess** with a different `PYTHONHASHSEED`. Running it twice in one process proves nothing.

## 4. Smaller notes on the plan (optional)

- Stage A asks for "calibrated relevance probabilities when a calibrator exists". Before Phase 5 there will be no calibrator. Make sure the `uncalibrated` label goes through to the route reason codes so shadow-mode reports don't mix the two kinds of value.
- `trace.gpu_ms` is hard to measure without a CUDA-event wrapper inside the detector adapter. Agree early whether wall-time is an acceptable proxy, so the GPU budget cap isn't enforced against a number nobody measures.
- The plan names no dependency pins. Pin `pydantic>=2,<3` and the SAHI version in the repo, because the plan says schemas must be exported "from the pinned version".

---

*Next review:* once Codex has the repository path and delivers Phase 0 (integration map and baseline fixture), I'll review that before Phases 1–4 begin, and add per-file findings under a new section 5.

## 5. Implementation review (2026-11-08)

### 5.1 Decisions recorded

| Decision | Choice | Source |
|---|---|---|
| Existing repository | None. Greenfield. Integration defaults are listed in [project_board.md](project_board.md) Part C. | user |
| Implementer | Claude (Codex blocked) | user |
| VLM role | Verifier only: `present / absent / uncertain`. It never produces a box. | user |
| VLM model | `qwen3-vl-4b-instruct` via LM Studio. 8B isn't loaded; switch with `vlm.model`. | user |
| Primary detector | Not chosen. `PrecomputedDetector` replays predictions from any model. | implementer, flagged as unresolved |

### 5.2 Checklist results (section 3)

Test run: `67 passed` (unit and acceptance), and `1 passed` for the live LM Studio test (`pytest -m lmstudio`).

| Area | Status | Evidence |
|---|---|---|
| Pydantic v2 + committed JSON Schema + drift test | ✅ | `schemas/*.schema.json`, `test_committed_json_schemas_match_models` |
| Rejects unknown class, NaN/inf, bad boxes, missing model version, other coordinate systems | ✅ | `test_detection_rejects_invalid`, `test_detection_context_checks`, `RouteRecord._registry_and_bounds` |
| `schema_version` + hash IDs on persisted records | ✅ | `RouteRecord`, `ErrorRecord`; detection, candidate and request IDs are SHA-256 based |
| Raw vs calibrated kept apart | ✅ | `raw_similarity` / `relevance_probability` / `raw_confidence`; `relevance_status` |
| `difficulty_pre_secondary` preserved | ✅ | `test_directional_disagreement_and_provenance` asserts pre ≠ final |
| Difficulty `unknown` is explicit (not `None` treated as 0) | ✅ | `Difficulty` validator forces status, value and bucket to agree |
| VLM verdict literal, no box field | ✅ | `VerificationResult` (extra=forbid), `test_vlm_valid_answer_cannot_carry_a_box` |
| Pure policy, `hashlib` sampling, replay across `PYTHONHASHSEED` in subprocesses | ✅ | `policy.py`, `test_routing_replays_identically_across_processes` |
| Zero boxes → unknown, never easy or accepted negative; missing capacity → review | ✅ | row 3 tests |
| No auto-discard | ✅ | No discard action exists. Low-relevance zero-box images get `negative_not_verified`. |
| Slice offset applied once, clipped, degenerate boxes logged | ✅ | `adapters.to_detections(origin, clip_to)`, SAHI tests |
| Config-driven IoU and class mapping | ✅ | `MatchingConfig`, `test_class_mapping_for_secondary` |
| Directional disagreement; `unavailable` without secondary | ✅ | row 5 tests |
| Fusion keeps all sources; tie rule documented | ✅ | `matching.canonical_key`; the union of `source_ids` equals all detections |
| VLM failures → uncertain, one attempt, cap enforced | ✅ | row 7 tests (malformed, extra `box` key, wrong class, 500, timeout, refused, raising adapter) |
| `complete` only by human or calibrated audit | ✅ | `AnnotationAudit` validator, `test_complete_requires_human_or_calibrated_policy` |
| Route persisted before expensive calls; resume without duplicate calls | ✅ | `test_resume_after_crash_does_not_repeat_calls` (crash mid-VLM, then resume) |
| Budget caps before each escalation | ✅ | `test_slice_budget_blocks_sahi`, `test_elapsed_and_gpu_caps_stop_escalation` |
| Corrupt image / adapter crash → error queue, batch continues | ✅ | `test_corrupt_image_goes_to_error_queue`, `test_one_failing_image_does_not_abort_the_batch` |
| Flag off by default; legacy output identical | ✅ | `test_feature_flag_off_keeps_legacy_output` (golden against the predictions file) |
| Frozen set hash-locked, isolation by group | ✅ | row 10 tests |
| "Exported annotations load in existing trainer" | ⚠️ substitute | No trainer exists. Uses `export.load_annotations`. |
| VLM agreement vs human labels | ⏳ Phase 5 | No human-labelled set exists yet |

### 5.3 Bugs found and fixed during implementation

1. **SAHI merge left cut objects split.** Greedy clustering compared each box only with its cluster head, so an object cut by 3+ tiles came out as two boxes. Fixed by merging against each cluster's growing union until nothing changes (`slicing.merge`). Covered by `test_sahi_cut_object_is_reassembled_by_full_image_pass`.
2. **Error-record ID depended on the absolute path**, which broke replay across directories. It now uses image ID + content hash + config hash.
3. **Hard route with only a VLM accepted primary boxes unverified.** Found in a live shadow run, where a 0.4-confidence tiny box was accepted. Primary boxes on a `high_miss_risk` route with no extra detector now go to `needs_review` and are sent to the VLM (`test_hard_route_without_extra_detector_verifies_primary_boxes`).

4. **One failing image aborted the whole batch.** Only image loading was guarded. A raising primary adapter, a NaN or `1.0000001` confidence, or a sub-pixel box that rounds to zero width would crash `run()`. Fixed: values are rounded before the degenerate check, bad confidences are dropped or clamped with a trace warning, and any failure after loading becomes an `ErrorRecord` naming the step (`routing_failed_at:<step>`).

### 5.4 Open concerns (for the next reviewer)

- ~~**SAHI provenance gap.**~~ Fixed in reviewer round 1; see 5.5.

- **Self-review bias.** The same agent wrote and reviewed this code. Have Codex or a human review `pipeline.py` (`_candidates`, `_verify`, `_apply`) and `policy.py` before relying on them.
- **The shadow review rate is about 100% by design** until calibration (`completeness_not_established`). That is correct but noisy. Use `route.priority` and `reason_codes` to triage.
- **Policy choices the plan left open** (please confirm):
  - medium relevance with no secondary detector goes to VLM, or to human review if there is no VLM;
  - unknown relevance is treated like moderate;
  - a cross-class conflict always needs review, even if the VLM confirms the box.
- **Heuristic thresholds** (blur ≥ 0.8, exposure ≥ 0.5 count as poor quality, in `policy.py`) are hard-coded constants. Move them into config when Phase 5 starts.
- `RouteStore` uses one SQLite connection with no locking across processes. Run shadow jobs one at a time per output directory.
- The live VLM call on the 4B model tends to answer `absent` on blank images with reasons like "No visual content". That is fine, but measure 4B against 8B on real crops before trusting `absent` to reject detector boxes.

### 5.5 Reviewer round 1 (Codex) — fixes

| # | Finding | Fix | Test |
|---|---|---|---|
| 1 | Merged SAHI `parent_ids` pointed at raw tile detections that were never stored | Raw tile detections are now stored in the `sahi` step and in `RouteRecord.sahi_raw_candidates`. The validator rejects records whose merged boxes cite missing parents. | `test_r1_sahi_merged_parents_are_persisted` |
| 2 | Stage A was recomputed on resume while the persisted route was reused | Stage A is cached as the `stage_a` step and reused on resume | `test_r2_resume_reuses_persisted_stage_a` |
| 3 | A custom VLM result was checked only for request ID and class | `_answers()` also requires the same `candidate_ids` and `region_xyxy`. Any mismatch becomes `uncertain` / `vlm_contradictory`. | `test_r3_vlm_result_must_match_request_candidates_and_region` (3 cases) |
| 4 | SAHI budget was checked only before the whole multi-tile call | `run_sliced(max_slices, stop_check)` checks slice, elapsed and GPU caps before every tile. A partial result adds `sahi_partial:<reason>`, which leads to review. | `test_r4_sahi_{elapsed,gpu}_budget_enforced_per_tile`, `test_r4_sahi_slice_cap_enforced_during_tiling` |
| 5 | CLI silently fell back to `secondary` as the model version | `--secondary-version` is now required whenever `--secondary-predictions` is given | `test_r5_cli_requires_secondary_version` |

### 5.6 Reviewer round 2 (Codex) — fixes

| # | Finding | Fix | Test |
|---|---|---|---|
| 1 | `PrecomputedDetector` treated a missing row as zero detections | An explicit row is now required, and `"detections": []` is valid. A missing row raises `MissingPrediction`, and the image goes to the per-image error queue (`routing_failed_at:primary`). Rows without `image_id` or `detections`, and duplicate image IDs, are rejected at load. A missing *relevance* row gives relevance `unavailable`, never low. | `test_r2_missing_prediction_row_goes_to_error_queue_and_batch_continues`, `test_r2_explicit_empty_row_is_a_valid_zero_box_result`, `test_r2_precomputed_rows_must_be_explicit_and_unique` (3 cases) |
| 2 | Caller-supplied versions could reuse stale records after a file changed | Documented that versions are unverified identifiers (README, integration map, CLI `--help`). Each precomputed file's SHA-256 is added to `RunInfo.{primary,secondary,relevance}_artifact_digest`, which feeds the record key. | `test_r2_changed_precomputed_file_with_same_version_does_not_reuse_records` |

Remaining limitation: the digest covers the predictions file, not the model weights that produced it. `legacy` mode stops on a missing row rather than queuing it, because it has no store or error queue.

## 6. Independent critical review (2026-11-09)

This round is a source review of the current implementation. The findings below are open; the earlier green checkmarks and test counts do not resolve them. I did not change implementation code.

| # | Severity | Finding and impact | Reviewer comment |
|---|---|---|---|
| R3-1 | **High** | **A reused output directory can export stale or incompatible annotations.** `cli.py:81-89` runs the current manifest, then calls `store.records()` and `store.errors()` without filtering to this run. `RouteStore.records()` in `pipeline.py:108-110` returns every stored record. A second shadow run with a different manifest, task, config, or prediction file can silently include earlier images in `annotations.json`, `route_trace.jsonl`, and `report.json`; a changed task can also fail at category lookup. | Build exports and the report from the records returned by the current `pipe.run(entries)`, or give runs an explicit ID and query by it. Test two runs sharing one `--out` directory with different manifests and tasks. |
| R3-2 | **High** | **Cross-class conflicts can be silently rejected.** `_candidates()` flags overlapping different-class clusters, but `_apply()` in `pipeline.py:446-451` handles a VLM `absent` verdict before checking `cross_class_conflict`. Both conflicting candidates can become `rejected`, despite section 5.4 saying a cross-class conflict always needs review. | Keep conflicted candidates in `needs_review` until a human adjudicates the class and geometry. Add a test with overlapping primary and secondary boxes of different classes and VLM `absent` answers. |
| R3-3 | **High** | **The “no duplicate VLM calls on resume” claim is stronger than the implementation.** `pipeline.py:396-409` calls the external VLM, then writes the result to SQLite. A process kill after the call completes but before `put_step()` commits causes a second call on restart. The existing resume test covers a different crash boundary. | Document at-least-once behavior, or use a durable request state and an idempotency key supported by the VLM service. Add a fault-injection test at the call/persist boundary; do not assert exactly-once execution without an external deduplication mechanism. |
| R3-4 | **Medium** | **Partial calibrated relevance can produce a falsely low route.** `scoring.py:57-79` warns when scorer output omits task classes, but with a calibrator it still assigns `relevance_status="calibrated"` and bands the image using `any_p`. The missing-class guard that prevents an uncalibrated low band is only in the later branch. A high-relevance class with no score can therefore be treated as a low-relevance zero-box image. | Require complete per-class scorer/calibrator output before a calibrated low band, or route incomplete output as `unknown`/`medium`. Test a missing class with a low `any_p`. |
| R3-5 | **Medium** | **A `human_outcome="complete"` bypasses every unresolved signal.** `completeness.py:85-94` sets `complete` before checking unmatched candidates, VLM-present missing classes, uncertain verification, or failed escalations. The stored record can simultaneously say `complete` and retain those open findings. | Define an explicit adjudication record that resolves or overrides each finding, with reviewer identity and reason. Until then, do not let the bare string `complete` clear unresolved evidence. |

The current CLI has no calibration or frozen-evaluation workflow; `evaluation.py` provides utilities but the shadow command does not call them. Treat the section 5.2 frozen-set checkmark as **utility-level coverage**, not evidence that production shadow runs enforce split isolation. The actual primary detector, trainer integration, and human-labelled evaluation remain unverified as noted above.

## 7. Project-board progress review (2026-11-09)

Scope: compared `project_board.md` Parts A–D with the present source and tests. Read-only verification: `.venv/Scripts/python.exe -m pytest -q` produced **82 passed, 2 deselected**. The two deselected tests require optional CLIP weights and LM Studio; I did not rerun either live integration in this review. Git has **no commits**; the working tree contains staged additions/modifications and untracked files. This section is review feedback only.

| Board phase | Reviewer assessment | Evidence and qualification |
|---|---|---|
| 0 | **Greenfield map and fixture done; baseline metrics unavailable.** | Part C identifies chosen interfaces and a tiny fixture. There was no existing detector/trainer to map or benchmark. |
| 1 | **Implementation and contract tests done.** | Pydantic contracts, schema drift test, disabled feature flag and trace tests pass. |
| 2 | **Heuristic scoring and optional CLIP scorer built; calibration incomplete.** | `scoring.py` and `embedder.py` exist. Compression/domain-distance signals and a human-fit relevance/difficulty calibrator remain open. R3-4 below also prevents treating calibrated low-relevance routing as safe. |
| 3 | **Policy, fusion and native slicing built; real-detector validation open.** | Deterministic replay and coordinate/budget tests pass with scripted detectors. No crop-capable production detector or independent secondary is integrated. |
| 4 | **VLM adapter, queue states and persistence built; operational guarantees need correction.** | Local LM Studio test is reported by the board, but was not rerun here. R3-1 through R3-3 and R3-5 remain open; therefore “done” means feature code exists, not that the export/recovery path is cleared for unattended use. |
| 5 | **Utility groundwork only.** | Frozen-set lock/isolation and VLM agreement functions have tests. No human-checked split, fitted thresholds, frozen benchmark or three-way comparison exists. |
| 6 | **Not started.** | No canary, promotion gates or rollback exercise. |

### Board comments to apply

1. **Promote R3-1 and R3-2 to release blockers in Part A.** A reused output directory can mix past records into a new export, and an enabled `vlm.absent_rejects` can reject cross-class conflicts before the mandatory-review check. These are concrete correctness risks, not merely Phase 5 calibration tasks. Keep R3-3 (at-least-once external VLM calls), R3-4 (incomplete calibrated scorer output) and R3-5 (bare human `complete` override) visible alongside them. See §6 for paths, impact and suggested tests.
2. **Qualify F-12, F-13 and F-14 rather than marking the whole operational path as verified.** `AnnotationAudit` checks who established `complete` but not whether unresolved findings were adjudicated (R3-5). SQLite step caching handles tested resume boundaries, not a kill between external VLM response and persistence (R3-3). The budget tests show stopping behavior; they do not remove the export mixing risk (R3-1). The 82 passing tests do not exercise those boundaries.
3. **Separate utility coverage from workflow enforcement for F-15 and the Phase 5 row.** `evaluation.lock_frozen_set`, `load_frozen` and `check_isolation` exist, but `cli.shadow` neither accepts a split/lock nor invokes those guards. The planned frozen comparison is still blocked on human labels and a detector.
4. **Correct the B-1 dependency.** A shadow run on *real images* can already replay a real primary detector's explicit prediction JSONL through `PrecomputedDetector`; it does not strictly require T-1's crop-capable adapter. T-1 is needed for live primary inference and real SAHI, and the current fixture alone cannot establish real-image performance. Split B-1 into “run real-image shadow from saved predictions” (ready once predictions and labels are supplied) and “live primary/SAHI shadow” (blocked by T-1).
5. **Keep D-5 and B-2/B-3 as measurement gates.** The CLIP prompt-softmax score is raw and prompt-set dependent. The scale guard catches live `OpenClipRelevance`, but replayed JSONL scores have no scale metadata check. Neither the example relevance bands nor any `absent` VLM verdict should be promoted to an automatic negative/rejection decision from the fixture tests.

**Suggested order:** fix R3-1 and R3-2 with regression tests; define the recovery guarantee and address R3-3; fix R3-4 and R3-5; then run a real-image shadow with saved predictions and build the human-labelled split before fitting thresholds or considering Phase 6. Preserve the board's current “routing disabled by default” setting during this work.
