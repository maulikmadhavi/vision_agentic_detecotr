# Reviewer suggestions — open work only

Updated 2026-11-10. Resolved findings and superseded architecture proposals have been removed. Requirements and research mapping remain in [specification.md](specification.md); implementation status belongs in [project_board.md](project_board.md). This file records actionable gaps without repeating those documents.

## S4 review: one output-boundary fix remains

Reviewed the uncommitted implementation above `732dad6`. **R5-6 is resolved:** fitting validates the originating config hash through `shadow_run_config`, including routing-on and VLM-on CLI overrides; incompatible mixed configs are refused. Matching the entire config is conservative but acceptable for this milestone. **R5-5 is resolved at fitting and evaluation inputs**, including identical/conflicting duplicates and the sample-minimum bypass. Those completed instructions have been removed.

### P2 — shadow still writes duplicate images (remaining R5-5)

**Location:** `src/routing/cli.py`, `_write_shadow_outputs`; upstream `RoutingPipeline.run` accepts repeated manifest entries.

**Reproduction / implication:** passing `[record, record]` to `_write_shadow_outputs` succeeds and produces two annotation-image entries, two trace rows and `report.images = 2`. Repeating an input image can therefore generate a successful shadow output that the new evaluator refuses. Training/export consumers can still receive duplicates. The new fitting/evaluation checks prevent silent consumption there, but do not satisfy S4's requirement to validate before writing run outputs.

**Suggested fix:** reject repeated manifest image IDs before processing where practical, and validate the collected record IDs by image at the output boundary **before any annotations, trace or report is written**. Apply a clear rule to conflicting files sharing an explicit ID. Avoid calling metric computation merely to validate uniqueness; a small shared ID validator is sufficient. Separate runs may contain the same images.

**Acceptance tests:** duplicate manifest entry and conflicting files with one ID fail clearly; no partial output files are created or existing outputs overwritten; a normal run remains unchanged. Preserve the existing fitting/evaluation duplicate tests and config-override tests. Once this passes, S4 can close; R5-7 remains a separate milestone.

## Metrics: standard mAP and comparable iterations

**Recommendation:** use **mAP50–95 as the primary detection metric** and **mAP50 as a companion**. Use the official COCO bbox evaluator with pinned version/settings for standard results: IoU 0.50–0.95 in steps of 0.05, 101 recall points, all-area summary and the standard maximum of 100 detections per image. Preserve standard crowd/ignore handling. [Official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

**Current implementation gap:** `metrics.average_precision` explicitly differs from COCO on the detection cap and crowd matching. A previous fixture parity check is insufficient to claim full standard parity. Prefer a small evaluation-only `pycocotools` dependency; keep it outside model-serving environments. Retain existing diagnostic metrics where useful. If keeping the custom evaluator, label its results as custom and test parity for crowd/ignore boxes, more than 100 detections, ties, empty images and classes without ground truth before using standard metric names.

**Measure the intended output:** `metrics.image_results` currently computes AP from `primary_candidates` only. Routing/VLM improvements may therefore leave both AP values unchanged. Report primary AP separately from a clearly defined scored routed-detection output. Preserve detector-authored scores and document a fixed scoring rule for fusion and multiple detectors; do not invent scores from VLM verdicts or assign every accepted box confidence 1. Score-free accepted annotations should retain precision/recall, false positives and missed-instance counts until a justified ranking exists. mAP alone does not assess relevance-probability calibration or annotation completeness.

### Comparing runs and deciding when to stop

1. **Iterate on development data.** Fit on calibration data; compare successive changes on the same development images and labels. Reserve frozen evaluation for predeclared milestone/promotion checks. Repeated tuning against frozen scores turns that set into development data.
2. **Record one immutable result per run and arm.** Store run/parent IDs, image and label hashes, split identity, model/artifact/config IDs, prediction source, evaluator version/settings, seed, image/error counts and metrics. Changing dataset, annotation policy or evaluator starts a new comparison series; re-evaluate prior predictions when possible.
3. **Compare against both the previous run and the best eligible run.** Show mAP50–95, mAP50 and absolute percentage-point deltas on a consistent scale. Preserve primary-only and always-expensive baselines. Keep per-class/size recall, FP/image, review rate and latency/calls as guardrails so an aggregate improvement cannot hide a costly or important-class regression.
4. **Treat a plateau as an operational stopping rule, not proof of convergence.** Predeclare a minimum meaningful AP gain, patience window and regression tolerances on development data. Use paired source-group bootstrap intervals when the sample supports them. If improvements are within uncertainty for the patience window, stop tuning and assess the selected candidate on frozen data. Choose numerical thresholds before inspecting frozen results.

**Implementation order:** finish the S4 output guard → standard evaluator and explicit prediction-source selection → versioned run-metrics JSON plus a small comparison command/table. A dashboard is optional; no experiment-tracking server is needed initially.

## Remaining correctness and release gates

| Priority / item | Action |
|---|---|
| **R5-7: accuracy and cost cohorts differ** | In `_run_evaluate`, select one evaluation cohort for both metrics and `route_report`. Adding a development record must not affect frozen costs. Keep missing labelled results visible as operational failures. |
| **Training can miss renamed frozen copies** | Carry image hashes in versioned annotation exports and compare them with frozen hashes in `load_annotations`. Reject the same bytes under a different ID/group. Require verifiable split provenance for training; do not silently trust untagged legacy exports. |
| **Evaluation provenance** | Reject task-incompatible or incompatible mixed-run traces. Separate descriptive metrics from promotion eligibility; human provenance alone must not override an unverified split/content status. |
| **Real-data evidence** | Human-reviewed labels, image-transfer permission, difficulty validation, VLM human-label agreement, numeric promotion gates and trainer integration remain dependencies. Refer to the existing board cards and specification research references instead of duplicating their work plans here. |

**Verification context:** S4 full CPU/mock suite: **499 passed, 7 live tests deselected, one Starlette warning**, 16.59 seconds (`python -B -m pytest -q -p no:cacheprovider` in the Pixi environment, local socket access for FastAPI tests). The duplicate-output reproduction above was run separately with a synthetic fixture. Only this review file was edited; no implementation changes or commits were made.
