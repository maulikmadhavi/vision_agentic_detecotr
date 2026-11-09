# Reviewer suggestions — open work only

Updated 2026-11-10. Resolved findings and superseded architecture proposals have been removed. Requirements and research mapping remain in [specification.md](specification.md); implementation status belongs in [project_board.md](project_board.md). This file records actionable gaps without repeating those documents.

## S4: make observations and scoring provenance trustworthy

S4 should remain a small correctness milestone before adding run-comparison features.

| Open finding | Implication | Suggested implementation and acceptance test |
|---|---|---|
| **R5-5: repeated images** — `calibration.fit_relevance`, `fit_relevance_rows`, `metrics.image_results` | Repeating one positive and one negative ten times can satisfy the 10/10 minimum. Evaluation silently keeps the last result for an image while cost reporting counts all rows. | Prefer rejecting duplicate image IDs within one run/arm, including identical repeated rows, with the offending IDs. Validate before fitting, scoring and writing outputs; do not reject the same image appearing in separate comparison runs. Count unique images for fit minima. Test duplicate rows, conflicting results, shuffled input and the two-image minimum bypass. |
| **R5-6: fit-time score scale** — `calibration.fit_relevance` | The artifact copies the supplied scale without proving the originating trace used it. Loader compatibility checks then trust that incorrect declaration. | Check the originating run config, explicitly accounting for supported shadow/VLM overrides, or persist a compact scoring-provenance record. Bind scale, scorer/build, prompts and preprocessing. Test changed scale, mixed incompatible runs and valid original-config fitting. Do not require irrelevant detector settings to match if using scoring-specific provenance. |

**S4 completion:** focused regressions and the full CPU/mock suite pass; valid fit output remains deterministic; no silent row selection; existing unrelated run identities remain stable. Keep artifact loading checks and frozen-content checks intact. Add no new service or generic validation framework.

## Metrics: standard mAP and comparable iterations

**Recommendation:** use **mAP50–95 as the primary detection metric** and **mAP50 as a companion**. Use the official COCO bbox evaluator with pinned version/settings for standard results: IoU 0.50–0.95 in steps of 0.05, 101 recall points, all-area summary and the standard maximum of 100 detections per image. Preserve standard crowd/ignore handling. [Official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

**Current implementation gap:** `metrics.average_precision` explicitly differs from COCO on the detection cap and crowd matching. A previous fixture parity check is insufficient to claim full standard parity. Prefer a small evaluation-only `pycocotools` dependency; keep it outside model-serving environments. Retain existing diagnostic metrics where useful. If keeping the custom evaluator, label its results as custom and test parity for crowd/ignore boxes, more than 100 detections, ties, empty images and classes without ground truth before using standard metric names.

**Measure the intended output:** `metrics.image_results` currently computes AP from `primary_candidates` only. Routing/VLM improvements may therefore leave both AP values unchanged. Report primary AP separately from a clearly defined scored routed-detection output. Preserve detector-authored scores and document a fixed scoring rule for fusion and multiple detectors; do not invent scores from VLM verdicts or assign every accepted box confidence 1. Score-free accepted annotations should retain precision/recall, false positives and missed-instance counts until a justified ranking exists. mAP alone does not assess relevance-probability calibration or annotation completeness.

### Comparing runs and deciding when to stop

1. **Iterate on development data.** Fit on calibration data; compare successive changes on the same development images and labels. Reserve frozen evaluation for predeclared milestone/promotion checks. Repeated tuning against frozen scores turns that set into development data.
2. **Record one immutable result per run and arm.** Store run/parent IDs, image and label hashes, split identity, model/artifact/config IDs, prediction source, evaluator version/settings, seed, image/error counts and metrics. Changing dataset, annotation policy or evaluator starts a new comparison series; re-evaluate prior predictions when possible.
3. **Compare against both the previous run and the best eligible run.** Show mAP50–95, mAP50 and absolute percentage-point deltas on a consistent scale. Preserve primary-only and always-expensive baselines. Keep per-class/size recall, FP/image, review rate and latency/calls as guardrails so an aggregate improvement cannot hide a costly or important-class regression.
4. **Treat a plateau as an operational stopping rule, not proof of convergence.** Predeclare a minimum meaningful AP gain, patience window and regression tolerances on development data. Use paired source-group bootstrap intervals when the sample supports them. If improvements are within uncertainty for the patience window, stop tuning and assess the selected candidate on frozen data. Choose numerical thresholds before inspecting frozen results.

**Implementation order:** finish S4 → standard evaluator and explicit prediction-source selection → versioned run-metrics JSON plus a small comparison command/table. A dashboard is optional; no experiment-tracking server is needed initially.

## Remaining correctness and release gates

| Priority / item | Action |
|---|---|
| **R5-7: accuracy and cost cohorts differ** | In `_run_evaluate`, select one evaluation cohort for both metrics and `route_report`. Adding a development record must not affect frozen costs. Keep missing labelled results visible as operational failures. |
| **Training can miss renamed frozen copies** | Carry image hashes in versioned annotation exports and compare them with frozen hashes in `load_annotations`. Reject the same bytes under a different ID/group. Require verifiable split provenance for training; do not silently trust untagged legacy exports. |
| **Evaluation provenance** | Reject task-incompatible or incompatible mixed-run traces. Separate descriptive metrics from promotion eligibility; human provenance alone must not override an unverified split/content status. |
| **Real-data evidence** | Human-reviewed labels, image-transfer permission, difficulty validation, VLM human-label agreement, numeric promotion gates and trainer integration remain dependencies. Refer to the existing board cards and specification research references instead of duplicating their work plans here. |

**Verification context:** source paths above were rechecked for this documentation update. Latest independently completed suite from the preceding review: **487 passed, 7 live tests deselected, one Starlette warning**. Tests were not rerun for this documentation-only cleanup. Only this file was changed.
