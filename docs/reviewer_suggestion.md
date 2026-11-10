# Reviewer suggestions — open work only

Updated 2026-11-10; reviewed through `4896bfd`. Completed S4, shared accuracy/cost cohort, evidence-status guard, official COCO evaluation, routed scoring and immutable metrics output have been removed from the action list. Requirements/research remain in [specification.md](specification.md); status belongs in [project_board.md](project_board.md).

## P2 — old split files crash the new metrics writer

**Location:** `src/routing/cli.py`, `_run_evaluate`, construction of `series.images_sha256`.

**Reproduced:** use a valid pre-content-hash frozen lock together with a genuine `split-v1` manifest, whose image entries contain no SHA-256/dimensions. Evaluation passes the legacy verification branch, then raises **`KeyError: 'sha256'`** when constructing the run-metrics file. It should report unverified image content. The existing legacy regression downgrades only the lock and leaves a v2 manifest with hashes, so it misses this case.

**Fix:** derive the series image digest only when the required image content exists and its verification status is explicit; otherwise write `null` and retain the unverified status. Never fabricate hashes or upgrade old files. Add a regression with both a real v1 manifest and old lock, plus the existing v2/old-lock combination. Both must produce a descriptive unverified report without modifying inputs; comparison must reject them as content-verified series.

## Before implementing run comparison

| Gap / implication | Suggested action and acceptance evidence |
|---|---|
| **Stored bootstrap outcomes describe the custom evaluator, not official COCO.** `per_image_outcomes` uses `match_image`; `map_from_outcomes` explicitly reconstructs custom AP. Crowd matching and detection caps differ from the official headline metrics. | Do not attach confidence intervals from these outcomes to COCO mAP50–95/mAP50. Persist sufficient official COCO match outcomes/settings, or retain inputs and rerun the official evaluator for resampled groups. Test reconstruction against official AP with crowd boxes and more than 100 detections. Custom intervals may remain separately labeled. |
| **Comparison compatibility is not yet enforced.** Parent validation checks integrity, not whether datasets/evaluators match. `series.images_sha256` is null without a split; a series key alone does not prove comparability. | In the comparison command require matching verified image/label/cohort identities, task, evaluator settings/version and prediction source/scoring rule. Compare matching working-IoU thresholds for diagnostic metrics. Reject incompatible mixed-run arms or explicitly separate them. Parent linkage alone must never authorize comparison. |
| **Development iteration needs explicit cohort support.** Supplying a split currently selects frozen images; omitting it loses verified content/group provenance. | Add explicit development/frozen cohort selection with content checks and group IDs. Fit on calibration, iterate on development and reserve frozen evaluation for predeclared promotion checks. Verify that development selection excludes frozen images and preserves group-bootstrap metadata. |
| **Run metadata can be misleading.** Evaluation takes `seed` from the supplied config without checking that config against the trace; task compatibility is also not enforced. Mixed runs are marked but can still receive an evidence label. | Validate task and originating config with supported shadow overrides, or derive run metadata from an authoritative recorded config. Reject incompatible traces for promotion/comparison; descriptive mixed-run reports must state they are ineligible. Test a changed evaluation seed/config and task with overlapping class names. |

Use official **mAP50–95 as primary** and **mAP50 as companion**, separately for primary and the selected routed variant. Keep the new `routed-score-v1` rule explicit: maximum source confidence is a defined ranking rule, not calibrated comparability between different detector models. Report unscored exclusions and accepted-annotation precision/recall alongside AP.

For each compatible development iteration, show deltas against the previous and best eligible run, with per-class/size recall, FP/image, review workload and cost guardrails. Predeclare a meaningful improvement threshold and patience window; a plateau within paired source-group uncertainty is a stopping rule, not proof of convergence. Do not average per-image AP to estimate dataset AP. No dashboard or tracking server is required initially.

## Other open boundaries

- **Training copies of frozen images:** annotation exports still need content hashes and training-side comparison against frozen hashes. Reject the same bytes under another ID/group and require verifiable split provenance; tags alone do not cover copied/renamed images.
- **Real-data evidence:** labels, transfer permission, difficulty validation, VLM agreement, promotion targets and trainer integration remain dependencies. Use existing board cards rather than duplicating their plans here.

**Verification:** full CPU/mock suite **548 passed, 7 live tests deselected, one Starlette warning**, 19.20 seconds (`python -B -m pytest -q -p no:cacheprovider` in Pixi, with local socket access for FastAPI). The genuine-v1 compatibility failure was reproduced separately using synthetic fixtures. Only this document was edited; no source changes, live calls or commits.
