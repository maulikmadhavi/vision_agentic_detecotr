# Reviewer suggestions — open work only

Reviewed 2026-11-10 against `be8be70`. Scope: all 34 Python files under `src` inventoried with AST analysis; manual tracing of CLI, routing/recovery, service contracts, sampling, review, calibration, export, evaluation and comparison. Execution flow and implementation boundaries are in [specification.md](specification.md); delivery status belongs on [project_board.md](project_board.md). Previously completed findings have been removed.

## P2 — R7-1: validate comparison limits before applying guardrails

**Location:** [compare_runs / guardrail_failures](../src/routing/compare.py), [CLI comparison input](../src/routing/cli.py).

**Evidence and implication:** guardrail validation checks the version and top-level keys, but not numeric values or nested fields. A YAML `.nan` for `max_fp_per_image_increase` makes a candidate with 12 additional false positives across eight images report `passed: true`: `value > NaN` is false. Separately, `resamples=0` reaches NumPy and raises `IndexError` instead of an actionable input error. Invalid inputs can disable a decision gate or crash comparison.

**Action:** validate finite, nonnegative numeric guardrail limits, exact nested recall keys, a positive integer patience and resample count, and a finite positive meaningful-gain threshold. Reject booleans as numeric settings. Raise `CompareError` before computing results; retain explicit unavailable-value failures.

**Acceptance:** YAML `.nan`/`.inf`, missing/extra nested keys, strings, booleans, negative limits and zero/negative resamples produce clear CLI errors without an output file. Valid limits still detect the false-positive regression.

## P2 — R7-2: bind source groups into comparison compatibility

**Location:** [compare_runs / _interval](../src/routing/compare.py), [evaluation series construction](../src/routing/cli.py).

**Evidence and implication:** with identical series fields, changing the candidate's eight source groups into one group is accepted; the returned interval still reports eight groups. `_interval` samples the baseline partition exclusively. Image/label identity does not establish an identical grouping: a corrected grouping manifest can silently invalidate the claimed paired source-group uncertainty.

**Action:** require identical per-image key sets and image-to-group mappings before resampling; bind a canonical group-map digest into newly emitted series metadata. For existing metrics, compare their stored mappings explicitly. Do not require equality of the entire manifest, which can include unrelated changes. Missing group evidence must remain explicitly unavailable.

**Acceptance:** changed group membership or missing/extra per-image entries are refused with the affected IDs; reordered identical mappings pass and produce the same seeded interval. Test through CLI metrics files with valid content IDs, not only direct function calls.

## P2 — R7-3: exclude unavailable AP when choosing the best run

**Location:** [compare_runs, `against == "best"`](../src/routing/compare.py).

**Evidence and implication:** the selection key uses `map50_95 or -inf`, treating valid AP `0.0` like unavailable AP `None`. A synthetic comparison using actual evaluator outputs selected an earlier unscored-primary run (`None`) over a later scored run (`0.0`). The selected baseline then has no usable AP/interval, despite an eligible measured baseline being available.

**Action:** exclude runs whose selected-source official AP is unavailable, state that reason in eligibility output, and preserve zero as a valid numeric score. Return a clear error if no measured eligible baseline remains.

**Acceptance:** exercise primary outputs with missing confidence, zero AP and positive AP on the same cohort. Zero must beat unavailable; positive must beat zero; all-unavailable inputs must not select a best run.

## Remaining evidence and integration boundaries

- **Metric interpretation:** keep official COCO **mAP50–95 primary** and **mAP50 companion**, with previous/best development-run deltas. The implemented bootstrap and `verdict` describe **custom-vad-1**, not official COCO uncertainty; name that evaluator in the verdict field/output. If promotion needs COCO confidence intervals, resample groups through the official evaluator or persist sufficient compatible outcomes and prove reconstruction parity. Current per-size guardrails describe the primary detector only; add routed size metrics before interpreting them as routed recall protection. A patience plateau is an operational stopping rule, not proof of convergence.
- **Frozen images entering training:** exported annotation images still lack content hashes. Add byte identity and verified split provenance to exports and require training-side rejection of frozen content, including renamed/copied images. Existing ID/group/tag checks cannot enforce that guarantee. No actual trainer integration is supplied.
- **Empirical validation:** behavior tests do not establish detection gains. Human labels, image-transfer permission, target-domain relevance/difficulty/VLM validation, numeric promotion thresholds and controlled annotation-to-training comparisons remain dependencies; use the existing board cards for ownership.

**Verification:** 592 CPU/mock tests passed; seven live tests deselected; one Starlette deprecation warning (22.20 s, Pixi `python -B -m pytest -q -p no:cacheprovider`, local socket access for FastAPI). Separate in-memory synthetic probes reproduced all three findings and the zero-resample crash. These probes exercise current evaluator/comparison functions; CLI regressions above are proposed acceptance tests. No GPU/live-service accuracy claims were checked. This review changes documentation only and makes no commits.
