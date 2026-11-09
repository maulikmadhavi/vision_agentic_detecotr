# Reviewer suggestions — adaptive object detection routing

**Reviewed:** 2026-11-09 · **Status:** phases 0–4 implemented for the greenfield repository; Phase 5 has utilities but no real-data calibration; Phase 6 has not started. Routing remains disabled by default. See [project_board.md](project_board.md) for the plan and project tasks.

**Verification:** the default suite passed **91 tests**; two optional live tests (CLIP weights and LM Studio) were deselected. The implemented paths have mostly been exercised with a tiny fixture and scripted detectors. These results do not establish performance on real images.

## Open finding

[P2] Route a class with no calibrated probability to an explicit reviewer — `src/vad/routing/scoring.py:103`

When the calibrator omits a class, `class_is_high_relevance` returns `False`. For a zero-box image with a VLM enabled, the raised medium band can select `vlm_verify`, yet `_verify` sends **no request** for that class. `completeness.audit` later records `calibrator_missing_class:<id>` and `unresolved`, but the route has no `human_review` action and `suspected_missing_classes` omits the class. I reproduced `actions=('vlm_verify',)`, zero VLM requests and `status='unresolved'`. **Suggested solution:** carry a per-class `unknown` state into routing. Either verify the missing class or add an explicit human-review action and structured missing-class item; never fall back to the raw threshold. Add a regression test with the VLM enabled. The current T-6 tests use no VLM.

## Implemented and reviewed

| Area | Current implementation | Evidence / limit |
|---|---|---|
| Contracts and legacy path | Pydantic v2 records, exported JSON Schemas, stable IDs, validated class/box coordinates, feature flag off by default. | Schema drift and legacy-output tests pass. No existing detector repository or trainer was available to integrate. |
| Routing and detection | Separate relevance, difficulty and completeness; deterministic policy; primary prediction replay; optional secondary and native sliced inference; original-image box coordinates and provenance. | Acceptance tests cover zero boxes, disagreement, replay, coordinates and budgets. Slicing uses scripted crop-capable detectors, not a production adapter. |
| Verification and recovery | LM Studio VLM is verifier-only; uncertain responses go to review. SQLite caches steps, records an attempt before a VLM call, and makes an interrupted call `uncertain` on resume. | At-most-once call behavior trades a potentially lost answer for no repeat call. Live VLM testing was reported previously, not rerun in this review. |
| Export and evaluation utilities | Current-run-only annotation export, per-image error queue, frozen-set lock/isolation helpers, VLM-vs-human agreement scoring. | Frozen-set checks are utilities; the shadow CLI does not enforce split isolation. The export loader substitutes for a nonexistent trainer. |
| Calibrator guards | Incomplete raw or calibrated per-class output cannot produce a low relevance band; a missing calibrated class does not use raw thresholds. | The open finding above concerns the subsequent verification/review handoff. No fitted calibrator or human-labelled dataset exists yet. |

Earlier review findings are fixed and covered by regressions: SAHI parent provenance and per-tile budgets; cached Stage A and explicit prediction rows; current-run export isolation (R3-1); cross-class VLM review (R3-2); at-most-once recovery (R3-3); incomplete low-band guard (R3-4/R4-1); and blocking a bare human `complete` when evidence remains open (R3-5). The implementation and tests are in commits `f2dd581`–`6a31afd` and `tests/routing/test_r3.py`. An explicit per-finding human adjudication record remains to be designed if reviewers must resolve open findings in the application.

## Remaining work and possible solutions

1. **Real detector and real-image shadow:** choose a primary detector, add a crop-capable adapter with a weights digest and measured GPU time, and replay or run predictions on representative images. Add an independent secondary detector only if it provides useful recovery.
2. **Human data and calibration:** label a sample across classes, domains, sizes and negatives; split by source/near-duplicate group. Fit relevance and difficulty calibrators on calibration data, choose thresholds on development data, and record data/model hashes. Decide CLIP score bands from that sample; replayed JSONL scores currently have no scale guard.
3. **Frozen benchmark and promotion gates:** lock a held-out set; set numeric recall, false-negative, precision and cost targets before evaluation. Compare primary-only, always-expensive and adaptive routes on the same images, including low-relevance false-negative audits. Integrate split checks into the run workflow rather than relying only on helper functions.
4. **Trainer and rollout:** connect the annotation export to an actual training entry point, then run a canary by source group with a versioned rollback configuration. Keep automatic discard disabled until the frozen gates pass.
5. **VLM policy:** compare the 4B and 8B models against human labels on real crops before considering `vlm.absent_rejects: true`; keep the safer `false` default in the meantime.

**Suggested order:** fix the open reviewer handoff, collect real predictions and labels, calibrate and benchmark on isolated splits, then consider canary rollout.
