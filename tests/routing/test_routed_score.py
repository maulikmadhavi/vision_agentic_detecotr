"""Stage (2): mAP on the routed output (routed-score-v1), next to primary AP (synthetic fixtures only).

A routed candidate's score is the highest raw_confidence among its own source detections; a VLM
verdict never changes it (it can only remove a candidate by rejecting it); a candidate without any
scored source is left out of AP and counted as unscored_routed.
"""

from __future__ import annotations

import pytest

from routing.metrics import Prediction, evaluate, image_results
from routing.schemas import Detection, FinalCandidate

from .helpers import CLASS_A, FakeVLM, config, task
from .test_metrics import _set
from .test_review_loop import SMALL_A, _run

CLASSES = task().class_ids


def _det(i, box, conf, source="primary"):
    return Detection(id=f"det-{i}", class_id=CLASS_A, box_xyxy=box, raw_confidence=conf, source=source,
                     model_version="scripted@1", slice_origin=(0, 0) if source == "sahi" else None)


def _cand(i, box, status, sources):
    return FinalCandidate(id=f"cand-{i}", class_id=CLASS_A, box_xyxy=box, status=status, source_ids=sources,
                          canonical_source_id=sources[0])


B1, B2, B3, B4 = (10, 10, 50, 50), (100, 100, 140, 140), (200, 10, 240, 50), (10, 150, 50, 190)


def _crafted(tmp_path):
    """Four candidates: accepted (canonical source 0.4 from slicing, other source 0.9 from the secondary), needs_review (the
    slicing detection, 0.4), candidate (unscored), rejected (0.95)."""
    rec = _run(tmp_path, "r")
    sliced, second = _det(2, B2, 0.4, source="sahi"), _det(1, B1, 0.9, source="secondary")
    dets = (_det(3, B3, None), _det(4, B4, 0.95))
    cands = (_cand(1, B1, "accepted", ("det-2", "det-1")), _cand(2, B2, "needs_review", ("det-2",)),
             _cand(3, B3, "candidate", ("det-3",)), _cand(4, B4, "rejected", ("det-4",)))
    return rec.model_copy(update={"primary_candidates": dets, "secondary_candidates": (second,),
                                  "sahi_candidates": (sliced,), "final_candidates": cands})


def test_routed_predictions_take_the_highest_source_confidence_and_drop_rejected_candidates(tmp_path):
    rec = _crafted(tmp_path)
    routed = image_results([rec])[rec.image.id].routed
    assert routed == ((Prediction(CLASS_A, B1, 0.9), "accepted"), (Prediction(CLASS_A, B2, 0.4), "needs_review"),
                      (Prediction(CLASS_A, B3, None), "candidate"))


def test_routed_ap_is_reported_with_and_without_unverified_boxes(tmp_path):
    rec = _crafted(tmp_path)
    gt = _set([rec.image.id], [(rec.image.id, 1, [100, 100, 40, 40], {})])  # the needs_review box is the real one
    out = evaluate(gt, CLASSES, image_results([rec]))["routed"]
    assert out["rule"].startswith("routed-score-v1")
    every, accepted = out["all_unrejected"], out["accepted_only"]
    assert every["statuses"] == ["accepted", "candidate", "needs_review"] and accepted["statuses"] == ["accepted"]
    # all_unrejected: scored boxes 0.9 (FP) then 0.4 (TP) -> AP50 = 0.5; the unscored candidate is counted, not ranked
    assert (every["scored"], every["unscored_routed"]) == (2, 1) and every["custom_map50"] == pytest.approx(0.5)
    # accepted_only: only the 0.9 false positive -> AP 0: the gain above comes from an unverified box
    assert (accepted["scored"], accepted["unscored_routed"]) == (1, 0) and accepted["custom_map50"] == 0.0
    assert every["coco"]["map50"] == pytest.approx(every["custom_map50"], abs=1e-12)
    assert accepted["coco"]["map50"] == pytest.approx(0.0, abs=1e-12)


def test_a_vlm_verdict_never_changes_the_routed_score(tmp_path):
    vlm_cfg = config(vlm={"enabled": True})
    rec = _run(tmp_path, "v", cfg=vlm_cfg, vlm=FakeVLM(default="present"))
    (cand,) = [c for c in rec.final_candidates if c.class_id == CLASS_A]
    assert cand.verification is not None and cand.verification.verdict == "present"
    ((pred, _),) = image_results([rec])[rec.image.id].routed
    assert pred.score == 0.97 and pred.box_xyxy == SMALL_A  # the detector's confidence, not a VLM-derived one


def test_a_vlm_rejection_removes_the_candidate_from_the_routed_output(tmp_path):
    vlm_cfg = config(vlm={"enabled": True, "absent_rejects": True})
    rec = _run(tmp_path, "x", cfg=vlm_cfg, vlm=FakeVLM(default="absent"))
    assert [c.status for c in rec.final_candidates if c.class_id == CLASS_A] == ["rejected"]
    assert image_results([rec])[rec.image.id].routed == ()


def test_primary_ap_stays_on_primary_candidates(tmp_path):
    rec = _crafted(tmp_path)
    gt = _set([rec.image.id], [(rec.image.id, 1, [100, 100, 40, 40], {})])
    det = evaluate(gt, CLASSES, image_results([rec]))["detector"]
    assert det["source"] == "primary_candidates" and det["custom_map50_95"] is None  # det-3 has no confidence
