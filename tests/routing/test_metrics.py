"""A3 stage 1: ground-truth label format, loader and one-to-one matching (synthetic fixtures only)."""

from __future__ import annotations

import copy

import pytest

from routing.metrics import GroundTruthBox, LabelError, Prediction, load_ground_truth, match_image

from .helpers import CLASS_A, CLASS_B, task

CATS = [{"id": 1, "name": CLASS_A}, {"id": 2, "name": CLASS_B}]


def _doc(annotations=(), images=None):
    images = images if images is not None else [
        {"id": 1, "file_name": "a.png", "width": 100, "height": 100, "vad_image_id": "a", "vad_exhaustive": True},
        {"id": 2, "file_name": "neg.png", "width": 100, "height": 100, "vad_image_id": "neg", "vad_exhaustive": True,
         "vad_ambiguous": True, "vad_reviewers": ["rev-1", "rev-2"]},
    ]
    return {"images": images, "categories": [dict(c) for c in CATS], "annotations": list(annotations)}


def _ann(i, image_id, cat, bbox, **extra):
    return {"id": i, "image_id": image_id, "category_id": cat, "bbox": bbox, **extra}


# --- loader -------------------------------------------------------------------------------------


def test_loader_converts_coco_boxes_and_keeps_ignore_and_review_fields():
    gt = load_ground_truth(_doc([_ann(1, 1, 1, [10, 20, 30, 40]), _ann(2, 1, 2, [50, 50, 10, 10], ignore=True)]), task())
    assert gt["a"].boxes == (GroundTruthBox(CLASS_A, (10, 20, 40, 60), False), GroundTruthBox(CLASS_B, (50, 50, 60, 60), True))
    assert gt["neg"].boxes == () and gt["neg"].ambiguous and gt["neg"].reviewers == ("rev-1", "rev-2")
    assert not gt["a"].ambiguous and gt["a"].reviewers == ()


def test_iscrowd_counts_as_ignore():
    gt = load_ground_truth(_doc([_ann(1, 1, 1, [10, 20, 30, 40], iscrowd=1)]), task())
    assert gt["a"].boxes[0].ignore


def _broken(mutate):
    doc = _doc([_ann(1, 1, 1, [10, 20, 30, 40])])
    mutate(doc)
    return doc


@pytest.mark.parametrize("name,mutate,match", [
    ("not exhaustive", lambda d: d["images"][0].update(vad_exhaustive=False), "exhaustive"),
    ("exhaustive missing", lambda d: d["images"][0].pop("vad_exhaustive"), "exhaustive"),
    ("no vad_image_id", lambda d: d["images"][0].pop("vad_image_id"), "vad_image_id"),
    ("duplicate image id", lambda d: d["images"][1].update(vad_image_id="a"), "duplicate"),
    ("unknown category", lambda d: d["categories"].append({"id": 3, "name": "dog"}), "dog"),
    ("missing task class", lambda d: d["categories"].pop(), "categories"),
    ("annotation for unknown image", lambda d: d["annotations"][0].update(image_id=99), "unknown image"),
    ("annotation with unknown category", lambda d: d["annotations"][0].update(category_id=7), "category"),
    ("zero-width box", lambda d: d["annotations"][0].update(bbox=[10, 20, 0, 40]), "bbox"),
    ("box outside the image", lambda d: d["annotations"][0].update(bbox=[80, 20, 30, 40]), "bbox"),
    # R5-4: numeric ids are lookup keys, so a repeat would merge images or reassign boxes silently
    ("two images share a numeric id", lambda d: d["images"][1].update(id=1), "duplicate image id 1"),
    ("two categories share a numeric id", lambda d: d["categories"][1].update(id=1), "duplicate category id 1"),
    ("one class under two category ids", lambda d: d["categories"].append({"id": 3, "name": CLASS_A}),
     f"category name '{CLASS_A}'.*ids 1 and 3"),
    ("two annotations share an id", lambda d: d["annotations"].append(_ann(1, 2, 2, [5, 5, 5, 5])),
     "duplicate annotation id 1"),
])
def test_loader_rejects_invalid_label_files(name, mutate, match):
    with pytest.raises(LabelError, match=match):
        load_ground_truth(_broken(mutate), task())


def test_loader_does_not_mutate_its_input():
    doc = _doc([_ann(1, 1, 1, [10, 20, 30, 40])])
    before = copy.deepcopy(doc)
    load_ground_truth(doc, task())
    assert doc == before


# --- matching -----------------------------------------------------------------------------------


def _gt(*boxes):
    return load_ground_truth(_doc([_ann(i + 1, 1, cat, bbox, **extra) for i, (cat, bbox, extra) in enumerate(boxes)]),
                             task())["a"]


A_BOX = (1, [10, 10, 40, 40], {})  # xyxy (10, 10, 50, 50)


def test_exact_box_is_a_true_positive():
    m = match_image([Prediction(CLASS_A, (10, 10, 50, 50), 0.9)], _gt(A_BOX))
    assert (m.tp, m.fp, m.fn, m.ignored) == (((0, 0),), (), (), ())


def test_iou_exactly_at_the_threshold_counts():
    # prediction (10,10,50,30) vs gt (10,10,50,50): IoU = 800 / 1600 = 0.5
    m = match_image([Prediction(CLASS_A, (10, 10, 50, 30), 0.9)], _gt(A_BOX), iou_threshold=0.5)
    assert m.tp == ((0, 0),)


def test_low_overlap_is_a_false_positive_and_a_miss():
    m = match_image([Prediction(CLASS_A, (10, 10, 50, 29), 0.9)], _gt(A_BOX))  # IoU 0.475
    assert (m.tp, m.fp, m.fn) == ((), (0,), (0,))


def test_duplicate_detection_is_a_false_positive_and_the_higher_score_matches():
    preds = [Prediction(CLASS_A, (11, 11, 50, 50), 0.6), Prediction(CLASS_A, (10, 10, 50, 50), 0.9)]
    m = match_image(preds, _gt(A_BOX))
    assert (m.tp, m.fp, m.fn) == (((1, 0),), (0,), ())


def test_matching_is_class_aware():
    m = match_image([Prediction(CLASS_B, (10, 10, 50, 50), 0.9)], _gt(A_BOX))
    assert (m.tp, m.fp, m.fn) == ((), (0,), (0,))


def test_detections_on_an_ignore_box_are_neutral_and_it_is_never_missed():
    preds = [Prediction(CLASS_A, (10, 10, 50, 50), 0.9), Prediction(CLASS_A, (12, 12, 50, 50), 0.8)]
    m = match_image(preds, _gt((1, [10, 10, 40, 40], {"ignore": True})))
    assert (m.tp, m.fp, m.fn, m.ignored) == ((), (), (), (0, 1))


def test_a_regular_box_is_preferred_over_an_overlapping_ignore_box():
    gt = _gt((1, [10, 10, 40, 40], {"ignore": True}), (1, [12, 12, 40, 40], {}))
    m = match_image([Prediction(CLASS_A, (10, 10, 50, 50), 0.9)], gt)  # higher IoU with the ignore box
    assert (m.tp, m.ignored, m.fn) == (((0, 1),), (), ())


def test_each_prediction_takes_the_best_unmatched_box():
    gt = _gt((1, [10, 10, 40, 40], {}), (1, [14, 14, 40, 40], {}))
    m = match_image([Prediction(CLASS_A, (14, 14, 54, 54), 0.9), Prediction(CLASS_A, (10, 10, 50, 50), 0.8)], gt)
    assert m.tp == ((0, 1), (1, 0)) and m.fn == ()


def test_unscored_predictions_keep_their_input_order():
    preds = [Prediction(CLASS_A, (11, 11, 50, 50), None), Prediction(CLASS_A, (10, 10, 50, 50), None)]
    m = match_image(preds, _gt(A_BOX))
    assert (m.tp, m.fp) == (((0, 0),), (1,))


def test_scored_predictions_are_matched_before_unscored_ones():
    preds = [Prediction(CLASS_A, (10, 10, 50, 50), None), Prediction(CLASS_A, (11, 11, 50, 50), 0.1)]
    m = match_image(preds, _gt(A_BOX))
    assert (m.tp, m.fp) == (((1, 0),), (0,))


def test_every_detection_on_a_negative_image_is_a_false_positive():
    gt = load_ground_truth(_doc(), task())["neg"]
    m = match_image([Prediction(CLASS_A, (1, 1, 9, 9), 0.9), Prediction(CLASS_B, (20, 20, 30, 30), 0.4)], gt)
    assert (m.tp, m.fp, m.fn) == ((), (0, 1), ())


def test_prediction_of_a_class_outside_the_task_is_rejected():
    with pytest.raises(LabelError, match="dog"):
        match_image([Prediction("dog", (1, 1, 9, 9), 0.9)], _gt(A_BOX), class_ids=task().class_ids)


# --- A3 stage 2: AP and mAP50-95 (hand-computed values) ----------------------------------------

from routing.metrics import IOU_THRESHOLDS, average_precision, mean_average_precision  # noqa: E402

IMG = {"width": 200, "height": 200, "vad_exhaustive": True}


def _set(images, annotations):
    doc = {"images": [{"id": i + 1, "vad_image_id": name, **IMG} for i, name in enumerate(images)],
           "categories": [dict(c) for c in CATS], "annotations": []}
    ids = {name: i + 1 for i, name in enumerate(images)}
    for k, (name, cat, bbox, extra) in enumerate(annotations):
        doc["annotations"].append({"id": k + 1, "image_id": ids[name], "category_id": cat, "bbox": bbox, **extra})
    return load_ground_truth(doc, task())


def _p(cls, xyxy, score):
    return Prediction(cls, xyxy, score)


# Class A: two boxes, detections in score order TP, FP, TP -> AP = (51 + 50 * 2/3) / 101 = 253/303.
GT_A = [("i1", 1, [0, 0, 50, 50], {}), ("i2", 1, [100, 100, 50, 50], {})]
PRED_A = {"i1": [_p(CLASS_A, (0, 0, 50, 50), 0.9), _p(CLASS_A, (150, 0, 190, 40), 0.8)],
          "i2": [_p(CLASS_A, (100, 100, 150, 150), 0.7)]}


def test_ap_tp_fp_tp_over_two_boxes_is_253_over_303():
    gt = _set(["i1", "i2"], GT_A)
    assert average_precision(PRED_A, gt, CLASS_A, 0.5) == pytest.approx(253 / 303, abs=1e-12)


def test_ap_when_the_top_detection_is_wrong_uses_the_precision_envelope():
    # Score order FP, TP, TP over two boxes: precision 0, 1/2, 2/3 -> envelope 2/3 at every recall -> AP = 2/3.
    gt = _set(["i1", "i2"], GT_A)
    preds = {"i1": [_p(CLASS_A, (150, 0, 190, 40), 0.9), _p(CLASS_A, (0, 0, 50, 50), 0.8)],
             "i2": [_p(CLASS_A, (100, 100, 150, 150), 0.7)]}
    assert average_precision(preds, gt, CLASS_A, 0.5) == pytest.approx(2 / 3, abs=1e-12)


def test_ap_with_an_undetected_box_is_56_over_101():
    # A third box nobody detected: recall steps 1/3, 1/3, 2/3 -> (34 * 1 + 33 * 2/3) / 101.
    gt = _set(["i1", "i2", "i3"], GT_A + [("i3", 1, [10, 10, 30, 30], {})])
    assert average_precision(PRED_A, gt, CLASS_A, 0.5) == pytest.approx(56 / 101, abs=1e-12)


def test_ap_ignores_a_detection_on_an_ignore_box():
    gt = _set(["i1", "i2", "i4"], GT_A + [("i4", 1, [20, 20, 40, 40], {"ignore": True})])
    preds = {**PRED_A, "i4": [_p(CLASS_A, (20, 20, 60, 60), 0.99)]}  # top score, neutral
    assert average_precision(preds, gt, CLASS_A, 0.5) == pytest.approx(253 / 303, abs=1e-12)


def test_ap_is_zero_without_detections_and_none_without_ground_truth():
    gt = _set(["i1", "i2"], GT_A)
    assert average_precision({}, gt, CLASS_A, 0.5) == 0.0
    assert average_precision(PRED_A, gt, CLASS_B, 0.5) is None  # no class-B box anywhere


def test_iou_thresholds_are_point_50_to_point_95():
    assert IOU_THRESHOLDS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def test_map_averages_classes_with_ground_truth_over_ten_thresholds():
    # Class B: one box, one detection at IoU 0.72 -> found at 0.50-0.70 (5 thresholds), missed above.
    gt = _set(["i1", "i2", "b"], GT_A + [("b", 2, [0, 0, 100, 100], {})])
    preds = {**PRED_A, "b": [_p(CLASS_B, (0, 0, 100, 72), 0.6)]}
    out = mean_average_precision(preds, gt, task().class_ids)
    assert out["per_class"][CLASS_A] == {"ap50_95": pytest.approx(253 / 303), "ap50": pytest.approx(253 / 303)}
    assert out["per_class"][CLASS_B] == {"ap50_95": pytest.approx(0.5), "ap50": pytest.approx(1.0)}
    assert out["map50_95"] == pytest.approx((253 / 303 + 0.5) / 2)
    assert out["map50"] == pytest.approx((253 / 303 + 1.0) / 2)
    assert out["classes_without_ground_truth"] == []


def test_map_lists_and_skips_a_class_without_ground_truth():
    gt = _set(["i1", "i2"], GT_A)
    out = mean_average_precision({**PRED_A, "i1": PRED_A["i1"] + [_p(CLASS_B, (0, 0, 9, 9), 0.5)]}, gt,
                                 task().class_ids)
    assert out["classes_without_ground_truth"] == [CLASS_B] and CLASS_B not in out["per_class"]
    assert out["map50_95"] == pytest.approx(253 / 303)


def test_map_refuses_unscored_predictions_and_unknown_images():
    gt = _set(["i1", "i2"], GT_A)
    with pytest.raises(LabelError, match="score"):
        average_precision({"i1": [_p(CLASS_A, (0, 0, 50, 50), None)]}, gt, CLASS_A, 0.5)
    with pytest.raises(LabelError, match="score"):
        mean_average_precision({"i1": [_p(CLASS_A, (0, 0, 50, 50), None)]}, gt, task().class_ids)
    with pytest.raises(LabelError, match="not in the ground truth"):
        mean_average_precision({"zz": [_p(CLASS_A, (0, 0, 50, 50), 0.5)]}, gt, task().class_ids)


# --- A3 stage 3: report metrics (hand-computed) and `vad-route evaluate` -------------------------

from routing.metrics import ImageResult, evaluate, evidence_status  # noqa: E402


def _report_fixture():
    gt = _set(["p1", "p2", "neg", "fail"], [
        ("p1", 1, [0, 0, 40, 40], {}),        # A, area 1600: medium
        ("p1", 2, [100, 100, 20, 20], {}),    # B, area 400: small, never detected
        ("p2", 1, [0, 0, 120, 120], {}),      # A, area 14400: large
        ("fail", 1, [50, 50, 40, 40], {}),    # A, medium; the image has no result (operational failure)
    ])
    results = {
        "p1": ImageResult(detections=(_p(CLASS_A, (0, 0, 40, 40), 0.9), _p(CLASS_A, (150, 150, 190, 190), 0.4)),
                          accepted=(_p(CLASS_A, (0, 0, 40, 40), None),), flagged_for_review=False),
        "p2": ImageResult(detections=(_p(CLASS_A, (0, 0, 120, 110), 0.8),), accepted=(), flagged_for_review=True),
        "neg": ImageResult(detections=(_p(CLASS_B, (10, 10, 30, 30), 0.7),),
                           accepted=(_p(CLASS_B, (10, 10, 30, 30), None),), flagged_for_review=False),
        "unl": ImageResult(detections=(), accepted=(), flagged_for_review=False),  # not labelled
    }
    return gt, results


def test_failed_images_count_as_operational_failures_with_all_ground_truth_missed():
    gt, results = _report_fixture()
    out = evaluate(gt, task().class_ids, results)
    assert out["images"] == {"labelled": 4, "with_result": 3, "negative": 1, "operational_failures": ["fail"],
                             "unlabelled_results": 1}
    assert out["accepted_annotations"]["missed_in_failed_images"] == 1


def test_detector_recall_by_class_and_size_and_false_positives():
    gt, results = _report_fixture()
    det = evaluate(gt, task().class_ids, results)["detector"]
    assert det["source"] == "primary_candidates"
    assert det["recall_by_class"] == {CLASS_A: {"found": 2, "boxes": 3, "recall": pytest.approx(2 / 3)},
                                      CLASS_B: {"found": 0, "boxes": 1, "recall": 0.0}}
    assert det["recall_by_size"] == {"small": {"found": 0, "boxes": 1, "recall": 0.0},
                                     "medium": {"found": 1, "boxes": 2, "recall": 0.5},
                                     "large": {"found": 1, "boxes": 1, "recall": 1.0}}
    assert det["fp_per_image"] == 0.5 and det["fp_per_negative_image"] == 1.0


def test_detector_map_includes_the_failed_image_as_missed_ground_truth():
    gt, results = _report_fixture()
    det = evaluate(gt, task().class_ids, results)["detector"]
    # Class A: TP, TP, FP over 3 boxes -> 67/101 at IoU 0.50-0.90; at 0.95 the second TP fails -> 34/101.
    assert det["custom_map50_95"] == pytest.approx((9 * 67 / 101 + 34 / 101) / 10 / 2)
    assert det["custom_map50"] == pytest.approx(67 / 101 / 2)


def test_accepted_annotations_get_precision_recall_misses_and_review_coverage():
    gt, results = _report_fixture()
    acc = evaluate(gt, task().class_ids, results)["accepted_annotations"]
    assert acc == {"true_positives": 1, "false_positives": 1, "precision": 0.5, "recall": 0.25, "missed": 3,
                   "missed_in_failed_images": 1, "missed_in_review": 1, "silent_misses": 1, "review_coverage": 0.5}


def test_detector_map_is_unavailable_when_a_detection_has_no_confidence():
    gt, results = _report_fixture()
    results["p2"] = ImageResult(detections=(_p(CLASS_A, (0, 0, 120, 110), None),), accepted=(), flagged_for_review=True)
    det = evaluate(gt, task().class_ids, results)["detector"]
    assert det["custom_map50_95"] is None and "confidence" in det["map_unavailable"]


HUMAN, VERIFIED = {"vad_label_provenance": "human"}, "verified frozen_eval"
NO_SPLIT = "unverified: no --split-manifest/--frozen-lock; every labelled image is scored, whatever its split"
OLD_LOCK = "unverified: the frozen labels match --frozen-lock, but the lock predates image hashes"


def test_evidence_needs_human_labels_and_a_verified_frozen_split():
    assert evidence_status(HUMAN, VERIFIED) == "evidence: human labels on a verified frozen split"


@pytest.mark.parametrize("labels, split, says, not_says", [
    ({}, VERIFIED, "vad_label_provenance", "frozen split"),
    ({"vad_label_provenance": "synthetic"}, VERIFIED, "vad_label_provenance", "frozen split"),
    (HUMAN, NO_SPLIT, "no verified frozen split", "vad_label_provenance"),
    (HUMAN, OLD_LOCK, "predates image hashes", "vad_label_provenance"),
], ids=["no provenance", "synthetic", "human, no split", "human, old lock"])
def test_not_a_finding_names_the_missing_part(labels, split, says, not_says):
    status = evidence_status(labels, split)
    assert status.startswith("NOT A FINDING") and says in status and not_says not in status


def test_not_a_finding_names_both_parts_when_both_are_missing():
    status = evidence_status({}, NO_SPLIT)
    assert "vad_label_provenance" in status and "no verified frozen split" in status


def _labels_for_fixture(fx_root, provenance=None):
    images = [("easy", 320, 240), ("zero", 320, 240), ("small", 640, 480), ("corrupt", 320, 240)]
    doc = {"images": [{"id": i + 1, "vad_image_id": n, "width": w, "height": h, "vad_exhaustive": True}
                      for i, (n, w, h) in enumerate(images)],
           "categories": [dict(c) for c in CATS],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [60, 60, 100, 100]},
                           {"id": 2, "image_id": 3, "category_id": 2, "bbox": [300, 200, 8, 8]}]}
    if provenance:
        doc["vad_label_provenance"] = provenance
    path = fx_root / "labels.json"
    path.write_text(__import__("json").dumps(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize("provenance,missing", [(None, "vad_label_provenance"), ("human", "no verified frozen split")])
def test_cli_evaluate_writes_metrics_from_a_shadow_trace(tmp_path, capsys, provenance, missing):
    """Without a split the metrics are still written (descriptive), but they are NOT A FINDING."""
    import json

    from routing import cli

    from .helpers import make_fixture
    fx = make_fixture(tmp_path / "fx")
    out = tmp_path / "run"
    assert cli.main(["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
                     "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
                     "--relevance-scores", str(fx["relevance"]), "--out", str(out)]) == 0
    capsys.readouterr()
    metrics = tmp_path / "metrics.json"
    assert cli.main(["evaluate", "--config", str(fx["config"]), "--labels", str(_labels_for_fixture(fx["root"], provenance)),
                     "--trace", str(out / "route_trace.jsonl"), "--out", str(metrics)]) == 0
    doc = json.loads(metrics.read_text(encoding="utf-8"))
    assert doc["evidence_status"].startswith("NOT A FINDING") and missing in doc["evidence_status"]
    assert "NOT A FINDING" in capsys.readouterr().out
    assert doc["images"]["operational_failures"] == ["corrupt"]  # the corrupt image produced an ErrorRecord
    assert doc["detector"]["recall_by_class"][CLASS_A] == {"found": 1, "boxes": 1, "recall": 1.0}
    assert doc["cost"]["images"] == 3 and doc["iou_threshold"] == 0.5


def test_review_coverage_separates_flagged_from_silent_misses():
    gt = _set(["r", "s"], [("r", 1, [0, 0, 40, 40], {}), ("r", 2, [100, 100, 40, 40], {}), ("s", 1, [0, 0, 40, 40], {})])
    results = {"r": ImageResult((), (), flagged_for_review=True), "s": ImageResult((), (), flagged_for_review=False)}
    acc = evaluate(gt, task().class_ids, results)["accepted_annotations"]
    assert (acc["missed_in_review"], acc["silent_misses"], acc["review_coverage"]) == (2, 1, pytest.approx(2 / 3))


def test_size_strata_follow_coco_area_limits():
    from routing.metrics import _size_stratum
    assert [_size_stratum((0, 0, w, w)) for w in (31.9, 32, 95.9, 96)] == ["small", "medium", "medium", "large"]


def test_ignore_boxes_are_neither_recall_boxes_nor_misses():
    gt = _set(["g"], [("g", 1, [0, 0, 40, 40], {}), ("g", 1, [100, 100, 40, 40], {"ignore": True})])
    out = evaluate(gt, task().class_ids, {"g": ImageResult((_p(CLASS_A, (0, 0, 40, 40), 0.9),), (), False)})
    assert out["detector"]["recall_by_class"][CLASS_A] == {"found": 1, "boxes": 1, "recall": 1.0}
    assert out["accepted_annotations"]["missed"] == 1  # only the regular box; nothing was accepted
