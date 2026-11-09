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
