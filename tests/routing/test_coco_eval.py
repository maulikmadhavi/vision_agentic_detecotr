"""Official COCO evaluator (pycocotools) next to the custom one: parity where they agree, documented
differences where they legitimately differ (synthetic fixtures only)."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys

import pytest

from routing.coco_eval import coco_map
from routing.metrics import ImageResult, evaluate, mean_average_precision

from .helpers import CLASS_A, CLASS_B, task
from .test_metrics import GT_A, PRED_A, _p, _report_fixture, _set

CLASSES = task().class_ids


def _both(predictions, ground_truth):
    return mean_average_precision(predictions, ground_truth, CLASSES), coco_map(predictions, ground_truth, CLASSES)


def _assert_parity(custom, coco):
    assert coco["map50_95"] == pytest.approx(custom["map50_95"], abs=1e-12)
    assert coco["map50"] == pytest.approx(custom["map50"], abs=1e-12)
    assert coco["classes_without_ground_truth"] == custom["classes_without_ground_truth"]
    for c, ap in custom["per_class"].items():
        assert coco["ap_per_class"][c] == {k: pytest.approx(v, abs=1e-12) for k, v in ap.items()}


# --- parity ---------------------------------------------------------------------------------------


def test_parity_on_the_hand_computed_fixture():
    _assert_parity(*_both(PRED_A, _set(["i1", "i2"], GT_A)))


def test_parity_with_a_failed_image_negatives_and_a_localisation_miss():
    gt, results = _report_fixture()
    _assert_parity(*_both({i: r.detections for i, r in results.items() if i in gt}, gt))


def test_parity_on_empty_images_and_a_false_positive_on_a_negative_image():
    gt = _set(["pos", "empty", "neg"], [("pos", 1, [10, 10, 40, 40], {})])
    preds = {"pos": [_p(CLASS_A, (10, 10, 50, 50), 0.9)], "neg": [_p(CLASS_A, (0, 0, 30, 30), 0.95)]}
    custom, coco = _both(preds, gt)
    _assert_parity(custom, coco)
    assert coco["map50"] == pytest.approx(0.5)  # the higher-scored negative-image FP comes first


def test_parity_for_a_class_without_ground_truth_even_with_its_false_positives():
    gt = _set(["i1"], [("i1", 1, [10, 10, 40, 40], {})])
    preds = {"i1": [_p(CLASS_A, (10, 10, 50, 50), 0.9), _p(CLASS_B, (100, 100, 140, 140), 0.99)]}
    custom, coco = _both(preds, gt)
    _assert_parity(custom, coco)
    assert coco["classes_without_ground_truth"] == [CLASS_B] and CLASS_B not in coco["ap_per_class"]


@pytest.mark.parametrize("order", ["fp_first", "tp_first"])
def test_ties_resolve_by_input_order_in_both(order):
    gt = _set(["i1", "i2"], [("i1", 1, [10, 10, 40, 40], {})])
    tp, fp = _p(CLASS_A, (10, 10, 50, 50), 0.5), _p(CLASS_A, (100, 100, 140, 140), 0.5)
    preds = {"i1": [fp, tp] if order == "fp_first" else [tp, fp], "i2": [_p(CLASS_A, (5, 5, 25, 25), 0.5)]}
    custom, coco = _both(preds, gt)
    _assert_parity(custom, coco)


def test_parity_when_nothing_was_detected():
    custom, coco = _both({}, _set(["i1"], [("i1", 1, [10, 10, 40, 40], {})]))
    _assert_parity(custom, coco)
    assert coco["map50_95"] == 0.0


def test_parity_with_many_detections_under_the_100_cap():
    gt = _set(["i1"], [("i1", 1, [10, 10, 20, 20], {})])
    fps = [_p(CLASS_A, (100 + k, 100, 140 + k, 140), 0.9) for k in range(50)]
    custom, coco = _both({"i1": fps + [_p(CLASS_A, (10, 10, 30, 30), 0.01)]}, gt)
    _assert_parity(custom, coco)
    assert coco["map50"] == pytest.approx(1 / 51)  # the true positive is the 51st detection: kept by both


def test_parity_on_a_detection_matching_an_ignore_box_by_ordinary_iou():
    gt = _set(["i1"], [("i1", 1, [10, 10, 40, 40], {}), ("i1", 1, [100, 100, 40, 40], {"ignore": True})])
    preds = {"i1": [_p(CLASS_A, (10, 10, 50, 50), 0.8), _p(CLASS_A, (100, 100, 140, 140), 0.9)]}
    _assert_parity(*_both(preds, gt))  # neutral in both: it overlaps the ignore box fully


# --- legitimate differences, documented --------------------------------------------------------------


def test_difference_a_small_detection_inside_a_large_ignore_region():
    """COCO matches crowd (ignore) boxes by intersection over the DETECTION's area, so a small box
    inside a large ignore region is neutral there; the custom evaluator uses ordinary IoU and counts
    it as a false positive."""
    gt = _set(["i1"], [("i1", 1, [10, 10, 40, 40], {}), ("i1", 1, [80, 80, 100, 100], {"ignore": True})])
    preds = {"i1": [_p(CLASS_A, (100, 100, 120, 120), 0.9), _p(CLASS_A, (10, 10, 50, 50), 0.8)]}
    custom, coco = _both(preds, gt)
    assert coco["map50"] == pytest.approx(1.0) and custom["map50"] == pytest.approx(0.5)


def test_difference_more_than_100_detections_in_one_image():
    """COCO keeps the 100 highest-scored detections per image; the custom evaluator keeps all, so a
    true positive ranked 151st still counts there."""
    gt = _set(["i1"], [("i1", 1, [10, 10, 20, 20], {})])
    fps = [_p(CLASS_A, (100 + k % 50, 100 + k // 50, 140 + k % 50, 140 + k // 50), 0.9) for k in range(150)]
    preds = {"i1": fps + [_p(CLASS_A, (10, 10, 30, 30), 0.01)]}
    custom, coco = _both(preds, gt)
    assert coco["map50"] == 0.0 and custom["map50"] == pytest.approx(1 / 151)


# --- in the metrics output ---------------------------------------------------------------------------


def test_evaluate_reports_the_custom_block_and_a_coco_block_with_the_pinned_version():
    gt, results = _report_fixture()
    det = evaluate(gt, CLASSES, results)["detector"]
    assert det["evaluator"] == "custom-vad-1" and "map50_95" not in det and "map50" not in det
    assert det["custom_map50"] == pytest.approx(67 / 101 / 2) and set(det["custom_ap_per_class"]) == {CLASS_A, CLASS_B}
    coco = det["coco"]
    assert coco["evaluator"] == "pycocotools" and coco["version"] == importlib.metadata.version("pycocotools")
    assert coco["settings"] == {"iou_type": "bbox", "iou_thresholds": "0.50:0.05:0.95", "recall_points": 101,
                                "max_detections": 100, "area_range": "all", "ignore_boxes_as": "iscrowd"}
    assert coco["map50"] == pytest.approx(det["custom_map50"], abs=1e-12)


def test_the_coco_block_is_unavailable_with_unscored_detections_too():
    gt, results = _report_fixture()
    results["p2"] = ImageResult(detections=(_p(CLASS_A, (0, 0, 120, 110), None),), accepted=(), flagged_for_review=True)
    det = evaluate(gt, CLASSES, results)["detector"]
    assert det["custom_map50_95"] is None and det["coco"]["map50_95"] is None
    assert "confidence" in det["coco"]["map_unavailable"]


def test_importing_routing_does_not_import_pycocotools():
    code = "import sys, routing.cli, routing.metrics, routing.coco_eval; print('pycocotools' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"
