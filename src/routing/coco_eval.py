"""The official COCO bbox evaluator (pycocotools ``COCOeval``) on our ground truth and predictions.

Stock settings: IoU 0.50:0.05:0.95, 101 recall points, at most 100 detections per image (the
highest scored), area range "all". An ``ignore`` box becomes a COCO crowd box (``iscrowd=1``):
COCO matches crowd boxes by intersection over the detection's area, which differs from the
custom evaluator's ordinary IoU (see ``metrics.average_precision``). pycocotools is imported
only when this runs, so importing ``routing`` never needs it.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
from collections.abc import Mapping, Sequence

import numpy as np

from .metrics import GroundTruthImage, Prediction

SETTINGS = {"iou_type": "bbox", "iou_thresholds": "0.50:0.05:0.95", "recall_points": 101,
            "max_detections": 100, "area_range": "all", "ignore_boxes_as": "iscrowd"}


def _header() -> dict:
    return {"evaluator": "pycocotools", "version": importlib.metadata.version("pycocotools"), "settings": SETTINGS}


def _mean(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def coco_map(predictions: Mapping[str, Sequence[Prediction]], ground_truth: Mapping[str, GroundTruthImage],
             class_ids: Sequence[str]) -> dict:
    """mAP50-95, mAP50 and per-class AP from COCOeval; classes without ground truth are listed, not averaged."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_ids = sorted(ground_truth)  # COCO numbers images in this order, as the custom evaluator pools them
    number = {image_id: k + 1 for k, image_id in enumerate(image_ids)}
    category = {c: k + 1 for k, c in enumerate(class_ids)}
    annotations = []
    for image_id in image_ids:
        for box in ground_truth[image_id].boxes:
            x1, y1, x2, y2 = box.box_xyxy
            annotations.append({"id": len(annotations) + 1, "image_id": number[image_id],
                                "category_id": category[box.class_id], "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1), "iscrowd": int(box.ignore)})
    with_gt = {box.class_id for gt in ground_truth.values() for box in gt.boxes if not box.ignore}
    without = [c for c in class_ids if c not in with_gt]
    detections = [{"image_id": number[image_id], "category_id": category[p.class_id],
                   "bbox": [p.box_xyxy[0], p.box_xyxy[1], p.box_xyxy[2] - p.box_xyxy[0], p.box_xyxy[3] - p.box_xyxy[1]],
                   "score": p.score} for image_id in image_ids for p in predictions.get(image_id, ())]
    if not detections:  # loadRes cannot take an empty list; with nothing detected every AP is 0
        per_class = {c: {"ap50_95": 0.0, "ap50": 0.0} for c in class_ids if c in with_gt}
    else:
        gt = COCO()
        gt.dataset = {"images": [{"id": number[i], "width": ground_truth[i].width, "height": ground_truth[i].height}
                                 for i in image_ids],
                      "categories": [{"id": category[c], "name": c} for c in class_ids], "annotations": annotations}
        with contextlib.redirect_stdout(io.StringIO()):  # COCO prints progress
            gt.createIndex()
            run = COCOeval(gt, gt.loadRes(detections), "bbox")
            run.evaluate()
            run.accumulate()
        precision = run.eval["precision"]  # [iou, recall, class, area, max_detections]; -1 = no ground truth
        area, top = run.params.areaRngLbl.index("all"), run.params.maxDets.index(100)
        per_class = {}
        for c in class_ids:
            if c in with_gt:
                k = run.params.catIds.index(category[c])
                per_class[c] = {"ap50_95": _mean(precision[:, :, k, area, top]), "ap50": _mean(precision[0, :, k, area, top])}
    return {**_header(),
            "map50_95": float(np.mean([v["ap50_95"] for v in per_class.values()])) if per_class else None,
            "map50": float(np.mean([v["ap50"] for v in per_class.values()])) if per_class else None,
            "ap_per_class": per_class, "classes_without_ground_truth": without}


def coco_unavailable(reason: str) -> dict:
    return {**_header(), "map50_95": None, "map50": None, "map_unavailable": reason}
