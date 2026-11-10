"""Detection metrics against human ground truth (reviewer work order A3).

Ground truth is COCO JSON, as exported by any labelling tool, plus a few ``vad_*`` fields:

- ``images[]``: ``id``, ``width``, ``height``, ``vad_image_id`` (the manifest image id, required),
  ``vad_exhaustive: true`` (required: every configured class was labelled, so a class with no
  box is absent from the image), optional ``vad_ambiguous`` and ``vad_reviewers``.
- ``categories[]``: ``name`` must be exactly the task's class ids.
- ``annotations[]``: ``bbox`` = ``[x, y, w, h]`` in pixels; ``ignore: true`` (or ``iscrowd``)
  marks an object that counts neither as found nor as missed (e.g. heavily occluded).

Matching is class-aware and one-to-one, highest score first (COCO-style): each prediction
takes the best-overlapping unmatched regular box at or above the IoU threshold; otherwise a
prediction overlapping an ignore box is neutral; otherwise it is a false positive. A second
prediction on an already matched box is a false positive. Unmatched regular boxes are misses.

These are tools only: numbers mean something only on human labels. Real-image evaluation is
blocked until the labelled sample (and permission to use its images) exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .image_inputs import repeated_ids
from .matching import iou
from .schemas import TaskSpec

Box = tuple[float, float, float, float]
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.50:0.05:0.95
_RECALL_POINTS = np.linspace(0.0, 1.0, 101)


class LabelError(ValueError):
    """The ground-truth file (or a prediction) does not meet the label format."""


@dataclass(frozen=True)
class GroundTruthBox:
    class_id: str
    box_xyxy: Box
    ignore: bool


@dataclass(frozen=True)
class GroundTruthImage:
    image_id: str
    width: int
    height: int
    boxes: tuple[GroundTruthBox, ...]
    ambiguous: bool = False
    reviewers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prediction:
    class_id: str
    box_xyxy: Box
    score: float | None  # None for accepted (score-free) annotations


@dataclass(frozen=True)
class ImageMatch:
    """Indices into the prediction list and the ground-truth boxes; each tuple is sorted."""

    tp: tuple[tuple[int, int], ...]  # (prediction index, ground-truth box index)
    fp: tuple[int, ...]
    fn: tuple[int, ...]
    ignored: tuple[int, ...]


def load_ground_truth(doc: Mapping, task: TaskSpec) -> dict[str, GroundTruthImage]:
    """Validate a COCO + vad_* label document and return ground truth by manifest image id."""
    # Numeric ids are lookup keys: a repeat would merge two images or move a box silently.
    for kind, rows in (("image", doc["images"]), ("category", doc["categories"]), ("annotation", doc["annotations"])):
        if dup := next((i for i, n in Counter(r.get("id") for r in rows).items() if n > 1), None):
            raise LabelError(f"duplicate {kind} id {dup!r}; every {kind} needs its own numeric id")
    by_name: dict[str, list] = {}
    for c in doc["categories"]:
        by_name.setdefault(c["name"], []).append(c["id"])
    if ambiguous := next(((n, ids) for n, ids in by_name.items() if len(ids) > 1), None):
        raise LabelError(f"category name {ambiguous[0]!r} is used by ids {' and '.join(map(str, ambiguous[1]))}")
    names = {c["id"]: c["name"] for c in doc["categories"]}
    if unknown := sorted(set(names.values()) - set(task.class_ids)):
        raise LabelError(f"label categories not in the task: {unknown}")
    if missing := sorted(set(task.class_ids) - set(names.values())):
        raise LabelError(f"label categories are missing task classes {missing}; labelling must cover every class")
    images: dict[int, dict] = {}
    seen: set[str] = set()
    for img in doc["images"]:
        image_id = img.get("vad_image_id")
        if not image_id:
            raise LabelError(f"image {img.get('id')!r} has no vad_image_id")
        if image_id in seen:
            raise LabelError(f"duplicate vad_image_id {image_id!r}")
        if img.get("vad_exhaustive") is not True:
            raise LabelError(f"image {image_id!r} is not marked vad_exhaustive: true (all classes labelled)")
        seen.add(image_id)
        images[img["id"]] = img
    boxes: dict[int, list[GroundTruthBox]] = {i: [] for i in images}
    for ann in doc["annotations"]:
        img = images.get(ann["image_id"])
        if img is None:
            raise LabelError(f"annotation {ann.get('id')!r} refers to unknown image {ann['image_id']!r}")
        if ann["category_id"] not in names:
            raise LabelError(f"annotation {ann.get('id')!r} has unknown category {ann['category_id']!r}")
        x, y, w, h = ann["bbox"]
        if not (w > 0 and h > 0 and x >= 0 and y >= 0
                and x + w <= img["width"] + 1e-6 and y + h <= img["height"] + 1e-6):
            raise LabelError(f"annotation {ann.get('id')!r} has an empty or out-of-image bbox {ann['bbox']}")
        ignore = bool(ann.get("ignore") or ann.get("iscrowd"))
        boxes[ann["image_id"]].append(GroundTruthBox(names[ann["category_id"]], (x, y, x + w, y + h), ignore))
    return {img["vad_image_id"]: GroundTruthImage(
        image_id=img["vad_image_id"], width=img["width"], height=img["height"], boxes=tuple(boxes[key]),
        ambiguous=bool(img.get("vad_ambiguous", False)), reviewers=tuple(img.get("vad_reviewers", ())))
        for key, img in images.items()}


def match_image(predictions: Sequence[Prediction], gt: GroundTruthImage, iou_threshold: float = 0.5,
                class_ids: Sequence[str] | None = None) -> ImageMatch:
    """One-to-one, class-aware matching of predictions to one image's ground truth."""
    if class_ids is not None:
        if unknown := sorted({p.class_id for p in predictions} - set(class_ids)):
            raise LabelError(f"predictions of classes not in the task: {unknown}")
    # Highest score first; score-free predictions keep their input order, after scored ones.
    order = sorted(range(len(predictions)),
                   key=lambda i: (predictions[i].score is None, -(predictions[i].score or 0.0)))
    matched: set[int] = set()
    tp: list[tuple[int, int]] = []
    fp: list[int] = []
    ignored: list[int] = []
    for i in order:
        pred = predictions[i]
        best, best_iou = None, iou_threshold
        overlaps_ignore = False
        for j, box in enumerate(gt.boxes):
            if box.class_id != pred.class_id:
                continue
            overlap = iou(pred.box_xyxy, box.box_xyxy)
            if box.ignore:
                overlaps_ignore = overlaps_ignore or overlap >= iou_threshold
            elif j not in matched and overlap >= best_iou and (best is None or overlap > best_iou):
                best, best_iou = j, overlap
        if best is not None:
            matched.add(best)
            tp.append((i, best))
        elif overlaps_ignore:
            ignored.append(i)
        else:
            fp.append(i)
    fn = [j for j, box in enumerate(gt.boxes) if not box.ignore and j not in matched]
    return ImageMatch(tp=tuple(sorted(tp)), fp=tuple(sorted(fp)), fn=tuple(fn), ignored=tuple(sorted(ignored)))


def average_precision(predictions: Mapping[str, Sequence[Prediction]], ground_truth: Mapping[str, GroundTruthImage],
                      class_id: str, iou_threshold: float) -> float | None:
    """COCO-style 101-point interpolated AP for one class at one IoU threshold; None without ground truth.

    Detections are matched per image (highest score first, see :func:`match_image`), pooled
    across images in score order, and turned into a monotone precision envelope sampled at
    recall 0.00, 0.01, ..., 1.00. Detections on ignore boxes take no part. Unlike COCO there is
    no per-image cap of 100 detections, and an ignore box uses ordinary IoU (COCO's crowd boxes
    use intersection over the detection's area).
    """
    pooled: list[tuple[float, bool]] = []
    n_boxes = 0
    for image_id in sorted(ground_truth):
        gt = ground_truth[image_id]
        n_boxes += sum(1 for box in gt.boxes if box.class_id == class_id and not box.ignore)
        if any(p.score is None for p in predictions.get(image_id, ()) if p.class_id == class_id):
            raise LabelError(f"image {image_id!r} has a prediction without a score; AP needs scored detections")
        preds = sorted((p for p in predictions.get(image_id, ()) if p.class_id == class_id), key=lambda p: -p.score)
        match = match_image(preds, gt, iou_threshold)
        true_positives, neutral = {i for i, _ in match.tp}, set(match.ignored)
        pooled += [(p.score, i in true_positives) for i, p in enumerate(preds) if i not in neutral]
    return _ap_from_pooled(pooled, n_boxes)


def _ap_from_pooled(pooled: list[tuple[float, bool]], n_boxes: int) -> float | None:
    """101-point AP from (score, true positive?) pairs pooled in image order; None without ground truth."""
    if n_boxes == 0:
        return None
    pooled = sorted(pooled, key=lambda item: -item[0])  # stable: ties keep image order, then within-image order
    hits = np.array([hit for _, hit in pooled], dtype=bool)
    tp, fp = np.cumsum(hits), np.cumsum(~hits)
    recall = tp / n_boxes
    precision = tp / np.maximum(tp + fp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # best precision at this recall or higher
    at = np.searchsorted(recall, _RECALL_POINTS, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in at]))


def mean_average_precision(predictions: Mapping[str, Sequence[Prediction]],
                           ground_truth: Mapping[str, GroundTruthImage], class_ids: Sequence[str]) -> dict:
    """mAP50-95 and mAP50 over the classes that have ground truth (classes without any are listed).

    Needs scored detector predictions; accepted (score-free) annotations are evaluated by
    precision and recall instead.
    """
    if unknown := sorted(set(predictions) - set(ground_truth)):
        raise LabelError(f"predictions for images not in the ground truth: {unknown[:5]}")
    for image_id, preds in predictions.items():
        if any(p.score is None for p in preds):
            raise LabelError(f"image {image_id!r} has a prediction without a score; mAP needs scored detections")
        if unknown := sorted({p.class_id for p in preds} - set(class_ids)):
            raise LabelError(f"predictions of classes not in the task: {unknown}")
    per_class: dict[str, dict[str, float]] = {}
    without: list[str] = []
    for class_id in class_ids:
        aps = [average_precision(predictions, ground_truth, class_id, t) for t in IOU_THRESHOLDS]
        if aps[0] is None:
            without.append(class_id)
            continue
        per_class[class_id] = {"ap50_95": float(np.mean(aps)), "ap50": aps[0]}
    return {
        "map50_95": float(np.mean([v["ap50_95"] for v in per_class.values()])) if per_class else None,
        "map50": float(np.mean([v["ap50"] for v in per_class.values()])) if per_class else None,
        "per_class": per_class, "classes_without_ground_truth": without,
        "iou_thresholds": list(IOU_THRESHOLDS),
    }


# --------------------------------------------------------------------------- run report

SIZE_STRATA = (("small", 32 ** 2), ("medium", 96 ** 2), ("large", float("inf")))  # COCO areas, in pixels


@dataclass(frozen=True)
class ImageResult:
    """What a run produced for one image, reduced to what the metrics need."""

    detections: tuple[Prediction, ...]  # scored detector output (primary candidates)
    accepted: tuple[Prediction, ...]    # accepted annotations; no scores
    flagged_for_review: bool            # completeness is not "complete": a human will look at it
    routed: tuple[tuple[Prediction, str], ...] = ()  # unrejected final candidates (routed-score-v1), with status


ROUTED_RULE = ("routed-score-v1: every final candidate that is not rejected, scored with the highest raw_confidence "
               "among its own source detections; a VLM verdict never changes a score (it can only reject); a "
               "candidate without a scored source is left out of AP and counted as unscored_routed")
ROUTED_VARIANTS = (("all_unrejected", ("accepted", "candidate", "needs_review")),
                   ("accepted_only", ("accepted",)))  # candidate and needs_review are both still unverified


def _routed(record) -> tuple[tuple[Prediction, str], ...]:
    confidence = {d.id: d.raw_confidence
                  for d in (*record.primary_candidates, *record.secondary_candidates, *record.sahi_candidates)}
    out = []
    for c in record.final_candidates:
        if c.status != "rejected":
            scores = [confidence[s] for s in c.source_ids if confidence.get(s) is not None]
            out.append((Prediction(c.class_id, c.box_xyxy, max(scores) if scores else None), c.status))
    return tuple(out)


def image_results(records: Sequence) -> dict[str, ImageResult]:
    """Reduce RouteRecords to :class:`ImageResult` by image id; a repeated image id is refused, not resolved."""
    if repeated := repeated_ids(r.image.id for r in records):
        raise LabelError(f"repeated image ids in the trace {repeated[:10]}; evaluation needs one result per image "
                         "per run (rebuild a concatenated or retried trace)")
    return {r.image.id: ImageResult(
        detections=tuple(Prediction(d.class_id, d.box_xyxy, d.raw_confidence) for d in r.primary_candidates),
        accepted=tuple(Prediction(c.class_id, c.box_xyxy, None) for c in r.final_candidates if c.status == "accepted"),
        flagged_for_review=r.completeness.status != "complete", routed=_routed(r)) for r in records}


def evidence_status(label_doc: Mapping, split_status: str, run_count: int = 1) -> str:
    """Metrics count as evidence only with BOTH human labels (top-level ``vad_label_provenance: "human"``)
    and a verified frozen split (``split_status`` "verified frozen_eval": labels and image content bound
    by a frozen-lock-v2), from ONE run (``run_count`` distinct RunInfos in the cohort). Otherwise the
    metrics are still reported, labelled NOT A FINDING with what is missing.
    """
    missing = []
    if label_doc.get("vad_label_provenance") != "human":
        missing.append('the label file does not declare "vad_label_provenance": "human" (synthetic or unreviewed labels)')
    if split_status != "verified frozen_eval":
        missing.append(f"no verified frozen split ({split_status})")
    if run_count > 1:
        missing.append(f"the cohort mixes {run_count} runs (different RunInfo); evaluate each run separately")
    if not missing:
        return "evidence: human labels on a verified frozen split"
    return "NOT A FINDING: " + "; ".join(missing) + "; these numbers describe the run, they are not evidence"


def _size_stratum(box: Box) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1])
    return next(name for name, limit in SIZE_STRATA if area < limit)


def _rate(found: int, total: int) -> float | None:
    return found / total if total else None


def evaluate(ground_truth: Mapping[str, GroundTruthImage], class_ids: Sequence[str],
             results: Mapping[str, ImageResult], iou_threshold: float = 0.5) -> dict:
    """Detector and accepted-annotation metrics over the labelled images.

    A labelled image without a result is an operational failure: it is listed, and all its
    ground truth counts as missed (never dropped). Results for unlabelled images are only counted.
    Accepted annotations carry no scores, so they get precision/recall, not AP.
    """
    failed = sorted(set(ground_truth) - set(results))
    empty = ImageResult((), (), False)
    by_class = {c: [0, 0] for c in class_ids}
    by_size = {name: [0, 0] for name, _ in SIZE_STRATA}
    det_fp = neg_fp = 0
    acc = dict.fromkeys(("tp", "fp", "missed", "failed", "review", "silent"), 0)
    for image_id, gt in sorted(ground_truth.items()):
        result = results.get(image_id, empty)
        det = match_image(result.detections, gt, iou_threshold, class_ids)
        found = {j for _, j in det.tp}
        for j, box in enumerate(gt.boxes):
            if not box.ignore:
                for counts in (by_class[box.class_id], by_size[_size_stratum(box.box_xyxy)]):
                    counts[0] += j in found
                    counts[1] += 1
        det_fp += len(det.fp)
        if not any(not box.ignore for box in gt.boxes):
            neg_fp += len(det.fp)
        ann = match_image(result.accepted, gt, iou_threshold, class_ids)
        acc["tp"] += len(ann.tp)
        acc["fp"] += len(ann.fp)
        acc["missed"] += len(ann.fn)
        if image_id in failed:
            acc["failed"] += len(ann.fn)
        elif result.flagged_for_review:
            acc["review"] += len(ann.fn)
        else:
            acc["silent"] += len(ann.fn)
    regular = sum(1 for gt in ground_truth.values() for box in gt.boxes if not box.ignore)
    negatives = sum(1 for gt in ground_truth.values() if not any(not box.ignore for box in gt.boxes))
    detections = {i: results[i].detections for i in ground_truth if i in results and results[i].detections}
    from .coco_eval import coco_map, coco_unavailable  # here: coco_eval imports this module

    # Standard names (map50_95, map50) appear only in the official evaluator's "coco" block; the
    # custom evaluator's results carry a custom_ prefix (it differs from COCO on the 100-detection
    # cap and on how ignore/crowd boxes match).
    detector = {"source": "primary_candidates", "evaluator": "custom-vad-1"}
    unscored = sum(1 for preds in detections.values() for p in preds if p.score is None)
    if unscored:
        reason = f"{unscored} detections have no confidence; AP needs scored detections"
        detector.update(custom_map50_95=None, custom_map50=None, map_unavailable=reason)
        detector["coco"] = coco_unavailable(reason)
    else:
        maps = mean_average_precision(detections, ground_truth, class_ids)
        detector.update(custom_map50_95=maps["map50_95"], custom_map50=maps["map50"],
                        custom_ap_per_class=maps["per_class"],
                        classes_without_ground_truth=maps["classes_without_ground_truth"])
        detector["coco"] = coco_map(detections, ground_truth, class_ids)
    detector.update(
        recall_by_class={c: {"found": f, "boxes": n, "recall": _rate(f, n)} for c, (f, n) in by_class.items()},
        recall_by_size={s: {"found": f, "boxes": n, "recall": _rate(f, n)} for s, (f, n) in by_size.items()},
        fp_per_image=_rate(det_fp, len(ground_truth)), fp_per_negative_image=_rate(neg_fp, negatives))
    routed: dict = {"rule": ROUTED_RULE}
    for name, statuses in ROUTED_VARIANTS:
        kept = {i: [(p, s) for p, s in results[i].routed if s in statuses] for i in ground_truth if i in results}
        scored = {i: [p for p, _ in kept[i] if p.score is not None] for i in kept}
        maps = mean_average_precision(scored, ground_truth, class_ids)
        routed[name] = {"statuses": sorted(statuses), "scored": sum(map(len, scored.values())),
                        "unscored_routed": sum(1 for i in kept for p, _ in kept[i] if p.score is None),
                        "custom_map50_95": maps["map50_95"], "custom_map50": maps["map50"],
                        "custom_ap_per_class": maps["per_class"],
                        "classes_without_ground_truth": maps["classes_without_ground_truth"],
                        "coco": coco_map(scored, ground_truth, class_ids)}
    return {
        "iou_threshold": iou_threshold,
        "images": {"labelled": len(ground_truth), "with_result": len(ground_truth) - len(failed),
                   "negative": negatives, "operational_failures": failed,
                   "unlabelled_results": len(set(results) - set(ground_truth))},
        "detector": detector,
        "routed": routed,
        "accepted_annotations": {
            "true_positives": acc["tp"], "false_positives": acc["fp"],
            "precision": _rate(acc["tp"], acc["tp"] + acc["fp"]), "recall": _rate(acc["tp"], regular),
            "missed": acc["missed"], "missed_in_failed_images": acc["failed"], "missed_in_review": acc["review"],
            "silent_misses": acc["silent"], "review_coverage": _rate(acc["review"], acc["review"] + acc["silent"]),
        },
    }


# --------------------------------------------------------------------------- per-image outcomes (run-metrics-v1)


def _outcomes(predictions: Sequence[Prediction], gt: GroundTruthImage, class_ids: Sequence[str]) -> dict:
    """Per class: ground-truth count and, for each scored prediction (score order), a match code per IoU
    threshold: 1 true positive, 0 false positive, -1 neutral (on an ignore box). Ids and scores only."""
    classes = {}
    for c in class_ids:
        preds = sorted((p for p in predictions if p.class_id == c and p.score is not None), key=lambda p: -p.score)
        n_boxes = sum(1 for box in gt.boxes if box.class_id == c and not box.ignore)
        if not preds and not n_boxes:
            continue
        codes = [[0] * len(IOU_THRESHOLDS) for _ in preds]
        for k, threshold in enumerate(IOU_THRESHOLDS):
            match = match_image(preds, gt, threshold)
            for i, _ in match.tp:
                codes[i][k] = 1
            for i in match.ignored:
                codes[i][k] = -1
        classes[c] = {"gt": n_boxes, "ranked": [[p.score, codes[i]] for i, p in enumerate(preds)]}
    return {"unscored": sum(1 for p in predictions if p.score is None), "classes": classes}


def per_image_outcomes(ground_truth: Mapping[str, GroundTruthImage], results: Mapping[str, ImageResult],
                       class_ids: Sequence[str]) -> dict[str, dict]:
    """What a paired bootstrap needs per labelled image, for primary and both routed variants."""
    out = {}
    for image_id, gt in sorted(ground_truth.items()):
        result = results.get(image_id)
        routed = result.routed if result else ()
        out[image_id] = {"result": result is not None,
                         "primary": _outcomes(result.detections if result else (), gt, class_ids),
                         **{source: _outcomes([p for p, s in routed if s in statuses], gt, class_ids)
                            for source, statuses in (("routed_all", ROUTED_VARIANTS[0][1]),
                                                     ("routed_accepted", ROUTED_VARIANTS[1][1]))}}
    return out


def map_from_outcomes(outcomes: Sequence[Mapping], class_ids: Sequence[str]) -> dict:
    """mAP50-95 and mAP50 (custom-vad-1) recomputed from stored per-image outcomes, pooled in the given
    image order; on the sorted full cohort it equals :func:`mean_average_precision`."""
    per_class: dict[str, dict[str, float]] = {}
    without: list[str] = []
    for c in class_ids:
        entries = [o["classes"][c] for o in outcomes if c in o["classes"]]
        n_boxes = sum(e["gt"] for e in entries)
        aps = [_ap_from_pooled([(score, codes[k] == 1) for e in entries for score, codes in e["ranked"]
                                if codes[k] != -1], n_boxes) for k in range(len(IOU_THRESHOLDS))]
        if aps[0] is None:
            without.append(c)
            continue
        per_class[c] = {"ap50_95": float(np.mean(aps)), "ap50": aps[0]}
    return {"map50_95": float(np.mean([v["ap50_95"] for v in per_class.values()])) if per_class else None,
            "map50": float(np.mean([v["ap50"] for v in per_class.values()])) if per_class else None,
            "per_class": per_class, "classes_without_ground_truth": without}
