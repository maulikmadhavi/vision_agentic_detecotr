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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .matching import iou
from .schemas import TaskSpec

Box = tuple[float, float, float, float]


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
