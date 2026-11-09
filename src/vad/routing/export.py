"""Versioned annotation export.

Greenfield default: COCO-style JSON with arbitrary category names taken from
the task registry (no COCO class list is assumed). Only ``accepted`` final
candidates become annotations; everything else stays in the route trace.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .schemas import Detection, RouteRecord, TaskSpec, stable_id


def _categories(task: TaskSpec) -> list[dict]:
    return [{"id": i + 1, "name": c.class_id, "display_name": c.display_name} for i, c in enumerate(task.classes)]


def _ann(i: int, image_idx: int, cat: int, box, extra: dict) -> dict:
    x1, y1, x2, y2 = box
    return {"id": i, "image_id": image_idx, "category_id": cat, "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, **extra}


def export_routed(records: Sequence[RouteRecord], task: TaskSpec, out: str | Path) -> dict:
    cat = {c["name"]: c["id"] for c in _categories(task)}
    images, anns = [], []
    for idx, r in enumerate(sorted(records, key=lambda r: r.image.id), start=1):
        images.append({"id": idx, "file_name": r.image.uri, "width": r.image.width, "height": r.image.height,
                       "vad_image_id": r.image.id, "vad_record_id": r.record_id,
                       "vad_completeness": r.completeness.status})
        for c in r.final_candidates:
            if c.status == "accepted":
                anns.append(_ann(len(anns) + 1, idx, cat[c.class_id], c.box_xyxy,
                                 {"vad_candidate_id": c.id, "vad_source_ids": list(c.source_ids)}))
    return _write(task, images, anns, out, {"kind": "routed", "record_ids": sorted(r.record_id for r in records)})


def export_legacy(preds: Sequence[tuple[dict, Sequence[Detection]]], task: TaskSpec, out: str | Path) -> dict:
    """Primary-only export used when the routing feature flag is off."""
    cat = {c["name"]: c["id"] for c in _categories(task)}
    images, anns = [], []
    for idx, (img, dets) in enumerate(preds, start=1):
        images.append({"id": idx, **img})
        for d in dets:
            anns.append(_ann(len(anns) + 1, idx, cat[d.class_id], d.box_xyxy, {"score": d.raw_confidence}))
    return _write(task, images, anns, out, {"kind": "legacy_primary_only"})


def _write(task: TaskSpec, images, anns, out, info: dict) -> dict:
    doc = {"info": {"task_id": task.task_id, "prompt_set_version": task.prompt_set_version, **info},
           "categories": _categories(task), "images": images, "annotations": anns}
    doc["info"]["version_id"] = stable_id("ann", {k: doc[k] for k in ("categories", "images", "annotations")})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def load_annotations(path: str | Path) -> dict:
    """Minimal loader standing in for the (not yet existing) training pipeline's dataset reader."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in doc["categories"]}
    for a in doc["annotations"]:
        x, y, w, h = a["bbox"]
        img = next(i for i in doc["images"] if i["id"] == a["image_id"])
        if not (w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= img["width"] + 1e-6 and y + h <= img["height"] + 1e-6):
            raise ValueError(f"annotation {a['id']} out of bounds")
        if a["category_id"] not in cats:
            raise ValueError(f"annotation {a['id']} has unknown category")
    expected = stable_id("ann", {k: doc[k] for k in ("categories", "images", "annotations")})
    if doc["info"].get("version_id") != expected:
        raise ValueError("annotation version_id does not match content")
    return doc
