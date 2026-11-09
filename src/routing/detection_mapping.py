"""Map raw detector boxes into validated original-image detections (class mapping, crop offset, clipping)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .interfaces import RawDetection
from .schemas import Detection, ImageRef, Source, stable_id


def to_detections(
    raw: Sequence[RawDetection],
    image: ImageRef,
    class_ids: Sequence[str],
    source: Source,
    model_version: str,
    class_mapping: Mapping[str, str] | None = None,
    origin: tuple[int, int] = (0, 0),
    clip_to: tuple[int, int, int, int] | None = None,
) -> tuple[list[Detection], list[str]]:
    """Map raw boxes into validated original-image detections with stable IDs.

    ``origin`` shifts crop-local boxes into original coordinates exactly once.
    Returns (detections, warnings); unknown classes and degenerate boxes are
    dropped and reported rather than silently coerced.
    """
    mapping = class_mapping or {}
    cx1, cy1, cx2, cy2 = clip_to or (0, 0, image.width, image.height)
    out: list[Detection] = []
    warnings: list[str] = []
    ox, oy = origin
    for r in raw:
        cls = mapping.get(r.class_id, r.class_id)
        if cls not in class_ids:
            warnings.append(f"{source}:unknown_class_dropped:{r.class_id}")
            continue
        x1, y1, x2, y2 = (float(v) for v in r.box_xyxy)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            warnings.append(f"{source}:nonfinite_box_dropped:{cls}")
            continue
        box = tuple(round(v, 2) for v in
                    (max(x1 + ox, cx1), max(y1 + oy, cy1), min(x2 + ox, cx2), min(y2 + oy, cy2)))
        if not (box[0] < box[2] and box[1] < box[3]):
            warnings.append(f"{source}:degenerate_box_dropped:{cls}")
            continue
        conf = r.confidence
        if conf is not None:
            conf = float(conf)
            if not math.isfinite(conf):
                warnings.append(f"{source}:nonfinite_confidence_dropped:{cls}")
                conf = None
            elif not 0.0 <= conf <= 1.0:
                warnings.append(f"{source}:confidence_clamped:{cls}")
                conf = min(1.0, max(0.0, conf))
        slice_origin = origin if source == "sahi" else None
        det_id = stable_id("det", [image.sha256, source, model_version, cls, box, slice_origin, conf])
        out.append(Detection(
            id=det_id, class_id=cls, box_xyxy=box, raw_confidence=conf,
            source=source, model_version=model_version, slice_origin=slice_origin,
        ))
    return out, warnings
