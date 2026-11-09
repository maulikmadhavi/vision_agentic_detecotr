"""Sliced (SAHI-style) inference through any crop-capable detector adapter.

Native implementation so the router does not depend on a specific SAHI
version; tile boxes are shifted to original-image pixels exactly once (in
``adapters.to_detections``), clipped to image bounds, then merged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from PIL import Image

from .adapters import Detector, to_detections
from .matching import canonical_key, ios
from .schemas import Detection, ImageRef, stable_id
from .task_config import SlicingConfig


def _starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)  # last tile flush with the edge
    return sorted(set(starts))


def tile_grid(width: int, height: int, cfg: SlicingConfig) -> list[tuple[int, int, int, int]]:
    step = max(1, int(cfg.tile * (1 - cfg.overlap)))
    return [
        (x, y, min(x + cfg.tile, width), min(y + cfg.tile, height))
        for y in _starts(height, cfg.tile, step) for x in _starts(width, cfg.tile, step)
    ]


def planned_slices(width: int, height: int, cfg: SlicingConfig) -> int:
    return len(tile_grid(width, height, cfg)) + (1 if cfg.include_full_image else 0)


@dataclass
class SliceResult:
    detections: list[Detection]  # merged; ``parent_ids`` point into ``raw``
    raw: list[Detection]  # every per-tile detection, kept so provenance resolves
    slices_run: int
    gpu_ms: float | None
    warnings: list[str]
    stopped_reason: str | None = None  # set when a budget cut tiling short (partial result)

    @property
    def raw_count(self) -> int:
        return len(self.raw)


def _union(group: Sequence[Detection]) -> tuple[float, float, float, float]:
    return (min(d.box_xyxy[0] for d in group), min(d.box_xyxy[1] for d in group),
            max(d.box_xyxy[2] for d in group), max(d.box_xyxy[3] for d in group))


def merge(dets: Sequence[Detection], ios_threshold: float) -> list[Detection]:
    """Non-maximum merging per class by IoS against each cluster's growing union box.

    Repeats until no two same-class clusters overlap above the threshold, so an
    object cut by several tiles is reassembled regardless of processing order.
    Merged box is the union; every parent detection id is kept.
    """
    groups = [[d] for d in sorted(dets, key=canonical_key)]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if a and b and a[0].class_id == b[0].class_id and ios(_union(a), _union(b)) >= ios_threshold:
                    a.extend(b)
                    b.clear()
                    changed = True
        groups = [g for g in groups if g]
    out = []
    for group in groups:
        if len(group) == 1:
            out.append(group[0])
            continue
        head = min(group, key=canonical_key)
        box = _union(group)
        parents = tuple(sorted(d.id for d in group))
        out.append(head.model_copy(update={
            "id": stable_id("det", ["merged", parents]), "box_xyxy": box, "parent_ids": parents,
        }))
    return out


def run_sliced(
    detector: Detector,
    image: Image.Image,
    ref: ImageRef,
    class_ids: Sequence[str],
    cfg: SlicingConfig,
    class_mapping: Mapping[str, str] | None = None,
    max_slices: int | None = None,
    stop_check: Callable[[float | None], str | None] | None = None,
) -> SliceResult:
    """Run the detector per tile. Budgets are checked before *every* tile.

    ``stop_check(gpu_ms_so_far)`` returns a reason string to stop early (e.g.
    elapsed or GPU budget); ``max_slices`` caps tiles run. On early stop the
    partial detections are returned with ``stopped_reason`` set so the caller
    can fall back to review instead of treating the result as complete.
    """
    tiles = tile_grid(ref.width, ref.height, cfg)
    if cfg.include_full_image:
        tiles = [(0, 0, ref.width, ref.height), *tiles]
    raw: list[Detection] = []
    warnings: list[str] = []
    gpu = 0.0
    gpu_known = True
    run = 0
    stopped: str | None = None
    for x1, y1, x2, y2 in tiles:
        if max_slices is not None and run >= max_slices:
            stopped = "slice_budget_exhausted"
            break
        if stop_check is not None and (reason := stop_check(gpu if gpu_known else None)):
            stopped = reason
            break
        run += 1
        res = detector.detect(image.crop((x1, y1, x2, y2)), ref.id, class_ids)
        dets, w = to_detections(res.detections, ref, class_ids, "sahi", detector.model_version,
                                class_mapping, origin=(x1, y1), clip_to=(x1, y1, x2, y2))
        raw.extend(dets)
        warnings.extend(w)
        if res.gpu_ms is None:
            gpu_known = False
        else:
            gpu += res.gpu_ms
    return SliceResult(merge(raw, cfg.merge_ios), raw, run, gpu if gpu_known else None, warnings, stopped)
