"""Class-aware matching, directional disagreement, and provenance-preserving fusion.

All inputs must already be in original-image xyxy pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schemas import Detection, Disagreement, FinalCandidate, stable_id

Box = tuple[float, float, float, float]

# Tie rule for choosing a canonical box: highest raw confidence, then source
# priority, then lexicographically smallest detection id.
SOURCE_PRIORITY = {"human": 0, "primary": 1, "secondary": 2, "sahi": 3}


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ios(a: Box, b: Box) -> float:
    """Intersection over the smaller box; robust for boxes cut at tile edges."""
    small = min(area(a), area(b))
    return intersection(a, b) / small if small > 0 else 0.0


def canonical_key(d: Detection) -> tuple:
    conf = d.raw_confidence if d.raw_confidence is not None else -1.0
    return (-conf, SOURCE_PRIORITY[d.source], d.id)


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[tuple[Detection, Detection, float], ...]
    primary_only: tuple[Detection, ...]
    secondary_only: tuple[Detection, ...]
    cross_class: tuple[tuple[Detection, Detection, float], ...]


def match(primary: Sequence[Detection], secondary: Sequence[Detection], iou_threshold: float) -> MatchResult:
    """Greedy one-to-one class-aware matching by descending IoU (ties broken by ids).

    Class mapping is applied when detections are created (see adapters.to_detections),
    so ``class_id`` values here are already task class IDs.
    """
    cands = sorted(
        ((iou(p.box_xyxy, s.box_xyxy), p, s) for p in primary for s in secondary
         if p.class_id == s.class_id),
        key=lambda t: (-t[0], t[1].id, t[2].id),
    )
    used_p: set[str] = set()
    used_s: set[str] = set()
    pairs = []
    for v, p, s in cands:
        if v < iou_threshold:
            break
        if p.id in used_p or s.id in used_s:
            continue
        used_p.add(p.id)
        used_s.add(s.id)
        pairs.append((p, s, v))
    p_only = tuple(p for p in primary if p.id not in used_p)
    s_only = tuple(s for s in secondary if s.id not in used_s)
    cross = tuple(
        (p, s, iou(p.box_xyxy, s.box_xyxy)) for p in p_only for s in s_only
        if p.class_id != s.class_id and iou(p.box_xyxy, s.box_xyxy) >= iou_threshold
    )
    return MatchResult(tuple(pairs), p_only, s_only, cross)


def disagreement(m: MatchResult, mislocalized_iou: float) -> Disagreement:
    ious = [v for _, _, v in m.pairs]
    return Disagreement(
        status="computed",
        matched=len(m.pairs),
        primary_only=len(m.primary_only),
        secondary_only=len(m.secondary_only),
        cross_class=len(m.cross_class),
        mislocalized=sum(v < mislocalized_iou for v in ious),
        mean_matched_iou=round(sum(ious) / len(ious), 4) if ious else None,
        primary_only_ids=tuple(d.id for d in m.primary_only),
        secondary_only_ids=tuple(d.id for d in m.secondary_only),
    )


def cluster(dets: Sequence[Detection], threshold: float, metric=iou) -> list[list[Detection]]:
    """Greedy class-aware clustering in canonical order; each detection joins at most one cluster."""
    clusters: list[list[Detection]] = []
    for d in sorted(dets, key=canonical_key):
        for c in clusters:
            if c[0].class_id == d.class_id and metric(c[0].box_xyxy, d.box_xyxy) >= threshold:
                c.append(d)
                break
        else:
            clusters.append([d])
    return clusters


@dataclass(frozen=True)
class FusedCluster:
    canonical: Detection
    members: tuple[Detection, ...]
    sources: frozenset[str]
    cross_class_conflict: bool


def fuse(dets: Sequence[Detection], iou_threshold: float) -> list[FusedCluster]:
    """Group duplicates across sources without discarding any member.

    The canonical box is the cluster head under :func:`canonical_key`; every
    member id stays in the cluster so disagreement remains auditable.
    """
    groups = cluster(dets, iou_threshold)
    fused = []
    for i, g in enumerate(groups):
        conflict = any(
            h[0].class_id != g[0].class_id and iou(h[0].box_xyxy, g[0].box_xyxy) >= iou_threshold
            for j, h in enumerate(groups) if j != i
        )
        fused.append(FusedCluster(g[0], tuple(g), frozenset(d.source for d in g), conflict))
    return fused


def final_candidate(fc: FusedCluster, status, flags: Sequence[str] = (), verification=None) -> FinalCandidate:
    src = tuple(sorted(d.id for d in fc.members))
    return FinalCandidate(
        id=stable_id("cand", src), class_id=fc.canonical.class_id, box_xyxy=fc.canonical.box_xyxy,
        status=status, source_ids=src, canonical_source_id=fc.canonical.id,
        flags=tuple(flags), verification=verification,
    )
