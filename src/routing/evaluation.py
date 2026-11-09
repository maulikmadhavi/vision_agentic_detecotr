"""Data isolation, frozen benchmark locking, and route/cost reports.

Frozen-evaluation annotations are locked by content hash before any tuning.
Nothing in this module writes to a frozen set after it is locked.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .schemas import ImageRef, RouteRecord, VerificationResult, canonical_json

SPLITS = ("train", "calibration", "development", "frozen_eval")


class IsolationError(ValueError):
    pass


def check_isolation(split_groups: Mapping[str, Sequence[str]]) -> None:
    """Raise if any source/near-duplicate group appears in more than one split."""
    unknown = set(split_groups) - set(SPLITS)
    if unknown:
        raise IsolationError(f"unknown splits {sorted(unknown)}")
    seen: dict[str, str] = {}
    for split in SPLITS:
        for g in split_groups.get(split, ()):
            if g in seen and seen[g] != split:
                raise IsolationError(f"group {g!r} is in both {seen[g]!r} and {split!r}")
            seen[g] = split


def assert_not_frozen(image_groups: Mapping[str, str], lock: Mapping, image_ids: Sequence[str]) -> None:
    """Guard for training / threshold fitting / prompt selection inputs."""
    frozen_groups = set(lock["group_ids"])
    leaked = sorted(i for i in image_ids if image_groups.get(i) in frozen_groups or i in set(lock["image_ids"]))
    if leaked:
        raise IsolationError(f"frozen-evaluation images used outside evaluation: {leaked[:5]}")


def assert_not_frozen_in_split(split_manifest: Mapping, image_ids: Sequence[str]) -> None:
    """Refuse any image the split manifest freezes, by id or by frozen_eval group; unknown ids pass."""
    images = split_manifest["images"]
    frozen_ids = [i for i, v in images.items() if v["split"] == "frozen_eval"]
    frozen_groups = ({images[i].get("group_id") for i in frozen_ids}
                     | {g for g, v in split_manifest.get("groups", {}).items() if v["split"] == "frozen_eval"})
    assert_not_frozen({i: v.get("group_id") for i, v in images.items()},
                      {"group_ids": sorted(g for g in frozen_groups if g), "image_ids": frozen_ids}, image_ids)


def assert_split_allows(split_manifest: Mapping, image_ids: Sequence[str],
                        allowed_splits: Sequence[str] | None = None) -> None:
    """Guard for fitting, threshold selection and prompt choice over a split manifest (``sample.make_split``).

    Refuses images the manifest does not know, any image of a frozen_eval group (by id or by
    group, so a late near-duplicate cannot slip in) and, when given, images outside ``allowed_splits``.
    """
    images = split_manifest["images"]
    if unknown := sorted(i for i in image_ids if i not in images):
        raise IsolationError(f"images not in the split manifest: {unknown[:5]}")
    assert_not_frozen_in_split(split_manifest, image_ids)
    if allowed_splits is not None:
        if outside := sorted(i for i in image_ids if images[i]["split"] not in allowed_splits):
            raise IsolationError(f"images outside {list(allowed_splits)}: "
                                 f"{[(i, images[i]['split']) for i in outside[:5]]}")


def _digest(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


LOCK_VERSION = "frozen-lock-v2"
_LOCK_V1_FIELDS = ("image_ids", "group_ids", "annotations_sha256")
_LOCK_V2_FIELDS = ("version", *_LOCK_V1_FIELDS, "images")
CONTENT_FIELDS = ("sha256", "width", "height")


def lock_frozen_set(image_ids: Sequence[str], group_ids: Sequence[str], annotations: Mapping,
                    lock_path: str | Path, *, content: Mapping[str, Mapping]) -> dict:
    """Write a lock pinning frozen image IDs, groups, annotation content and each frozen image's
    sha256 and size (``content``: image id -> {sha256, width, height}). Refuses to overwrite."""
    p = Path(lock_path)
    if p.exists():
        raise FileExistsError(f"frozen lock already exists: {p}; create a new versioned lock instead")
    lock = {"version": LOCK_VERSION, "image_ids": sorted(image_ids), "group_ids": sorted(set(group_ids)),
            "annotations_sha256": _digest(annotations),
            "images": {i: {k: content[i][k] for k in CONTENT_FIELDS} for i in sorted(image_ids)}}
    lock["lock_sha256"] = _digest({k: lock[k] for k in _LOCK_V2_FIELDS})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def load_frozen(lock_path: str | Path, annotations: Mapping) -> dict:
    """Load and verify the frozen set; raises if the lock or the annotations have changed.

    A lock without ``version`` predates image hashes: it still verifies (over its own fields) and is
    returned as it is, never upgraded; callers report its image content as unverified.
    """
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if "version" in lock and lock["version"] != LOCK_VERSION:
        raise IsolationError(f"unknown frozen lock version {lock['version']!r}")
    fields = _LOCK_V2_FIELDS if "version" in lock else _LOCK_V1_FIELDS
    if lock["lock_sha256"] != _digest({k: lock[k] for k in fields}):
        raise IsolationError("frozen lock file was modified")
    if lock["annotations_sha256"] != _digest(annotations):
        raise IsolationError("frozen annotations do not match the lock")
    return lock


def check_image_content(expected: Mapping[str, Mapping], refs: Sequence[ImageRef], source: str) -> None:
    """Each ref must have the sha256, width and height ``expected`` gives for its id (the uri is not compared)."""
    for ref in refs:
        want = expected[ref.id]
        if diff := [f"{k} {want[k]!r} != {getattr(ref, k)!r}" for k in CONTENT_FIELDS if want[k] != getattr(ref, k)]:
            raise IsolationError(f"image {ref.id!r} is not the image in {source}: {'; '.join(diff)} "
                                 "(expected != routed)")


@dataclass(frozen=True)
class HumanLabel:
    """A human reviewer's answer to one verification question (ground truth for VLM agreement).

    Keyed by what was asked, not by the run-specific request id: ``image_id``, ``class_id``
    and ``candidate_ids`` (empty for a suspected-missing-class check). Candidate ids are
    hashes of detection ids, which do not depend on the VLM model or the routing config,
    so one label set can score several VLMs (e.g. 4B vs 8B) over the same detections.
    """

    image_id: str
    class_id: str
    candidate_ids: tuple[str, ...]
    verdict: Literal["present", "absent"]
    labeled_by: str  # reviewer identity; must not be a model

    @property
    def key(self) -> tuple[str, str, tuple[str, ...]]:
        return (self.image_id, self.class_id, tuple(self.candidate_ids))


def verification_items(records: Sequence[RouteRecord]) -> list[tuple[str, VerificationResult]]:
    """(image_id, result) pairs for :func:`vlm_agreement` from routed records."""
    return [(r.image.id, v) for r in records for v in r.verifications]


def _wilson_lower(k: int, n: int, z: float = 1.96) -> float | None:
    if n == 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((centre - margin) / (1 + z * z / n), 4)


def vlm_agreement(items: Sequence[tuple[str, VerificationResult]], labels: Sequence[HumanLabel]) -> dict:
    """Score VLM verdicts against human labels, per VLM model and request purpose.

    ``items`` are ``(image_id, result)`` pairs (see :func:`verification_items`).
    ``absent_precision`` (share of VLM "absent" that humans also call absent) and its
    Wilson 95% lower bound are the evidence needed before setting
    ``vlm.absent_rejects: true``. A label authored by a scored model is refused:
    a VLM's own judgement is never ground truth.
    """
    models = {r.model_version for _, r in items}
    by_key: dict[tuple, HumanLabel] = {}
    for lab in labels:
        if not lab.labeled_by.strip() or lab.labeled_by in models:
            raise ValueError(f"label for {lab.key!r} is not human-authored ({lab.labeled_by!r})")
        if lab.key in by_key and by_key[lab.key].verdict != lab.verdict:
            raise ValueError(f"conflicting human labels for {lab.key!r}; adjudicate first")
        by_key[lab.key] = lab

    groups: dict[str, list[tuple[VerificationResult, HumanLabel]]] = {}
    unlabeled = 0
    for image_id, r in items:
        lab = by_key.get((image_id, r.class_id, tuple(r.candidate_ids)))
        if lab is None:
            unlabeled += 1
            continue
        groups.setdefault(r.model_version, []).append((r, lab))

    def summarise(pairs: list[tuple[VerificationResult, HumanLabel]]) -> dict:
        conf = Counter(f"{r.verdict}|{h.verdict}" for r, h in pairs)
        decided = [(r, h) for r, h in pairs if r.verdict != "uncertain"]
        agree = sum(r.verdict == h.verdict for r, h in decided)
        out = {"n": len(pairs), "confusion_vlm|human": dict(sorted(conf.items())),
               "decided_rate": round(len(decided) / len(pairs), 4) if pairs else None,
               "accuracy_on_decided": round(agree / len(decided), 4) if decided else None}
        for v in ("present", "absent"):
            n_v = sum(r.verdict == v for r, _ in decided)
            k_v = conf.get(f"{v}|{v}", 0)
            out[f"{v}_precision"] = round(k_v / n_v, 4) if n_v else None
            out[f"{v}_precision_wilson_lower95"] = _wilson_lower(k_v, n_v)
            out[f"{v}_n"] = n_v
        return out

    report: dict = {"labeled": sum(len(v) for v in groups.values()), "unlabeled": unlabeled, "by_model": {}}
    for model, pairs in sorted(groups.items()):
        entry = summarise(pairs)
        entry["by_purpose"] = {p: summarise([(r, h) for r, h in pairs if _purpose(r) == p])
                               for p in ("confirm_candidate", "suspected_missing_class")}
        report["by_model"][model] = entry
    return report


def _purpose(r: VerificationResult) -> str:
    # confirm_candidate requests always cite candidate ids; missing-class checks never do.
    return "confirm_candidate" if r.candidate_ids else "suspected_missing_class"


def _pct(vals: Sequence[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def route_report(records: Sequence[RouteRecord]) -> dict:
    n = len(records)
    lat = [r.trace.elapsed_ms for r in records]
    review = sum(r.completeness.status != "complete" for r in records)
    scorer_failed = [w.split(":", 1)[1] for r in records for w in r.stage_a.warnings
                     if w.startswith("relevance_scorer_failed:")]
    return {
        "relevance_scorer_failures": {"count": len(scorer_failed), "types": dict(Counter(scorer_failed))},
        "images": n,
        "actions": dict(Counter(a for r in records for a in r.route.actions)),
        "reason_codes": dict(Counter(c for r in records for c in r.route.reason_codes)),
        "completeness": dict(Counter(r.completeness.status for r in records)),
        "difficulty_pre": dict(Counter(r.stage_b.difficulty_pre_secondary.bucket for r in records)),
        "relevance_band": dict(Counter(r.stage_a.relevance_band for r in records)),
        "review_rate": review / n if n else None,
        "vlm_calls_per_image": sum(r.trace.budget_used.vlm_calls for r in records) / n if n else None,
        "slices_per_image": sum(r.trace.budget_used.slices for r in records) / n if n else None,
        "latency_ms": {"p50": _pct(lat, 0.5), "p90": _pct(lat, 0.9), "p99": _pct(lat, 0.99)},
    }
