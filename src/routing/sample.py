"""Sample tooling for the human-checked trial set (reviewer work order A2; board T-11).

Stage 1: group exact and near-duplicate images so a split never separates them, and load the
exclusions file that must be recorded before any split.
Stage 2: a seeded, group-wise split into calibration / development / frozen_eval, stratified by
positive and negative groups, written as a versioned manifest that pins its input digests.

- Exact copies share a SHA-256 of their bytes.
- Near-duplicates (resized, recompressed) have 64-bit difference hashes (dHash) at Hamming
  distance <= ``max_hamming`` (default 6). Pairs a little further apart (up to
  ``max_hamming + REVIEW_MARGIN``) are listed for a human to check but never merged.
- Images with the same manifest ``group_id`` (same source page or photographer) are merged.

Only synthetic fixtures are used in tests; the collected trial images are not read here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .image_inputs import ManifestEntry, load_image
from .metrics import load_ground_truth
from .schemas import TaskSpec, canonical_json, stable_id

REVIEW_MARGIN = 4
EXCLUSION_REASONS = ("illustration", "collage", "screenshot", "duplicate", "license_unclear", "other")


class SampleError(ValueError):
    """Invalid sample-tooling input (exclusions, manifests, splits)."""


@dataclass(frozen=True)
class Fingerprint:
    sha256: str
    dhash: int | None  # None when the image cannot be decoded


@dataclass(frozen=True)
class Grouping:
    groups: dict[str, str]  # image id -> group id
    pairs: list[dict]       # exact / near-duplicate pairs for human confirmation (merged or not)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _dhash(image: Image.Image) -> int:
    """64-bit difference hash: is each pixel brighter than its left neighbour, on a 9x8 grayscale thumbnail."""
    small = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    return int("".join("1" if bit else "0" for bit in (px[:, 1:] > px[:, :-1]).flatten()), 2)


def fingerprint(path: str | Path) -> Fingerprint:
    data = Path(path).read_bytes()
    with Image.open(path) as image:
        return Fingerprint(sha256=hashlib.sha256(data).hexdigest(), dhash=_dhash(image))


def group_images(fingerprints: Mapping[str, Fingerprint], sources: Mapping[str, str | None],
                 max_hamming: int = 6) -> Grouping:
    """Union exact copies, near-duplicates (<= max_hamming) and same-source images into groups."""
    ids = sorted(fingerprints)
    parent = {i: i for i in ids}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs = []
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            fa, fb = fingerprints[a], fingerprints[b]
            distance = hamming(fa.dhash, fb.dhash) if fa.dhash is not None and fb.dhash is not None else None
            if fa.sha256 == fb.sha256:
                pairs.append({"a": a, "b": b, "reason": "exact", "distance": distance, "merged": True})
                union(a, b)
            elif distance is not None and distance <= max_hamming + REVIEW_MARGIN:
                merged = distance <= max_hamming
                pairs.append({"a": a, "b": b, "reason": "near_duplicate", "distance": distance, "merged": merged})
                if merged:
                    union(a, b)
    by_source: dict[str, str] = {}
    for i in ids:
        source = sources.get(i)
        if source:
            if source in by_source:
                union(by_source[source], i)
            else:
                by_source[source] = i
    members: dict[str, list[str]] = {}
    for i in ids:
        members.setdefault(find(i), []).append(i)
    group_of = {i: stable_id("grp", sorted(m)) for m in members.values() for i in m}
    return Grouping(groups=group_of, pairs=pairs)


def group_manifest(entries: Sequence[ManifestEntry], max_hamming: int = 6) -> dict:
    """Fingerprint and group manifest images; unreadable images get their own group and are listed."""
    fingerprints: dict[str, Fingerprint] = {}
    sources: dict[str, str | None] = {}
    unreadable: list[str] = []
    for entry in entries:
        try:
            _, ref = load_image(entry)
            image_id, fp = ref.id, fingerprint(entry.uri)
        except Exception:  # unreadable or corrupt: keep it visible, never drop it
            data = Path(entry.uri).read_bytes() if Path(entry.uri).exists() else b""
            sha = hashlib.sha256(data).hexdigest()
            image_id, fp = entry.image_id or f"sha256-{sha[:16]}", Fingerprint(sha256=sha, dhash=None)
            unreadable.append(image_id)
        if image_id in fingerprints:
            raise SampleError(f"duplicate image id {image_id!r} in the manifest")
        fingerprints[image_id], sources[image_id] = fp, entry.group_id
    grouping = group_images(fingerprints, sources, max_hamming)
    return {
        "version": "groups-v1", "max_hamming": max_hamming, "review_margin": REVIEW_MARGIN,
        "images": {i: {"group_id": grouping.groups[i], "sha256": fp.sha256,
                       "dhash": None if fp.dhash is None else f"{fp.dhash:016x}", "source": sources[i]}
                   for i, fp in sorted(fingerprints.items())},
        "pairs": grouping.pairs, "unreadable": sorted(unreadable),
    }


# --------------------------------------------------------------------------- exclusions


@dataclass(frozen=True)
class Exclusion:
    image_id: str
    reason: str
    note: str = ""
    out_of_domain_set: bool = False  # keep as a separately labelled out-of-domain set


def load_exclusions(doc: Mapping, image_ids: Collection[str]) -> tuple[Exclusion, ...]:
    """Validate an exclusions file: ``{"vad_exclusions": 1, "recorded_by": ..., "exclusions": [...]}``.

    It must be written before the split; the split manifest pins its digest.
    """
    if doc.get("vad_exclusions") != 1:
        raise SampleError('the exclusions file must declare "vad_exclusions": 1')
    if not doc.get("recorded_by"):
        raise SampleError("the exclusions file needs recorded_by (the person who recorded them)")
    known, seen, out = set(image_ids), set(), []
    for row in doc["exclusions"]:
        image_id, reason = row["image_id"], row["reason"]
        if image_id not in known:
            raise SampleError(f"excluded image {image_id!r} is not in the manifest")
        if image_id in seen:
            raise SampleError(f"image {image_id!r} is excluded twice")
        if reason not in EXCLUSION_REASONS:
            raise SampleError(f"unknown exclusion reason {reason!r}; use one of {list(EXCLUSION_REASONS)}")
        if reason == "other" and not row.get("note"):
            raise SampleError(f"image {image_id!r}: reason 'other' needs a note")
        seen.add(image_id)
        out.append(Exclusion(image_id, reason, row.get("note", ""), bool(row.get("out_of_domain_set", False))))
    return tuple(out)


def exclusions_digest(doc: Mapping) -> str:
    """Content digest (independent of formatting and key order) to pin in the split manifest."""
    return hashlib.sha256(canonical_json(doc).encode()).hexdigest()


# --------------------------------------------------------------------------- split

FROZEN_FRACTION = 0.3       # of each stratum's groups
CALIBRATION_FRACTION = 0.5  # of each stratum's tuning groups; the rest is development
SPLIT_NAMES = ("calibration", "development", "frozen_eval")


def _digest(doc: Mapping) -> str:
    return hashlib.sha256(canonical_json(doc).encode()).hexdigest()


def _round(x: float) -> int:
    return int(x + 0.5)  # half up, not banker's rounding


def frozen_label_subset(label_doc: Mapping, image_ids: Collection[str]) -> dict:
    """The part of the label file that the frozen lock pins: categories, frozen images, their annotations."""
    wanted = set(image_ids)
    images = sorted((img for img in label_doc["images"] if img["vad_image_id"] in wanted),
                    key=lambda img: img["vad_image_id"])
    keys = {img["id"] for img in images}
    annotations = sorted((a for a in label_doc["annotations"] if a["image_id"] in keys), key=lambda a: a["id"])
    return {"categories": label_doc["categories"], "images": images, "annotations": annotations}


def make_split(groups_doc: Mapping, exclusions_doc: Mapping, label_doc: Mapping, task: TaskSpec,
               seed: int) -> tuple[dict, dict]:
    """Seeded group-wise split; returns the manifest and the frozen label subset to lock.

    Within each stratum (groups with any labelled object vs none), groups are ordered by a
    seeded hash of their id: about 30% go to frozen_eval, the rest is halved into calibration
    and development. A non-empty stratum needs at least 3 groups (one per split), otherwise the
    split is refused rather than padded; an empty stratum is stated in ``warnings``. Excluded
    images stay in the manifest as ``excluded``.
    """
    if groups_doc.get("version") != "groups-v1":
        raise SampleError("--groups must be a groups-v1 file from `vad-route sample group`")
    image_group = {i: v["group_id"] for i, v in groups_doc["images"].items()}
    exclusions = {e.image_id: e for e in load_exclusions(exclusions_doc, image_group)}
    if missing := sorted(set(groups_doc["unreadable"]) - set(exclusions)):
        raise SampleError(f"unreadable images must be excluded before the split: {missing[:5]}")
    ground_truth = load_ground_truth(label_doc, task)
    if unknown := sorted(set(ground_truth) - set(image_group)):
        raise SampleError(f"labelled images that are not in the groups file: {unknown[:5]}")
    included = sorted(i for i in image_group if i not in exclusions)
    if unlabelled := [i for i in included if i not in ground_truth]:
        raise SampleError(f"images that are neither labelled nor excluded: {unlabelled[:5]}")

    members: dict[str, list[str]] = {}
    for image_id in sorted(image_group):
        members.setdefault(image_group[image_id], []).append(image_id)
    stratum: dict[str, str | None] = {}
    for group_id, ids in members.items():
        kept = [i for i in ids if i in ground_truth and i not in exclusions]
        stratum[group_id] = None if not kept else (
            "positive" if any(not b.ignore for i in kept for b in ground_truth[i].boxes) else "negative")
    split = {g: "excluded" for g, s in stratum.items() if s is None}
    warnings = []
    for name in ("positive", "negative"):
        ordered = sorted((g for g, s in stratum.items() if s == name),
                         key=lambda g: hashlib.sha256(f"{seed}|{g}".encode()).hexdigest())
        if not ordered:
            warnings.append(f"no {name} groups: no split contains {name} images")
            continue
        if len(ordered) < 3:
            raise SampleError(f"the {name} stratum has {len(ordered)} groups; it needs at least 3 (one each for "
                              "calibration, development and frozen_eval). Add groups or change the sample; "
                              "the split does not pad")
        n_frozen = max(1, _round(FROZEN_FRACTION * len(ordered)))
        n_calibration = max(1, _round(CALIBRATION_FRACTION * (len(ordered) - n_frozen)))
        for k, group_id in enumerate(ordered):
            split[group_id] = ("frozen_eval" if k < n_frozen else
                               "calibration" if k < n_frozen + n_calibration else "development")

    images = {}
    for image_id in sorted(image_group):
        group_id, excl = image_group[image_id], exclusions.get(image_id)
        images[image_id] = {
            "group_id": group_id,
            "split": "excluded" if excl else split[group_id],
            "stratum": None if excl else stratum[group_id],
            "exclusion": {"reason": excl.reason, "out_of_domain_set": excl.out_of_domain_set} if excl else None,
        }
    counts = {name: {"groups": 0, "images": 0, "positive_groups": 0, "negative_groups": 0}
              for name in (*SPLIT_NAMES, "excluded")}
    for group_id, name in split.items():
        counts[name]["groups"] += 1
        if stratum[group_id]:
            counts[name][f"{stratum[group_id]}_groups"] += 1
    for img in images.values():
        counts[img["split"]]["images"] += 1
    manifest = {
        "version": "split-v1", "seed": seed,
        "fractions": {"frozen_eval": FROZEN_FRACTION, "calibration_of_tuning": CALIBRATION_FRACTION},
        "inputs": {"groups_sha256": _digest(groups_doc), "exclusions_sha256": exclusions_digest(exclusions_doc),
                   "labels_sha256": _digest(label_doc)},
        "groups": {g: {"split": split[g], "stratum": stratum[g], "images": members[g]} for g in sorted(members)},
        "images": images, "counts": counts, "warnings": warnings,
    }
    frozen_ids = [i for i, img in images.items() if img["split"] == "frozen_eval"]
    return manifest, frozen_label_subset(label_doc, frozen_ids)
