"""Sample tooling for the human-checked trial set (reviewer work order A2; board T-11).

Stage 1: group exact and near-duplicate images so a split never separates them, and load the
exclusions file that must be recorded before any split.

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
from .schemas import canonical_json, stable_id

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
