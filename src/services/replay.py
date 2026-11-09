"""Replay adapters: detections and relevance scores precomputed elsewhere, read from JSONL files."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from routing.interfaces import DetectorResult, RawDetection


class MissingPrediction(LookupError):
    """No precomputed row for an image: the image was never processed (not a zero-box result)."""


def _read_jsonl_rows(path: str | Path, required: str) -> tuple[dict[str, dict], str]:
    """Parse JSONL keyed by ``image_id``; returns (rows, sha256 of file bytes).

    Each row must carry ``required`` explicitly and image ids must be unique, so
    an omitted image can never be confused with an explicit empty result.
    """
    data = Path(path).read_bytes()
    rows: dict[str, dict] = {}
    for n, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "image_id" not in row or required not in row:
            raise ValueError(f"{path}:{n}: row needs 'image_id' and '{required}' (use [] / {{}} for none)")
        if row["image_id"] in rows:
            raise ValueError(f"{path}:{n}: duplicate image_id {row['image_id']!r}")
        rows[row["image_id"]] = row
    return rows, hashlib.sha256(data).hexdigest()


class PrecomputedDetector:
    """Reads predictions produced elsewhere (JSONL: {"image_id", "detections": [{class_id, box_xyxy, confidence}]}).

    Lets the router run in shadow mode over an existing detector's outputs.
    Every image needs an explicit row (``"detections": []`` is a valid zero-box
    result); a missing row raises :class:`MissingPrediction` so the image goes to
    the error queue. ``model_version`` is a caller-supplied identifier; the file's
    content digest (``artifact_digest``) is bound into the run key separately.
    Cannot run on crops, so SAHI is unavailable with this adapter.
    """

    supports_crops = False

    def __init__(self, path: str | Path, model_version: str):
        self.model_version = model_version
        rows, self.artifact_digest = _read_jsonl_rows(path, "detections")
        self._preds: dict[str, list[RawDetection]] = {
            image_id: [RawDetection(d["class_id"], tuple(d["box_xyxy"]), d.get("confidence")) for d in row["detections"]]
            for image_id, row in rows.items()
        }

    def detect(self, image: Image.Image, image_id: str, class_ids: Sequence[str]) -> DetectorResult:
        if image_id not in self._preds:
            raise MissingPrediction(f"no precomputed prediction row for image_id {image_id!r}")
        return DetectorResult(list(self._preds[image_id]))


class PrecomputedRelevance:
    """Reads raw similarities from JSONL: {"image_id", "scores": {class_id: float}}.

    A missing row yields no scores, which Stage A records as relevance
    ``unavailable`` (never as low relevance), so the image is routed to review.
    """

    def __init__(self, path: str | Path, model_version: str):
        self.model_version = model_version
        rows, self.artifact_digest = _read_jsonl_rows(path, "scores")
        self._scores = {i: {k: float(v) for k, v in r["scores"].items()} for i, r in rows.items()}

    def score(self, image, image_id, prompts):
        return dict(self._scores.get(image_id, {}))
