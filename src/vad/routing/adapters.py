"""Model adapters: primary/secondary detectors, relevance scorers, and the VLM verifier.

No detector family is assumed. Anything that returns boxes for a set of
class names can be wrapped by implementing :class:`Detector`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
from PIL import Image
from pydantic import ValidationError

from .schemas import Detection, ImageRef, Source, VerificationRequest, VerificationResult, stable_id


# --------------------------------------------------------------------------- detectors


@dataclass(frozen=True)
class RawDetection:
    class_id: str
    box_xyxy: tuple[float, float, float, float]
    confidence: float | None = None


@dataclass
class DetectorResult:
    detections: list[RawDetection]
    gpu_ms: float | None = None


@runtime_checkable
class Detector(Protocol):
    model_version: str
    supports_crops: bool  # True if detect() works on an arbitrary crop (needed for slicing)

    def detect(self, image: Image.Image, image_id: str, class_ids: Sequence[str]) -> DetectorResult: ...


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


# --------------------------------------------------------------------------- relevance


class RelevanceScorer(Protocol):
    """Per-class raw text-image similarity (e.g. a CLIP-like model with positive/negative prompts)."""

    model_version: str

    def score(self, image: Image.Image, image_id: str, prompts: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]) -> dict[str, float]: ...


class RelevanceCalibrator(Protocol):
    version: str

    def calibrate(self, raw: Mapping[str, float]) -> tuple[dict[str, float], float]:
        """Return per-class probabilities and P(any target present), modelling class correlation."""
        ...


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


# --------------------------------------------------------------------------- VLM


class VLMVerifier(Protocol):
    model_version: str

    def verify(self, image: Image.Image, request: VerificationRequest, class_desc: str) -> VerificationResult: ...


def uncertain(request: VerificationRequest, reason_code: str, model_version: str, note: str = "") -> VerificationResult:
    return VerificationResult(
        request_id=request.id, class_id=request.class_id, verdict="uncertain",
        candidate_ids=request.candidate_ids, region_xyxy=request.region_xyxy,
        reason_code=reason_code, model_note=note[:200], model_version=model_version,
    )


_VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "class_id": {"type": "string"},
        "verdict": {"type": "string", "enum": ["present", "absent", "uncertain"]},
        "reason": {"type": "string"},
    },
    "required": ["class_id", "verdict", "reason"],
    "additionalProperties": False,
}


@dataclass
class LMStudioVLM:
    """Self-hosted VLM via LM Studio's OpenAI-compatible endpoint.

    Exactly one HTTP attempt per request, hard timeout, no retries. Any
    transport error, malformed or contradictory answer becomes ``uncertain``.
    """

    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen3-vl-4b-instruct"
    timeout_s: float = 60
    max_tokens: int = 200
    crop_padding: float = 0.25
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    @property
    def model_version(self) -> str:
        return f"lmstudio:{self.model}"

    def _image_payload(self, image: Image.Image, region) -> str:
        if region is not None:
            x1, y1, x2, y2 = region
            px, py = (x2 - x1) * self.crop_padding, (y2 - y1) * self.crop_padding
            image = image.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                                min(image.width, int(x2 + px + 1)), min(image.height, int(y2 + py + 1))))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _prompt(self, request: VerificationRequest, class_desc: str) -> str:
        if request.purpose == "confirm_candidate":
            q = f"Does this image crop contain a '{class_desc}'?"
        else:
            q = f"Is there any '{class_desc}' visible anywhere in this image?"
        return (f"{q} Answer only with JSON: class_id must be exactly \"{request.class_id}\", "
                "verdict is one of present|absent|uncertain, reason is a short phrase. "
                "Say uncertain if you cannot tell.")

    def verify(self, image: Image.Image, request: VerificationRequest, class_desc: str) -> VerificationResult:
        body = {
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self._prompt(request, class_desc)},
                {"type": "image_url", "image_url": {"url": self._image_payload(image, request.region_xyxy)}},
            ]}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "verification", "strict": True, "schema": _VLM_SCHEMA}},
        }
        try:
            with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                resp = client.post(f"{self.base_url}/chat/completions", json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException as e:
            return uncertain(request, "vlm_timeout", self.model_version, str(e))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            return uncertain(request, "vlm_unavailable", self.model_version, f"{type(e).__name__}: {e}")
        return parse_vlm_answer(content, request, self.model_version)


def parse_vlm_answer(content: object, request: VerificationRequest, model_version: str) -> VerificationResult:
    try:
        ans = json.loads(content) if isinstance(content, str) else None
        if not isinstance(ans, dict) or set(ans) - {"class_id", "verdict", "reason"}:
            return uncertain(request, "vlm_malformed", model_version, str(content)[:200])
        verdict, cls = ans.get("verdict"), ans.get("class_id")
        if verdict not in ("present", "absent", "uncertain"):
            return uncertain(request, "vlm_malformed", model_version, str(content)[:200])
        if cls != request.class_id:
            return uncertain(request, "vlm_contradictory", model_version, f"answered for class {cls!r}")
        return VerificationResult(
            request_id=request.id, class_id=request.class_id, verdict=verdict,
            candidate_ids=request.candidate_ids, region_xyxy=request.region_xyxy,
            reason_code=f"vlm_{verdict}", model_note=str(ans.get("reason", ""))[:200],
            model_version=model_version,
        )
    except (json.JSONDecodeError, ValidationError) as e:
        return uncertain(request, "vlm_malformed", model_version, f"{type(e).__name__}")
