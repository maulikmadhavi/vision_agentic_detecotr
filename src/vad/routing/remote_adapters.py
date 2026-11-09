"""HTTP clients for separately served models. Increment 1: :class:`RemoteDetector`.

Implements the existing :class:`~vad.routing.adapters.Detector` protocol, so the
pipeline, slicing and policy are unchanged. Any failure raises
:class:`RemoteDetectorError`; the pipeline turns that into a per-image error record
(primary) or a ``*_failed`` review reason (secondary / SAHI), never "no boxes".
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import httpx
from PIL import Image
from pydantic import ValidationError

from .adapters import DetectorResult, RawDetection
from .schemas import TaskSpec, stable_id
from .service_contracts import DetectClass, DetectRequest, DetectResponse


class RemoteDetectorError(RuntimeError):
    """The remote detector did not return a verified result for this image or crop."""


WEMM_TEXT_SCALE = "wemm_text_cosine_v1"


class RemoteScorerError(RuntimeError):
    """The remote relevance scorer did not return a verified score; never treated as a score of 0."""


@dataclass
class RemoteDetector:
    """One POST per image or crop, hard timeout, one attempt, strict response checks.

    ``model_id`` / ``model_revision`` / ``weights_sha256`` are what we expect the server
    to run; every response is checked against them. They are fixed up front because the
    run key (``RunInfo``) is built before any call. With ``weights_sha256=None`` the
    weights are unverified and ``artifact_digest`` is ``None`` (board assumption 6).
    """

    base_url: str
    task: TaskSpec
    model_id: str
    model_revision: str
    weights_sha256: str | None = None
    timeout_s: float = 60
    max_image_pixels: int = 4096 * 4096
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    supports_crops: ClassVar[bool] = True

    @property
    def model_version(self) -> str:
        return f"remote:{self.model_id}@{self.model_revision}"

    @property
    def artifact_digest(self) -> str | None:
        return self.weights_sha256

    def detect(self, image: Image.Image, image_id: str, class_ids: Sequence[str]) -> DetectorResult:
        if image.width * image.height > self.max_image_pixels:
            raise RemoteDetectorError(f"image {image.width}x{image.height} exceeds max_image_pixels "
                                      f"{self.max_image_pixels}; not sent")
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "PNG")
        png = buf.getvalue()
        sha = hashlib.sha256(png).hexdigest()
        prompts = {c.class_id: c.positive_prompts for c in self.task.classes}
        if unknown := sorted(set(class_ids) - set(prompts)):
            raise RemoteDetectorError(f"class_ids not in task: {unknown}")
        req = DetectRequest(
            request_id=stable_id("dreq", [image_id, sha, self.model_version]),
            image_png_b64=base64.b64encode(png).decode(), image_sha256=sha,
            width=image.width, height=image.height,
            classes=tuple(DetectClass(class_id=c, prompts=prompts[c]) for c in class_ids),
        )
        try:
            with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                resp = client.post(f"{self.base_url}/detect", json=req.model_dump(mode="json"))
            resp.raise_for_status()
            out = DetectResponse.model_validate_json(resp.content)
        except (httpx.HTTPError, ValidationError, ValueError) as e:
            raise RemoteDetectorError(f"{type(e).__name__}: {e}"[:300]) from e

        problems = [name for name, bad in (
            ("status_failed", out.status != "ok"),
            ("request_id_mismatch", out.request_id != req.request_id),
            ("size_mismatch", (out.width, out.height) != (req.width, req.height)),
            ("model_mismatch", (out.model_id, out.model_revision) != (self.model_id, self.model_revision)),
            ("weights_mismatch", self.weights_sha256 is not None and out.weights_sha256 != self.weights_sha256),
        ) if bad]
        if problems:
            raise RemoteDetectorError(f"{','.join(problems)}: {out.error or ''}"[:300])
        return DetectorResult([RawDetection(d.class_id, d.box_xyxy, d.score) for d in out.detections], out.gpu_ms)


@dataclass
class RemoteRelevanceScorer:
    """Text-image relevance from a WeMM-Embedding MCP server (``compute_similarity``, cosine).

    Implements the existing ``RelevanceScorer`` protocol. One JSON-RPC POST per image
    with all classes' positive prompts; per-class raw score = max cosine over that
    class's prompts (scale ``wemm_text_cosine_v1``, not a probability, not comparable
    with CLIP). Images are downscaled so the longest side is at most ``max_side``;
    that setting is part of ``model_version``. Retries once on a network error only.
    ``isError``, JSON-RPC errors, malformed results and ``meta`` mismatches raise
    :class:`RemoteScorerError`; they never become a score. Without a server-reported
    ``build`` the version is ``@unverified`` (board assumption 6).
    """

    url: str
    model: str = "tencent/WeMM-Embedding-4B"
    dim: int = 2560
    build: str | None = None
    max_side: int = 448
    timeout_s: float = 120
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    score_scale: ClassVar[str] = WEMM_TEXT_SCALE
    artifact_digest: ClassVar[None] = None

    @property
    def model_version(self) -> str:
        return f"wemm:{self.model}/d{self.dim}@{self.build or 'unverified'}+max{self.max_side}"

    def _data_uri(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        if max(img.size) > self.max_side:
            scale = self.max_side / max(img.size)
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _call(self, tool: str, arguments: dict) -> dict:
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
        headers = {"Accept": "application/json, text/event-stream"}
        resp = None
        for attempt in (1, 2):  # one retry, only for network errors
            try:
                with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                    resp = client.post(self.url, json=body, headers=headers)
                break
            except httpx.TransportError as e:
                if attempt == 2:
                    raise RemoteScorerError(f"network: {type(e).__name__}: {e}"[:300]) from e
        try:
            resp.raise_for_status()
            text = resp.text
            if "text/event-stream" in resp.headers.get("content-type", ""):
                text = [ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")][-1]
            msg = json.loads(text)
            if "error" in msg:
                raise RemoteScorerError(f"jsonrpc error: {msg['error']}"[:300])
            result = msg["result"]
            if result.get("isError"):
                raise RemoteScorerError(f"isError: {result.get('content')}"[:300])
            payload = json.loads(result["content"][0]["text"])
            meta = payload["meta"]
        except RemoteScorerError:
            raise
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            raise RemoteScorerError(f"{type(e).__name__}: {e}"[:300]) from e
        expected = {"backend": "wemm", "model": self.model, "dim": self.dim}
        if self.build is not None:
            expected["build"] = self.build
        if bad := {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}:
            raise RemoteScorerError(f"meta mismatch: {bad}")
        return payload

    def score(self, image: Image.Image, image_id: str,
              prompts: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]) -> dict[str, float]:
        texts, owners = [], []
        for class_id, (positives, _negatives) in prompts.items():
            texts += positives
            owners += [class_id] * len(positives)
        payload = self._call("compute_similarity", {"left": [self._data_uri(image)], "right": texts,
                                                    "mode": "matrix", "metric": "cosine", "top_k": len(texts)})
        try:
            by_index = {int(m["index"]): float(m["score"]) for m in payload["results"][0]["matches"]}
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise RemoteScorerError(f"malformed matches: {type(e).__name__}") from e
        if set(by_index) != set(range(len(texts))) or not all(math.isfinite(v) for v in by_index.values()):
            raise RemoteScorerError(f"expected finite scores for indices 0..{len(texts) - 1}")
        out: dict[str, float] = {}
        for i, class_id in enumerate(owners):
            out[class_id] = max(out.get(class_id, -math.inf), by_index[i])
        return out
