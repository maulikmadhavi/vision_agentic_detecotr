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
