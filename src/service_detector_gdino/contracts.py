"""Versioned wire contracts for separately served models (increment 1: detector only).

Servers run outside this package (own environment or container) and speak these
JSON shapes over HTTP. Boxes are xyxy pixels relative to the *submitted* image or
crop; the router maps crops to original-image pixels exactly once.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SERVICE_SCHEMA_VERSION = "svc-1.0"


class Strict(BaseModel):
    """Same config as ``routing.schemas.Strict`` (a test checks they match); kept here so the server needs no routing import."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DetectClass(Strict):
    class_id: str = Field(min_length=1)
    prompts: tuple[str, ...] = Field(min_length=1)


class DetectRequest(Strict):
    schema_version: Literal["svc-1.0"] = SERVICE_SCHEMA_VERSION
    request_id: str
    image_png_b64: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    classes: tuple[DetectClass, ...] = Field(min_length=1)


class ServedBox(Strict):
    class_id: str
    box_xyxy: tuple[float, float, float, float]
    score: float | None = None  # raw model score; range/finiteness checked by adapters.to_detections


class DetectResponse(Strict):
    """``status="ok"`` with no detections is a valid empty result; ``"failed"`` is never one."""

    schema_version: Literal["svc-1.0"]
    request_id: str
    status: Literal["ok", "failed"]
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    coord_system: Literal["xyxy_pixel_submitted"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: tuple[ServedBox, ...] = ()
    gpu_ms: float | None = Field(default=None, ge=0)  # server-measured device time only (D-6)
    error: str | None = None
