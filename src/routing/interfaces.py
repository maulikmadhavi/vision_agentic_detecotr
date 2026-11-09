"""Integration protocols and result types: detectors, relevance scorers, calibrators, VLM verifiers.

No detector family is assumed. Anything that returns boxes for a set of class names
can be wrapped by implementing :class:`Detector`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

from .schemas import VerificationRequest, VerificationResult


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


class RelevanceScorer(Protocol):
    """Per-class raw text-image similarity (e.g. a CLIP-like model with positive/negative prompts)."""

    model_version: str

    def score(self, image: Image.Image, image_id: str, prompts: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]) -> dict[str, float]: ...


class RelevanceCalibrator(Protocol):
    version: str

    def calibrate(self, raw: Mapping[str, float]) -> tuple[dict[str, float], float]:
        """Return per-class probabilities and P(any target present), modelling class correlation."""
        ...


class VLMVerifier(Protocol):
    model_version: str

    def verify(self, image: Image.Image, request: VerificationRequest, class_desc: str) -> VerificationResult: ...
