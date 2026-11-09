"""Typed contracts for the adaptive routing layer.

All boxes are canonical ``xyxy`` pixel coordinates in the *original* image.
Raw model outputs (similarity, confidence) are kept separate from calibrated
probabilities and from heuristic difficulty scores.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
COORD_SYSTEM = "xyxy_pixel_original"

Source = Literal["primary", "secondary", "sahi", "human"]
Action = Literal["standard", "secondary", "sahi", "vlm_verify", "human_review", "audit_sample"]
Verdict = Literal["present", "absent", "uncertain"]
CandidateStatus = Literal["accepted", "candidate", "needs_review", "rejected"]
CompletenessStatus = Literal["complete", "needs_review", "unresolved"]


_MODEL_TOKENS = ("vlm", "gpt", "qwen", "llama", "claude", "gemini", "mistral", "dino", "wemm", "clip", "model",
                 "detector", "openai", "lmstudio")


def check_reviewer_handle(handle: object, role: str) -> str:
    """A reviewer is a free-text human handle. Model-like ids are refused: a guard against pasted
    model ids, not authentication (a real name can trip it; the message says why)."""
    text = handle.strip() if isinstance(handle, str) else ""
    if not text:
        raise ValueError(f"{role} reviewer handle is empty; use a person's handle")
    guard = "this is a guard against pasted model ids; use a person's handle"
    if bad := next((ch for ch in ":@/" if ch in text), None):
        raise ValueError(f"{role} reviewer handle {handle!r} contains {bad!r}, as model ids do; {guard}")
    if token := next((t for t in _MODEL_TOKENS if t in text.lower()), None):
        raise ValueError(f"{role} reviewer handle {handle!r} contains {token!r}; {guard} "
                         f"(pick a variant without {token!r})")
    return text


def _valid_xyxy(b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError(f"invalid xyxy box {b}: need 0 <= x1 < x2 and 0 <= y1 < y2")
    return b


def canonical_json(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_id(prefix: str, payload: Any, n: int = 16) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:n]}"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


# --------------------------------------------------------------------------- task / image


class ClassSpec(Strict):
    """One project-defined class. ``class_id`` is free text; no public taxonomy is assumed."""

    class_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.\-]+$")
    display_name: str = Field(min_length=1)
    positive_prompts: tuple[str, ...] = Field(min_length=1)
    negative_prompts: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


class TaskSpec(Strict):
    task_id: str = Field(min_length=1)
    prompt_set_version: str = Field(min_length=1)
    classes: tuple[ClassSpec, ...] = Field(min_length=1)

    @field_validator("classes")
    @classmethod
    def _unique(cls, v: tuple[ClassSpec, ...]) -> tuple[ClassSpec, ...]:
        ids = [c.class_id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate class_id in registry: {ids}")
        return v

    @property
    def class_ids(self) -> tuple[str, ...]:
        return tuple(c.class_id for c in self.classes)


class ImageRef(Strict):
    id: str = Field(min_length=1)
    uri: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_id: str | None = None


# --------------------------------------------------------------------------- detections


class Detection(Strict):
    id: str
    class_id: str
    box_xyxy: tuple[float, float, float, float]
    raw_confidence: float | None = Field(default=None, ge=0, le=1)
    source: Source
    model_version: str = Field(min_length=1)
    coord_system: Literal["xyxy_pixel_original"] = COORD_SYSTEM
    slice_origin: tuple[int, int] | None = None
    parent_ids: tuple[str, ...] = ()

    @field_validator("box_xyxy")
    @classmethod
    def _box_order(cls, b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return _valid_xyxy(b)

    def check_in(self, image: ImageRef, class_ids: tuple[str, ...]) -> None:
        if self.class_id not in class_ids:
            raise ValueError(f"unknown class_id {self.class_id!r}; registry={class_ids}")
        x1, y1, x2, y2 = self.box_xyxy
        if x2 > image.width or y2 > image.height:
            raise ValueError(f"box {self.box_xyxy} exceeds image {image.width}x{image.height}")


# --------------------------------------------------------------------------- stage A


class ClassRelevance(Strict):
    raw_similarity: float | None = None
    relevance_probability: float | None = Field(default=None, ge=0, le=1)


class Quality(Strict):
    blur: float = Field(ge=0, le=1, description="0 sharp .. 1 very blurry (heuristic, uncalibrated)")
    exposure: float = Field(ge=0, le=1, description="fraction of clipped pixels")
    min_side_px: int = Field(gt=0)
    low_resolution: bool


class StageA(Strict):
    per_class: dict[str, ClassRelevance]
    image_relevance_probability: float | None = Field(default=None, ge=0, le=1)
    relevance_status: Literal["calibrated", "uncalibrated", "unavailable"]
    relevance_band: Literal["high", "medium", "low", "unknown"]
    quality: Quality
    domain_distance: float | None = None
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> StageA:
        if self.relevance_status == "calibrated" and self.image_relevance_probability is None:
            raise ValueError("calibrated relevance requires image_relevance_probability")
        if self.relevance_status != "calibrated" and self.image_relevance_probability is not None:
            raise ValueError("image_relevance_probability is only set by a calibrator")
        if self.relevance_status == "unavailable" and self.relevance_band != "unknown":
            raise ValueError("unavailable relevance must have band 'unknown'")
        return self


# --------------------------------------------------------------------------- stage B


class Difficulty(Strict):
    """Miss/mislocalization risk *if a target is present*. Never inferred from zero boxes."""

    value: float | None = Field(default=None, ge=0, le=1)
    status: Literal["calibrated", "uncalibrated", "unknown"]
    bucket: Literal["easy", "medium", "hard", "unknown"]
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Difficulty:
        unknown = self.status == "unknown"
        if unknown != (self.value is None) or unknown != (self.bucket == "unknown"):
            raise ValueError("difficulty status/value/bucket must agree on 'unknown'")
        return self


class BoxFeatures(Strict):
    num_boxes: int = Field(ge=0)
    per_class_counts: dict[str, int]
    min_box_area_fraction: float | None = Field(default=None, ge=0, le=1)
    crowding: float = Field(ge=0, le=1, description="fraction of boxes overlapping another above crowding_iou")
    class_ambiguity: float = Field(ge=0, le=1, description="fraction of boxes overlapping a different-class box")
    mean_confidence: float | None = Field(default=None, ge=0, le=1)
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    high_relevance_zero_boxes: bool


class Disagreement(Strict):
    status: Literal["computed", "unavailable"]
    matched: int = 0
    primary_only: int = 0
    secondary_only: int = 0
    cross_class: int = 0
    mislocalized: int = 0
    mean_matched_iou: float | None = None
    primary_only_ids: tuple[str, ...] = ()
    secondary_only_ids: tuple[str, ...] = ()


class StageB(Strict):
    difficulty_pre_secondary: Difficulty
    difficulty_final: Difficulty
    features: BoxFeatures
    disagreement: Disagreement


# --------------------------------------------------------------------------- routing


class Budget(Strict):
    slices: int = Field(default=0, ge=0)
    vlm_calls: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0, ge=0)
    gpu_ms: float | None = Field(default=None, ge=0)


class RouteDecision(Strict):
    actions: tuple[Action, ...]
    priority: int = Field(ge=0)
    reason_codes: tuple[str, ...]
    policy_version: str
    expected_budget: Budget


# --------------------------------------------------------------------------- verification


class VerificationRequest(Strict):
    id: str
    image_id: str
    class_id: str
    candidate_ids: tuple[str, ...] = ()
    region_xyxy: tuple[float, float, float, float] | None = None
    purpose: Literal["confirm_candidate", "suspected_missing_class"]


class VerificationResult(Strict):
    """Constrained VLM output. Has no box field by design: a VLM cannot author boxes."""

    request_id: str
    class_id: str
    verdict: Verdict
    candidate_ids: tuple[str, ...] = ()
    region_xyxy: tuple[float, float, float, float] | None = None
    reason_code: str
    model_note: str = Field(default="", max_length=200)
    model_version: str


# --------------------------------------------------------------------------- outputs


class FinalCandidate(Strict):
    id: str
    class_id: str
    box_xyxy: tuple[float, float, float, float]
    status: CandidateStatus
    source_ids: tuple[str, ...] = Field(min_length=1)
    canonical_source_id: str
    flags: tuple[str, ...] = ()
    verification: VerificationResult | None = None


class AnnotationAudit(Strict):
    status: CompletenessStatus
    unmatched_candidate_ids: tuple[str, ...] = ()
    suspected_missing_classes: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    established_by: Literal["human", "calibrated_audit_policy", "none"] = "none"
    reviewer_outcome: str | None = None

    @model_validator(mode="after")
    def _complete_needs_authority(self) -> AnnotationAudit:
        if self.status == "complete" and self.established_by == "none":
            raise ValueError("'complete' must be established by a human or a calibrated audit policy")
        return self


class RunInfo(Strict):
    config_hash: str = Field(min_length=8)
    policy_version: str
    primary_model: str = Field(min_length=1)
    secondary_model: str | None = None
    relevance_scorer: str | None = None
    relevance_calibrator: str | None = None
    difficulty_calibrator: str | None = None
    vlm_model: str | None = None
    # Content digests of file-backed artifacts (e.g. precomputed predictions). Model
    # identifiers above are caller-supplied and unverified; these bind the run key
    # to the actual inputs so changed files never reuse stale cached records.
    primary_artifact_digest: str | None = None
    secondary_artifact_digest: str | None = None
    relevance_artifact_digest: str | None = None
    # A dedicated slicing detector (one that is not the primary). None when slicing is off or
    # the primary slices, since the primary's identity already covers that. Unset values are
    # left out of the record key, so runs without a dedicated slicer keep their keys.
    sahi_model: str | None = None
    sahi_artifact_digest: str | None = None


class Trace(Strict):
    elapsed_ms: float = Field(ge=0)
    gpu_ms: float | None = Field(default=None, ge=0)
    budget_used: Budget
    events: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class RouteRecord(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    record_id: str
    image: ImageRef
    task: TaskSpec
    run: RunInfo
    stage_a: StageA
    primary_candidates: tuple[Detection, ...]
    secondary_candidates: tuple[Detection, ...] = ()
    sahi_candidates: tuple[Detection, ...] = ()
    sahi_raw_candidates: tuple[Detection, ...] = Field(default=(), description="per-tile detections before merging")
    stage_b: StageB
    route: RouteDecision
    verifications: tuple[VerificationResult, ...] = ()
    final_candidates: tuple[FinalCandidate, ...]
    completeness: AnnotationAudit
    trace: Trace

    @model_validator(mode="after")
    def _registry_and_bounds(self) -> RouteRecord:
        ids = self.task.class_ids
        for d in (*self.primary_candidates, *self.secondary_candidates, *self.sahi_candidates,
                  *self.sahi_raw_candidates):
            d.check_in(self.image, ids)
        raw_ids = {d.id for d in self.sahi_raw_candidates}
        for d in self.sahi_candidates:
            if set(d.parent_ids) - raw_ids:
                raise ValueError(f"sahi detection {d.id} cites unpersisted parents {sorted(set(d.parent_ids) - raw_ids)}")
        known = {d.id for d in (*self.primary_candidates, *self.secondary_candidates, *self.sahi_candidates)}
        for f in self.final_candidates:
            if f.class_id not in ids:
                raise ValueError(f"final candidate has unknown class_id {f.class_id!r}")
            missing = set(f.source_ids) - known
            if missing:
                raise ValueError(f"final candidate {f.id} cites unknown sources {sorted(missing)}")
        for c in (*self.stage_a.per_class, *self.completeness.suspected_missing_classes):
            if c not in ids:
                raise ValueError(f"unknown class_id {c!r}")
        return self


class ErrorRecord(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    record_id: str
    uri: str
    error: str
    config_hash: str


# --------------------------------------------------------------------------- human review (A7)


AdjudicationDecision = Literal["accept", "reject", "reclassify", "add_box", "class_absent", "complete"]
_NEEDS = {  # decision -> (candidate_id, class_id, box_xyxy) required (True) or forbidden (False)
    "accept": (True, False, False), "reject": (True, False, False), "reclassify": (True, True, False),
    "add_box": (False, True, True), "class_absent": (False, True, False), "complete": (False, False, False),
}


class Adjudication(Strict):
    """One human decision on one finding of a routed image (reviewer work order A7).

    ``accept`` / ``reject`` / ``reclassify`` act on a model candidate; ``add_box`` draws a
    missing object; ``class_absent`` confirms a suspected class is not there; ``complete``
    closes the image once its findings are decided. ``decided_at`` is supplied by the caller
    (no hidden clock). ``id`` is derived from the content, so an edited line no longer verifies.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    id: str = ""
    record_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    decision: AdjudicationDecision
    candidate_id: str | None = None
    class_id: str | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    reviewer: str
    reason: str = ""
    decided_at: str
    source: Literal["human"] = "human"

    @field_validator("reviewer")
    @classmethod
    def _human_handle(cls, v: str) -> str:
        return check_reviewer_handle(v, "adjudicating")

    @field_validator("decided_at")
    @classmethod
    def _iso_with_zone(cls, v: str) -> str:
        from datetime import datetime
        try:
            moment = datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("decided_at must be an ISO 8601 date-time") from None
        if moment.tzinfo is None:
            raise ValueError("decided_at needs a time zone, e.g. 2026-11-10T09:30:00+08:00")
        return v

    @field_validator("box_xyxy")
    @classmethod
    def _box_order(cls, b):
        return None if b is None else _valid_xyxy(b)

    @model_validator(mode="after")
    def _shape_and_id(self) -> Adjudication:
        for name, needed in zip(("candidate_id", "class_id", "box_xyxy"), _NEEDS[self.decision]):
            present = getattr(self, name) is not None
            if needed and not present:
                raise ValueError(f"decision {self.decision!r} needs {name}")
            if present and not needed:
                raise ValueError(f"decision {self.decision!r} does not take {name}")
        expected = stable_id("adj", self.model_dump(mode="json", exclude={"id"}))
        if not self.id:
            object.__setattr__(self, "id", expected)
        elif self.id != expected:
            raise ValueError("id does not match the adjudication content (edited after it was recorded?)")
        return self


class ReviewedRecord(Strict):
    """A routed image after human adjudication.

    Model suggestions stay in the base RouteRecord (``record_id``); human-drawn boxes are
    ``human_detections`` with source "human"; ``final_candidates`` and ``completeness`` are the
    reviewed state.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    record_id: str
    image: ImageRef
    task: TaskSpec
    adjudications: tuple[Adjudication, ...]
    human_detections: tuple[Detection, ...] = ()
    final_candidates: tuple[FinalCandidate, ...]
    completeness: AnnotationAudit

    @model_validator(mode="after")
    def _consistent(self) -> ReviewedRecord:
        ids = self.task.class_ids
        for d in self.human_detections:
            if d.source != "human":
                raise ValueError(f"human_detections must have source 'human'; {d.id} has {d.source!r}")
            d.check_in(self.image, ids)
        for a in self.adjudications:
            if a.record_id != self.record_id:
                raise ValueError(f"adjudication {a.id} is for record {a.record_id!r}, not {self.record_id!r}")
            if a.image_id != self.image.id:
                raise ValueError(f"adjudication {a.id} is for image {a.image_id!r}, not {self.image.id!r}")
        for f in self.final_candidates:
            if f.class_id not in ids:
                raise ValueError(f"final candidate has unknown class_id {f.class_id!r}")
        return self


EXPORTED_MODELS: tuple[type[BaseModel], ...] = (
    TaskSpec, ImageRef, Detection, StageA, StageB, RouteDecision,
    VerificationRequest, VerificationResult, AnnotationAudit, RouteRecord, ErrorRecord,
    Adjudication, ReviewedRecord,
)


def export_json_schemas(out_dir: str) -> list[str]:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for m in EXPORTED_MODELS:
        p = out / f"{m.__name__}.schema.json"
        p.write_text(json.dumps(m.model_json_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(str(p))
    return written
