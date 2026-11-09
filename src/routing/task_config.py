"""Routing configuration: class registry, thresholds, budgets, versions.

Every number here is a configuration placeholder until Phase 5 calibration.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .schemas import Strict, TaskSpec, canonical_json


class RelevanceThresholds(Strict):
    # Raw-similarity bands are only meaningful on the scorer's own scale. A scorer that
    # declares ``score_scale`` (e.g. services_embedding.client.WeMMMCPRelevanceScorer) is refused unless
    # ``raw_scale`` names the same scale, so bands tuned for one scale never apply to another.
    raw_scale: str = "unspecified"
    raw_high: float = 0.5
    raw_low: float = 0.3
    prob_high: float = Field(default=0.7, ge=0, le=1)
    prob_low: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def _order(self) -> RelevanceThresholds:
        if not (self.raw_low < self.raw_high and self.prob_low < self.prob_high):
            raise ValueError("relevance low threshold must be below high threshold")
        return self


class DifficultyConfig(Strict):
    medium: float = Field(default=0.35, ge=0, le=1)
    hard: float = Field(default=0.6, ge=0, le=1)
    small_box_area_fraction: float = Field(default=0.001, gt=0, lt=1)
    crowding_iou: float = Field(default=0.3, gt=0, lt=1)
    severe_crowding: float = Field(default=0.5, ge=0, le=1)
    ambiguity_iou: float = Field(default=0.7, gt=0, lt=1)
    low_resolution_px: int = Field(default=320, gt=0)
    weights: dict[str, float] = Field(default_factory=lambda: {
        "small": 0.3, "crowding": 0.2, "low_confidence": 0.25, "blur": 0.1,
        "ambiguity": 0.1, "disagreement": 0.25,
    })

    @model_validator(mode="after")
    def _order(self) -> DifficultyConfig:
        if self.medium >= self.hard:
            raise ValueError("difficulty.medium must be below difficulty.hard")
        unknown = set(self.weights) - {"small", "crowding", "low_confidence", "blur", "ambiguity", "disagreement"}
        if unknown:
            raise ValueError(f"unknown difficulty weights {sorted(unknown)}")
        return self


class QualityConfig(Strict):
    """Heuristic "poor quality" cut-offs used by the policy for audit sampling (uncalibrated)."""

    poor_blur: float = Field(default=0.8, gt=0, le=1)
    poor_exposure: float = Field(default=0.5, gt=0, le=1)


class PolicyConfig(Strict):
    """Policy choices the plan left open. Defaults reproduce route-v1 as first shipped.

    These are decisions for the project owner; see docs/project_board.md (D-1..D-3).
    """

    # How relevance band "unknown" (no scorer / missing scores) is treated by the router.
    unknown_relevance_as: Literal["medium", "high"] = "medium"
    # Moderate route with no secondary detector: VLM first (human if no VLM), or always human.
    moderate_without_secondary: Literal["vlm_then_human", "human"] = "vlm_then_human"
    # A fused cluster whose members disagree on class stays in review even if the VLM confirms it.
    cross_class_always_review: bool = True


class MatchingConfig(Strict):
    iou_threshold: float = Field(default=0.5, gt=0, lt=1)
    mislocalized_iou: float = Field(default=0.75, gt=0, le=1)
    # secondary-detector class name -> task class_id. Identity when absent.
    class_mapping: dict[str, str] = Field(default_factory=dict)


class SlicingConfig(Strict):
    tile: int = Field(default=640, gt=0)
    overlap: float = Field(default=0.2, ge=0, lt=1)
    include_full_image: bool = True
    merge_ios: float = Field(default=0.5, gt=0, le=1)


class BudgetConfig(Strict):
    max_slices: int = Field(default=16, ge=0)
    max_vlm_calls: int = Field(default=3, ge=0)
    max_elapsed_ms: float = Field(default=60_000, gt=0)
    max_gpu_ms: float = Field(default=30_000, gt=0)


class AuditConfig(Strict):
    seed: str = "audit-seed-1"
    low_relevance_sample_rate: float = Field(default=0.05, ge=0, le=1)
    zero_box_sample_rate: float = Field(default=0.1, ge=0, le=1)
    poor_quality_sample_rate: float = Field(default=0.2, ge=0, le=1)
    calibrated: bool = False  # only a calibrated audit policy may mark images complete


class VLMConfig(Strict):
    enabled: bool = False
    # No built-in endpoint or model: any OpenAI-compatible server, named explicitly in the
    # config whenever the VLM is enabled (the example config uses local LM Studio).
    base_url: str | None = None
    model: str | None = None
    # Name of an environment variable holding a bearer token (never the token itself).
    # Not part of config_hash: credentials do not change results or cache identity.
    api_key_env: str | None = None
    timeout_s: float = Field(default=60, gt=0)
    max_tokens: int = Field(default=200, gt=0)
    crop_padding: float = Field(default=0.25, ge=0)
    # A detector-authored (non-primary) box confirmed by the VLM may be accepted.
    accept_confirmed_detector_proposals: bool = False
    # An "absent" verdict rejects the candidate only when this is true. Off until the
    # VLM's "absent" precision is measured against human labels (reviewer §5.4);
    # otherwise the candidate goes to review with flag ``vlm_absent``.
    absent_rejects: bool = False


class RoutingConfig(Strict):
    routing_enabled: bool = False  # feature flag; disabled = legacy primary-only output
    policy_version: str = "route-v1"
    task: TaskSpec
    relevance: RelevanceThresholds = RelevanceThresholds()
    difficulty: DifficultyConfig = DifficultyConfig()
    quality: QualityConfig = QualityConfig()
    policy: PolicyConfig = PolicyConfig()
    matching: MatchingConfig = MatchingConfig()
    slicing: SlicingConfig = SlicingConfig()
    budgets: BudgetConfig = BudgetConfig()
    audit: AuditConfig = AuditConfig()
    vlm: VLMConfig = VLMConfig()

    @model_validator(mode="after")
    def _mapping_targets(self) -> RoutingConfig:
        bad = set(self.matching.class_mapping.values()) - set(self.task.class_ids)
        if bad:
            raise ValueError(f"class_mapping targets unknown class_ids {sorted(bad)}")
        return self

    def config_hash(self) -> str:
        data = self.model_dump(mode="json", exclude={"vlm": {"api_key_env"}})
        return hashlib.sha256(canonical_json(data).encode()).hexdigest()[:16]


def load_config(path: str | Path) -> RoutingConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RoutingConfig.model_validate(data)
