"""Deterministic routing policy (route-v1).

``route`` is a pure function of typed scores, capabilities and config: no I/O,
clock, environment or process-dependent randomness. Audit sampling uses a
SHA-256 of (seed, image id), so it is stable across processes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .schemas import Action, BoxFeatures, Budget, Difficulty, RouteDecision, StageA
from .scoring import calibrated_probability_missing
from .slicing import planned_slices
from .task_config import RoutingConfig


@dataclass(frozen=True)
class Capabilities:
    secondary: bool = False
    sahi: bool = False
    vlm: bool = False


def stable_sample(image_id: str, seed: str, rate: float, salt: str = "") -> bool:
    if rate <= 0:
        return False
    h = hashlib.sha256(f"{seed}|{salt}|{image_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < rate


def route(
    image_id: str,
    width: int,
    height: int,
    sa: StageA,
    features: BoxFeatures,
    difficulty: Difficulty,
    caps: Capabilities,
    cfg: RoutingConfig,
) -> RouteDecision:
    actions: list[Action] = []
    reasons: list[str] = []
    priority = 0
    band = sa.relevance_band
    zero = features.num_boxes == 0
    d, q, pol = cfg.difficulty, cfg.quality, cfg.policy
    small = features.min_box_area_fraction is not None and features.min_box_area_fraction < d.small_box_area_fraction
    crowded = features.crowding >= d.severe_crowding
    poor_quality = sa.quality.low_resolution or sa.quality.blur >= q.poor_blur or sa.quality.exposure >= q.poor_exposure

    def add(*acts: Action) -> None:
        for a in acts:
            if a not in actions:
                actions.append(a)

    if sa.relevance_status != "calibrated":
        reasons.append(f"relevance_{sa.relevance_status}")
    if difficulty.status == "uncalibrated":
        reasons.append("difficulty_uncalibrated")
    if band == "unknown":
        reasons.append("relevance_unknown")
        band = pol.unknown_relevance_as  # "medium" (default) or "high"; never "low"
        if band == "high":
            reasons.append("unknown_relevance_treated_as_high")

    escalate = False  # rule 6: SAHI + secondary + VLM
    moderate = False  # rule 5: secondary + verify discrepancies

    if band == "low":
        if zero:
            reasons.append("low_relevance_zero_boxes")
            add("standard")
            sample_rate = max(cfg.audit.low_relevance_sample_rate, cfg.audit.zero_box_sample_rate)
            if stable_sample(image_id, cfg.audit.seed, sample_rate, "low_rel"):
                reasons.append("audit_sampled_low_relevance")
                add("audit_sample", "human_review")
                priority = max(priority, 1)
        else:
            reasons.append("low_relevance_with_detections")
            add("standard", "vlm_verify" if caps.vlm else "human_review")
            priority = max(priority, 1)
    elif zero:
        if band == "high":
            reasons.append("high_relevance_zero_boxes" if sa.relevance_band == "high" else "unknown_relevance_zero_boxes")
            escalate = True
        else:
            reasons.append("moderate_or_unknown_relevance_zero_boxes")
            moderate = True
    else:
        if difficulty.bucket == "hard" or small or crowded:
            reasons.append("high_miss_risk")
            if small:
                reasons.append("small_candidate")
            if crowded:
                reasons.append("severe_crowding")
            escalate = True
        elif difficulty.bucket in ("medium", "unknown") or band == "medium":
            reasons.append("medium_difficulty" if difficulty.bucket == "medium" else
                           "difficulty_unknown" if difficulty.bucket == "unknown" else "moderate_relevance")
            moderate = True
        else:
            reasons.append("high_relevance_low_difficulty")
            add("standard")

    expected_slices = 0
    if escalate:
        priority = 2
        ran_detector = False
        n_slices = planned_slices(width, height, cfg.slicing)
        if caps.sahi and n_slices <= cfg.budgets.max_slices:
            add("sahi")
            expected_slices = n_slices
            ran_detector = True
        elif caps.sahi:
            reasons.append("slice_budget_exceeded")
        else:
            reasons.append("sahi_unavailable")
        if caps.secondary:
            add("secondary")
            ran_detector = True
        else:
            reasons.append("secondary_unavailable")
        if caps.vlm:
            add("vlm_verify")
        if not ran_detector and not caps.vlm:
            add("human_review")
        elif not ran_detector:
            reasons.append("vlm_spot_check_only")
    elif moderate:
        priority = max(priority, 1)
        if caps.secondary:
            add("secondary")
            if caps.vlm:
                add("vlm_verify")
        elif caps.vlm and pol.moderate_without_secondary == "vlm_then_human":
            reasons.append("secondary_unavailable")
            add("vlm_verify")
        else:
            reasons.append("secondary_unavailable")
            add("human_review")

    unknown = [c for c in cfg.task.class_ids if calibrated_probability_missing(sa, c)]
    if unknown:  # calibrator omitted these classes: relevance unknown, a VLM cannot clear it -> human
        reasons.extend(f"calibrator_missing_class:{c}" for c in unknown)
        add("human_review")
        priority = max(priority, 1)
    if not actions:
        add("standard")
    if poor_quality and "audit_sample" not in actions and stable_sample(
            image_id, cfg.audit.seed, cfg.audit.poor_quality_sample_rate, "quality"):
        reasons.append("audit_sampled_poor_quality")
        add("audit_sample", "human_review")

    expected_vlm = cfg.budgets.max_vlm_calls if "vlm_verify" in actions else 0
    return RouteDecision(
        actions=tuple(actions), priority=priority, reason_codes=tuple(reasons),
        policy_version=cfg.policy_version,
        expected_budget=Budget(slices=expected_slices, vlm_calls=expected_vlm),
    )
