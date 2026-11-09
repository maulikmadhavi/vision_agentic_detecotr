"""Stage A (pre-detection) and Stage B (post-detection) feature extraction.

Relevance, image quality, and difficulty are separate quantities and are never
folded into each other. Heuristic scores are labelled ``uncalibrated``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from PIL import Image

from .adapters import RelevanceCalibrator, RelevanceScorer
from .matching import iou
from .schemas import BoxFeatures, ClassRelevance, Detection, Difficulty, Disagreement, ImageRef, Quality, StageA
from .task_config import RoutingConfig


def quality_signals(image: Image.Image, low_resolution_px: int) -> Quality:
    g = np.asarray(image.convert("L"), dtype=np.float32)
    if min(g.shape) >= 3:
        lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
        var = float(lap.var())
    else:
        var = 0.0
    min_side = min(image.width, image.height)
    return Quality(
        blur=round(1.0 / (1.0 + var / 100.0), 4),
        exposure=round(float(((g <= 5) | (g >= 250)).mean()), 4),
        min_side_px=min_side,
        low_resolution=min_side < low_resolution_px,
    )


def stage_a(
    image: Image.Image,
    ref: ImageRef,
    cfg: RoutingConfig,
    scorer: RelevanceScorer | None = None,
    calibrator: RelevanceCalibrator | None = None,
) -> StageA:
    quality = quality_signals(image, cfg.difficulty.low_resolution_px)
    warnings: list[str] = []
    raw: dict[str, float] = {}
    if scorer is not None:
        prompts = {c.class_id: (c.positive_prompts, c.negative_prompts) for c in cfg.task.classes}
        raw = {k: v for k, v in scorer.score(image, ref.id, prompts).items() if k in cfg.task.class_ids}
        missing = set(cfg.task.class_ids) - set(raw)
        if missing:
            warnings.append(f"relevance_missing_classes:{','.join(sorted(missing))}")
    if quality.low_resolution:
        warnings.append("low_resolution")

    if not raw:
        return StageA(per_class={}, relevance_status="unavailable", relevance_band="unknown",
                      quality=quality, warnings=tuple(warnings))

    t = cfg.relevance
    incomplete = bool(set(cfg.task.class_ids) - set(raw))

    def no_low_if_incomplete(band: str) -> str:
        # Incomplete evidence must never produce a low-relevance gate, calibrated or not:
        # an unscored class could be the one that is present.
        if incomplete and band == "low":
            warnings.append("incomplete_relevance_low_band_raised_to_medium")
            return "medium"
        return band

    if calibrator is not None:
        probs, any_p = calibrator.calibrate(raw)
        # R4-1: a low band needs per-class probabilities for every task class and a coherent
        # aggregate, P(any) >= P(class). Range/finiteness is enforced by the schema.
        if not (set(cfg.task.class_ids) <= set(probs) and all(any_p >= p - 1e-6 for p in probs.values())):
            warnings.append("calibrator_output_incomplete_or_inconsistent")
            incomplete = True
        per_class = {k: ClassRelevance(raw_similarity=v, relevance_probability=probs.get(k)) for k, v in raw.items()}
        band = no_low_if_incomplete("high" if any_p >= t.prob_high else "low" if any_p < t.prob_low else "medium")
        return StageA(per_class=per_class, image_relevance_probability=any_p, relevance_status="calibrated",
                      relevance_band=band, quality=quality, warnings=tuple(warnings))

    top = max(raw.values())
    band = no_low_if_incomplete("high" if top >= t.raw_high else "low" if top < t.raw_low else "medium")
    per_class = {k: ClassRelevance(raw_similarity=v) for k, v in raw.items()}
    return StageA(per_class=per_class, relevance_status="uncalibrated", relevance_band=band,
                  quality=quality, warnings=tuple(warnings))


def calibrated_probability_missing(sa: StageA, class_id: str) -> bool:
    """Calibrated Stage A without a probability for this class: a calibrator failure, relevance unknown."""
    r = sa.per_class.get(class_id)
    return sa.relevance_status == "calibrated" and (r is None or r.relevance_probability is None)


def class_is_high_relevance(sa: StageA, class_id: str, cfg: RoutingConfig) -> bool:
    r = sa.per_class.get(class_id)
    if r is None:
        return False
    if r.relevance_probability is not None:
        return r.relevance_probability >= cfg.relevance.prob_high
    if sa.relevance_status == "calibrated":
        return False  # calibrator omitted it: unknown, handled via calibrated_probability_missing
    return r.raw_similarity is not None and r.raw_similarity >= cfg.relevance.raw_high


def box_features(dets: Sequence[Detection], ref: ImageRef, sa: StageA, cfg: RoutingConfig) -> BoxFeatures:
    n = len(dets)
    img_area = float(ref.width * ref.height)
    confs = [d.raw_confidence for d in dets if d.raw_confidence is not None]
    crowded = ambiguous = 0
    for i, a in enumerate(dets):
        others = [b for j, b in enumerate(dets) if j != i]
        if any(iou(a.box_xyxy, b.box_xyxy) >= cfg.difficulty.crowding_iou for b in others):
            crowded += 1
        if any(b.class_id != a.class_id and iou(a.box_xyxy, b.box_xyxy) >= cfg.difficulty.ambiguity_iou for b in others):
            ambiguous += 1
    return BoxFeatures(
        num_boxes=n,
        per_class_counts=dict(sorted(Counter(d.class_id for d in dets).items())),
        min_box_area_fraction=(min((d.box_xyxy[2] - d.box_xyxy[0]) * (d.box_xyxy[3] - d.box_xyxy[1]) for d in dets) / img_area) if n else None,
        crowding=crowded / n if n else 0.0,
        class_ambiguity=ambiguous / n if n else 0.0,
        mean_confidence=sum(confs) / len(confs) if confs else None,
        min_confidence=min(confs) if confs else None,
        high_relevance_zero_boxes=(n == 0 and sa.relevance_band == "high"),
    )


def _bucket(value: float, cfg: RoutingConfig) -> str:
    return "hard" if value >= cfg.difficulty.hard else "medium" if value >= cfg.difficulty.medium else "easy"


def estimate_difficulty(sa: StageA, f: BoxFeatures, dis: Disagreement, cfg: RoutingConfig) -> Difficulty:
    """Heuristic miss-risk score in [0, 1]; ``unknown`` when evidence is insufficient.

    Zero boxes never yield a difficulty value: absence of detections says
    nothing about how hard a present object would be to find.
    """
    if f.num_boxes == 0:
        return Difficulty(status="unknown", bucket="unknown", reasons=("zero_primary_boxes",))
    if f.min_confidence is None:
        return Difficulty(status="unknown", bucket="unknown", reasons=("confidence_unavailable",))
    d = cfg.difficulty
    small = min(1.0, d.small_box_area_fraction / max(f.min_box_area_fraction or 0.0, 1e-9))
    terms = {
        "small": small,
        "crowding": f.crowding,
        "low_confidence": 1.0 - f.min_confidence,
        "blur": sa.quality.blur,
        "ambiguity": f.class_ambiguity,
    }
    if dis.status == "computed":
        total = dis.matched + dis.primary_only + dis.secondary_only
        terms["disagreement"] = (dis.primary_only + dis.secondary_only + dis.mislocalized) / total if total else 0.0
    w = {k: d.weights.get(k, 0.0) for k in terms}
    wsum = sum(w.values()) or 1.0
    value = round(min(1.0, max(0.0, sum(w[k] * terms[k] for k in terms) / wsum)), 4)
    reasons = tuple(f"{k}_high" for k, v in terms.items() if v >= 0.5 and w[k] > 0)
    return Difficulty(value=value, status="uncalibrated", bucket=_bucket(value, cfg), reasons=reasons)
