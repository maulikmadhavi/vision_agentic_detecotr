"""Relevance calibrator tooling (reviewer work order A5; board B-2).

Fits P(class present | raw score) for each class and P(any target present | max raw score) from
human labels (A3 format) and the raw per-class similarities a shadow run recorded in Stage A.
Fitting uses CALIBRATION-split images only. The model is a two-parameter logistic regression,
fitted by Newton's method (numpy only) with a small ridge on the standardized slope, so perfectly
separable data still gives finite parameters (flagged ``separable``). It refuses to fit with fewer
than MIN_EACH positives or negatives, per class and for any-target.

The any-target probability is fitted directly (no independence assumption) and then kept
coherent with the class probabilities: max(p_class) <= P(any) <= min(1, sum(p_class)).

Real-image fitting waits for the human-labelled sample; the artifact records the labels'
declared provenance, so a calibrator fitted on synthetic labels says so.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .evaluation import assert_split_allows
from .metrics import load_ground_truth
from .sample import label_digest
from .schemas import RouteRecord, canonical_json
from .task_config import RoutingConfig

MIN_EACH = 10
RIDGE = 1.0  # L2 penalty on the standardized slope; the intercept is not penalized


class CalibrationError(ValueError):
    """The calibrator cannot be fitted or applied as asked."""


@dataclass(frozen=True)
class FitRow:
    image_id: str
    scores: Mapping[str, float]          # raw similarity per class
    present: Mapping[str, bool | None]   # None: excluded for that class (only ignore boxes)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def fit_logistic(xs: Sequence[float], ys: Sequence[bool], ridge: float = RIDGE) -> dict:
    """Logistic regression y ~ x; returns intercept and slope on the raw x scale."""
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    mean, std = float(x.mean()), float(x.std()) or 1.0
    design = np.column_stack([np.ones_like(x), (x - mean) / std])
    penalty = np.diag([0.0, ridge])
    w = np.zeros(2)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(design @ w)))
        gradient = design.T @ (p - y) + penalty @ w
        hessian = design.T @ (design * (p * (1.0 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        w = w - step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    else:
        raise CalibrationError("the logistic fit did not converge")
    positives, negatives = x[y == 1], x[y == 0]
    separable = bool(positives.size and negatives.size and
                     (positives.min() > negatives.max() or positives.max() < negatives.min()))
    return {"intercept": float(w[0] - w[1] * mean / std), "slope": float(w[1] / std), "separable": separable}


def _fit_counted(name: str, data: list[tuple[float, bool]], min_each: int) -> dict:
    n_pos = sum(1 for _, y in data if y)
    n_neg = len(data) - n_pos
    if n_pos < min_each or n_neg < min_each:
        raise CalibrationError(f"{name}: {n_pos} positives and {n_neg} negatives in the calibration split; "
                               f"fitting needs at least {min_each} of each")
    return {**fit_logistic([x for x, _ in data], [y for _, y in data]), "n_pos": n_pos, "n_neg": n_neg}


def fit_relevance_rows(rows: Sequence[FitRow], class_ids: Sequence[str], min_each: int = MIN_EACH) -> dict:
    """Fit any-target first, then each class; refuses below ``min_each`` positives or negatives."""
    rows = sorted(rows, key=lambda r: r.image_id)
    any_data = []
    for r in rows:
        labels = [r.present[c] for c in class_ids]
        if any(v is True for v in labels):
            any_data.append((max(r.scores[c] for c in class_ids), True))
        elif all(v is False for v in labels):
            any_data.append((max(r.scores[c] for c in class_ids), False))
    any_target = _fit_counted("any target", any_data, min_each)
    classes = {c: _fit_counted(f"class {c!r}", [(r.scores[c], r.present[c]) for r in rows
                                                if r.present[c] is not None], min_each)
               for c in class_ids}
    return {"classes": classes, "any_target": any_target}


def calibrate_scores(params: Mapping, raw: Mapping[str, float]) -> tuple[dict[str, float], float]:
    """Per-class probabilities and a coherent P(any target) for one image's raw scores."""
    per_class = {c: _sigmoid(p["intercept"] + p["slope"] * raw[c]) for c, p in params["classes"].items()}
    a = params["any_target"]
    any_p = _sigmoid(a["intercept"] + a["slope"] * max(raw[c] for c in per_class))
    low, high = max(per_class.values()), min(1.0, sum(per_class.values()))
    return per_class, min(max(any_p, low), high)


def _digest(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def fit_relevance(records: Sequence[RouteRecord], label_doc: Mapping, split_manifest: Mapping,
                  cfg: RoutingConfig) -> dict:
    """Fit a relevance calibrator from a shadow trace, labels and the split; returns the artifact."""
    scale, task = cfg.relevance.raw_scale, cfg.task
    if scale == "unspecified":
        raise CalibrationError("relevance.raw_scale is 'unspecified'; a calibrator is fitted for one declared "
                               "score scale")
    scorers = sorted({str(r.run.relevance_scorer) for r in records})
    if len(scorers) != 1 or scorers == ["None"]:
        raise CalibrationError(f"the trace must come from exactly one relevance scorer; found {scorers}")
    if any(r.task != task for r in records):
        raise CalibrationError("the trace was routed with a different task (classes or prompts) than --config")
    assert_split_allows(split_manifest, sorted(r.image.id for r in records), ("calibration",))
    ground_truth = load_ground_truth(label_doc, task)
    rows, excluded = [], {"unlabelled": [], "relevance_unavailable": []}
    for record in sorted(records, key=lambda r: r.image.id):
        gt = ground_truth.get(record.image.id)
        if gt is None:
            excluded["unlabelled"].append(record.image.id)
            continue
        per_class = record.stage_a.per_class
        if any(c not in per_class or per_class[c].raw_similarity is None for c in task.class_ids):
            excluded["relevance_unavailable"].append(record.image.id)
            continue
        present = {}
        for c in task.class_ids:
            boxes = [b for b in gt.boxes if b.class_id == c]
            present[c] = True if any(not b.ignore for b in boxes) else (None if boxes else False)
        rows.append(FitRow(record.image.id, {c: per_class[c].raw_similarity for c in task.class_ids}, present))
    fitted = fit_relevance_rows(rows, task.class_ids)
    scorer = scorers[0]
    wemm = re.fullmatch(r"wemm:.+@([^+]+)\+max(\d+)", scorer)
    return {
        "version": "relevance-calibrator-v1", "scale": scale, "scorer": scorer,
        "scorer_build": wemm.group(1) if wemm else None, "max_side": int(wemm.group(2)) if wemm else None,
        "task_id": task.task_id, "class_ids": list(task.class_ids),
        "prompt_sha256": _digest([[c.class_id, list(c.positive_prompts), list(c.negative_prompts)]
                                  for c in task.classes]),
        "label_provenance": label_doc.get("vad_label_provenance"),
        "inputs": {"labels_sha256": label_digest(label_doc),
                   "data_sha256": _digest([[r.image_id, dict(r.scores), dict(r.present)] for r in rows]),
                   "split_manifest_sha256": _digest(split_manifest)},
        "ridge": RIDGE, "min_each": MIN_EACH, **fitted, "excluded": excluded,
    }
