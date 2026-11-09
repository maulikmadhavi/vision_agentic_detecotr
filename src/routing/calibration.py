"""Relevance calibrator tooling (reviewer work order A5; board B-2).

Fits P(class present | raw score) for each class and P(any target present | max raw score) from
human labels (A3 format) and the raw per-class similarities a shadow run recorded in Stage A.
Fitting uses CALIBRATION-split images only. The model is a two-parameter logistic regression,
fitted by Newton's method (numpy only) with a ridge (RIDGE) on the standardized slope. The ridge
is applied to EVERY fit, not only to separable data: with the 20 to 30 points a calibration split
holds, slopes are shrunk towards zero, so the parameters are penalized estimates, not plain
maximum likelihood (the artifact records ``ridge``). Perfectly separable data still gives finite
parameters (flagged ``separable``). It refuses to fit with fewer than MIN_EACH positives or
negatives, per class and for any-target.

The any-target probability is fitted directly (no independence assumption) and then kept
coherent with the class probabilities: max(p_class) <= P(any) <= min(1, sum(p_class)).

A fitted artifact is loaded with FittedRelevanceCalibrator.load, which refuses another score
scale, another scorer (model, build or max_side) or other prompts. Its ``version`` carries the
file's SHA-256, so it enters the run key only of runs that use it. A class the scorer did not
score gets no probability: Stage A keeps it unknown and the policy sends the image to review.

Real-image fitting waits for the human-labelled sample; the artifact records the labels'
declared provenance, so a calibrator fitted on synthetic labels says so.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from .evaluation import IsolationError, assert_split_allows, check_image_content
from .image_inputs import repeated_ids
from .metrics import load_ground_truth
from .sample import label_digest
from .schemas import RelevanceCalibratorArtifact, RouteRecord, TaskSpec, canonical_json
from .task_config import RoutingConfig, shadow_run_config

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


def _refuse_repeats(image_ids: Sequence[str]) -> None:
    """One observation per image: repeated rows would reweight the fit and count towards MIN_EACH."""
    if repeated := repeated_ids(image_ids):
        raise CalibrationError(f"repeated image ids {repeated[:10]}; the fit needs one observation per image "
                               "(rebuild a concatenated or retried trace)")


def fit_relevance_rows(rows: Sequence[FitRow], class_ids: Sequence[str], min_each: int = MIN_EACH) -> dict:
    """Fit any-target first, then each class; refuses repeated images and fewer than ``min_each``
    positives or negatives (so the minimum counts distinct images)."""
    _refuse_repeats([r.image_id for r in rows])
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
    # A class without a raw score gets no probability (Stage A then treats it as unknown).
    per_class = {c: _sigmoid(p["intercept"] + p["slope"] * raw[c]) for c, p in params["classes"].items() if c in raw}
    a = params["any_target"]
    any_p = _sigmoid(a["intercept"] + a["slope"] * max(raw[c] for c in per_class))
    low, high = max(per_class.values()), min(1.0, sum(per_class.values()))
    return per_class, min(max(any_p, low), high)


def _digest(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def prompt_digest(task: TaskSpec) -> str:
    """SHA-256 over each class id with its positive and negative prompts, in task order."""
    return _digest([[c.class_id, list(c.positive_prompts), list(c.negative_prompts)] for c in task.classes])


_WEMM = re.compile(r"wemm:(.+)@([^+]+)\+max(\d+)")


class FittedRelevanceCalibrator:
    """A RelevanceCalibrator loaded from a `vad-route fit-relevance` artifact.

    The scorer must match exactly. A calibrator fitted under a WeMM "@unverified" build therefore
    loads for another "@unverified" run, which does not prove the same build; requiring a verified
    build is a promotion gate (project board), not a load rule.
    """

    def __init__(self, artifact: RelevanceCalibratorArtifact, file_sha256: str):
        self.artifact = artifact
        self.version = f"{artifact.version}@sha256:{file_sha256}"
        self._params = artifact.model_dump(mode="json")

    @classmethod
    def load(cls, path: str | Path, cfg: RoutingConfig, scorer_version: str) -> FittedRelevanceCalibrator:
        data = Path(path).read_bytes()
        try:
            art = RelevanceCalibratorArtifact.model_validate_json(data)
        except ValidationError as e:
            raise CalibrationError(f"{path} is not a relevance calibrator: {e}") from None
        if art.scale != cfg.relevance.raw_scale:
            raise CalibrationError(f"the calibrator was fitted for score scale {art.scale!r}; this config declares "
                                   f"relevance.raw_scale {cfg.relevance.raw_scale!r}")
        if art.scorer != scorer_version:
            fitted, used = _WEMM.fullmatch(art.scorer), _WEMM.fullmatch(scorer_version)
            parts = [name for name, k in (("model", 1), ("build", 2), ("max_side", 3))
                     if fitted and used and fitted.group(k) != used.group(k)]
            raise CalibrationError(f"the calibrator was fitted for scorer {art.scorer!r}; this run uses "
                                   f"{scorer_version!r}" + (f" (differs in {', '.join(parts)})" if parts else ""))
        if art.prompt_sha256 != prompt_digest(cfg.task):
            raise CalibrationError("the calibrator was fitted for other classes or prompts than this config's task "
                                   "(prompt_sha256 differs)")
        return cls(art, hashlib.sha256(data).hexdigest())

    def calibrate(self, raw: Mapping[str, float]) -> tuple[dict[str, float], float]:
        return calibrate_scores(self._params, raw)


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
    _refuse_repeats([r.image.id for r in records])
    if shadow_run_config(cfg, {r.run.config_hash for r in records}) is None:
        raise CalibrationError(f"--config does not match the run that made the trace (run config hashes "
                               f"{sorted({r.run.config_hash for r in records})}); the fit records the scale, scorer "
                               "and prompts of that run, so it needs the config the shadow run used")
    if split_manifest.get("version") != "split-v2":
        raise CalibrationError("fitting needs a split-v2 manifest (with each image's sha256 and size) from "
                               "`vad-route sample split`")
    assert_split_allows(split_manifest, sorted(r.image.id for r in records), ("calibration",))
    images = split_manifest["images"]
    check_image_content(images, sorted((r.image for r in records), key=lambda ref: ref.id), "the split manifest")
    frozen_bytes = {img["sha256"]: i for i, img in sorted(images.items()) if img["split"] == "frozen_eval"}
    if reentered := sorted((r.image.id, frozen_bytes[r.image.sha256]) for r in records
                           if r.image.sha256 in frozen_bytes):
        raise IsolationError(f"image {reentered[0][0]!r} has the same bytes as frozen image {reentered[0][1]!r}; "
                             f"a frozen image must not be fitted on under another id ({len(reentered)} such images)")
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
    wemm = _WEMM.fullmatch(scorer)
    return {
        "version": "relevance-calibrator-v1", "scale": scale, "scorer": scorer,
        "scorer_build": wemm.group(2) if wemm else None, "max_side": int(wemm.group(3)) if wemm else None,
        "task_id": task.task_id, "class_ids": list(task.class_ids), "prompt_sha256": prompt_digest(task),
        "label_provenance": label_doc.get("vad_label_provenance"),
        "inputs": {"labels_sha256": label_digest(label_doc),
                   "data_sha256": _digest([[r.image_id, dict(r.scores), dict(r.present)] for r in rows]),
                   "split_manifest_sha256": _digest(split_manifest)},
        "ridge": RIDGE, "min_each": MIN_EACH, **fitted, "excluded": excluded,
    }
