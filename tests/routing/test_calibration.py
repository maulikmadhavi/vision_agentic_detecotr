"""A5 stage 1: relevance calibrator fitting (synthetic scores and labels only)."""

from __future__ import annotations

import json
import math
import random

import pytest
import yaml

from routing import cli
from routing.calibration import (
    MIN_EACH, CalibrationError, FitRow, calibrate_scores, fit_logistic, fit_relevance, fit_relevance_rows,
)
from routing.evaluation import IsolationError
from routing.image_inputs import ManifestEntry
from routing.pipeline import RoutingPipeline
from routing.store import RouteStore

from .helpers import CLASS_A, CLASS_B, ScriptedDetector, ScriptedRelevance, config, draw

CLASSES = (CLASS_A, CLASS_B)


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


# --- the logistic fit ---------------------------------------------------------------------------


def test_fit_recovers_a_known_logistic_relation():
    rng = random.Random(7)
    xs = [rng.uniform(0.0, 0.8) for _ in range(3000)]
    ys = [rng.random() < _sigmoid(-4 + 10 * x) for x in xs]
    fit = fit_logistic(xs, ys)
    assert fit["slope"] == pytest.approx(10, rel=0.15) and fit["intercept"] == pytest.approx(-4, rel=0.15)
    assert fit["separable"] is False


def test_separable_data_gets_finite_ridge_limited_parameters_and_is_flagged():
    xs = [0.1 + 0.01 * i for i in range(12)] + [0.5 + 0.01 * i for i in range(12)]
    ys = [False] * 12 + [True] * 12
    fit = fit_logistic(xs, ys)
    assert fit["separable"] is True and math.isfinite(fit["slope"]) and fit["slope"] > 0
    probs = [_sigmoid(fit["intercept"] + fit["slope"] * x) for x in sorted(xs)]
    assert probs == sorted(probs) and 0 < probs[0] < 0.5 < probs[-1] < 1


def test_fit_is_deterministic():
    xs = [0.05 * i for i in range(20)]
    ys = [i % 3 == 0 or i > 12 for i in range(20)]
    assert fit_logistic(xs, ys) == fit_logistic(xs, ys)


# --- per class and any-target, with refusal ------------------------------------------------------


def _rows(n=40, n_pos_a=12, any_neg=True, seed=3):
    """40 images: class A positive for the first 12, class B on even images; positives score higher
    (with overlap). 14 images have no target at all unless any_neg is False."""
    rng, rows = random.Random(seed), []
    for k in range(n):
        a, b = k < n_pos_a, k % 2 == 0
        if not any_neg and not a and not b:
            b = True
        scores = {CLASS_A: (0.45 if a else 0.2) + rng.uniform(-0.15, 0.15),
                  CLASS_B: (0.45 if b else 0.2) + rng.uniform(-0.15, 0.15)}
        rows.append(FitRow(f"img{k}", scores, {CLASS_A: a, CLASS_B: b}))
    return rows


def test_fit_rows_gives_a_logistic_per_class_and_for_any_target_with_counts():
    out = fit_relevance_rows(_rows(), CLASSES)
    for key in (CLASS_A, CLASS_B):
        assert out["classes"][key]["n_pos"] >= MIN_EACH and out["classes"][key]["n_neg"] >= MIN_EACH
        assert out["classes"][key]["slope"] > 0
    assert set(out["any_target"]) >= {"intercept", "slope", "n_pos", "n_neg", "separable"}


def test_too_few_positives_for_a_class_is_refused_with_the_counts():
    rows = _rows()
    rows = [FitRow(r.image_id, r.scores, {**r.present, CLASS_A: False if r.present[CLASS_A] and
                                           int(r.image_id[3:]) >= 9 else r.present[CLASS_A]}) for r in rows]
    with pytest.raises(CalibrationError, match=f"{CLASS_A}.*9 positives and 31 negatives.*at least 10"):
        fit_relevance_rows(rows, CLASSES)


def test_too_few_negatives_for_any_target_is_refused():
    with pytest.raises(CalibrationError, match="any target.*negatives"):
        fit_relevance_rows(_rows(any_neg=False), CLASSES)


def test_excluded_labels_do_not_count():
    rows = [FitRow(r.image_id, r.scores, {**r.present, CLASS_B: None}) if int(r.image_id[3:]) < 5 else r
            for r in _rows()]
    out = fit_relevance_rows(rows, CLASSES)
    assert out["classes"][CLASS_B]["n_pos"] + out["classes"][CLASS_B]["n_neg"] == 35


# --- coherence ----------------------------------------------------------------------------------


def _params(any_intercept, any_slope):
    flat = {"intercept": 0.0, "slope": 0.0, "n_pos": 10, "n_neg": 10, "separable": False}
    return {"classes": {CLASS_A: {**flat, "intercept": -1.0, "slope": 4.0}, CLASS_B: {**flat, "intercept": -2.0}},
            "any_target": {**flat, "intercept": any_intercept, "slope": any_slope}}


def test_any_target_is_clipped_up_to_the_largest_class_probability():
    per_class, any_p = calibrate_scores(_params(-10.0, 0.0), {CLASS_A: 0.9, CLASS_B: 0.1})
    assert any_p == pytest.approx(max(per_class.values()))


def test_any_target_is_clipped_down_to_the_sum_of_class_probabilities():
    per_class, any_p = calibrate_scores(_params(10.0, 0.0), {CLASS_A: 0.0, CLASS_B: 0.0})
    assert any_p == pytest.approx(min(1.0, sum(per_class.values())))


def test_any_target_stays_within_the_coherent_range():
    rng = random.Random(11)
    for _ in range(500):
        params = _params(rng.uniform(-6, 6), rng.uniform(-10, 10))
        per_class, any_p = calibrate_scores(params, {CLASS_A: rng.uniform(0, 1), CLASS_B: rng.uniform(0, 1)})
        assert max(per_class.values()) - 1e-12 <= any_p <= min(1.0, sum(per_class.values())) + 1e-12


# --- from a trace, labels and a split manifest ---------------------------------------------------


SCALE_CFG = config(relevance={"raw_scale": "test_scale_v1", "raw_high": 0.6, "raw_low": 0.2})


def _records(tmp_path, n=30):
    rng, records, label_images, label_anns, manifest = random.Random(5), [], [], [], {}
    for k in range(n):
        name = f"img{k}"
        has_a, has_b = k % 2 == 0, k % 3 == 0
        scores = {CLASS_A: (0.5 if has_a else 0.2) + rng.uniform(-0.15, 0.15),
                  CLASS_B: (0.5 if has_b else 0.2) + rng.uniform(-0.15, 0.15)}
        entry = ManifestEntry(str(draw(tmp_path / f"{name}.png", (64, 48), [])), image_id=name)
        records.append(RoutingPipeline(SCALE_CFG, ScriptedDetector({}), RouteStore(":memory:"),
                                       relevance=ScriptedRelevance({name: scores})).process(entry))
        label_images.append({"id": k + 1, "vad_image_id": name, "width": 64, "height": 48, "vad_exhaustive": True})
        for cat, present in ((1, has_a), (2, has_b)):
            if present:
                label_anns.append({"id": len(label_anns) + 1, "image_id": k + 1, "category_id": cat,
                                   "bbox": [5, 5, 10, 10]})
        manifest[name] = {"group_id": f"g{k}", "split": "calibration"}
    labels = {"vad_label_provenance": "synthetic", "categories": [{"id": 1, "name": CLASS_A}, {"id": 2, "name": CLASS_B}],
              "images": label_images, "annotations": label_anns}
    return records, labels, {"version": "split-v1", "images": manifest}


def test_fit_from_a_trace_writes_a_complete_artifact(tmp_path):
    records, labels, manifest = _records(tmp_path)
    art = fit_relevance(records, labels, manifest, SCALE_CFG)
    assert art["version"] == "relevance-calibrator-v1" and art["scale"] == "test_scale_v1"
    assert art["scorer"] == "rel@1" and art["task_id"] == "t-test" and art["class_ids"] == list(CLASSES)
    assert len(art["prompt_sha256"]) == 64 and art["max_side"] is None and art["scorer_build"] is None
    assert set(art["inputs"]) == {"labels_sha256", "data_sha256", "split_manifest_sha256"}
    assert art["label_provenance"] == "synthetic" and art["ridge"] > 0
    assert art["classes"][CLASS_A]["n_pos"] == 15 and art["classes"][CLASS_A]["n_neg"] == 15
    assert (art["classes"][CLASS_B]["n_pos"], art["any_target"]["n_neg"]) == (10, 10)
    assert set(art["any_target"]) >= {"intercept", "slope", "n_pos", "n_neg", "separable"}


def test_the_same_inputs_give_a_byte_identical_artifact(tmp_path):
    records, labels, manifest = _records(tmp_path)
    first = json.dumps(fit_relevance(records, labels, manifest, SCALE_CFG), sort_keys=True)
    assert first == json.dumps(fit_relevance(records[::-1], labels, manifest, SCALE_CFG), sort_keys=True)


@pytest.mark.parametrize("split", ["development", "frozen_eval"])
def test_fitting_refuses_images_outside_the_calibration_split(tmp_path, split):
    records, labels, manifest = _records(tmp_path)
    manifest["images"]["img3"]["split"] = split
    with pytest.raises(IsolationError, match="img3"):
        fit_relevance(records, labels, manifest, SCALE_CFG)


def test_fitting_needs_a_declared_scale_and_one_scorer(tmp_path):
    records, labels, manifest = _records(tmp_path)
    with pytest.raises(CalibrationError, match="raw_scale"):
        fit_relevance(records, labels, manifest, config())
    mixed = records[:-1] + [records[-1].model_copy(update={"run": records[-1].run.model_copy(
        update={"relevance_scorer": "other@2"})})]
    with pytest.raises(CalibrationError, match="scorer"):
        fit_relevance(mixed, labels, manifest, SCALE_CFG)


def test_an_image_whose_only_box_of_a_class_is_ignore_is_left_out_of_that_class(tmp_path):
    records, labels, manifest = _records(tmp_path)
    # img2 has class A and no class B: an ignore-only class-B box removes it from class B only
    labels["annotations"].append({"id": 999, "image_id": 3, "category_id": 2, "bbox": [5, 5, 10, 10], "ignore": True})
    art = fit_relevance(records, labels, manifest, SCALE_CFG)
    assert (art["classes"][CLASS_B]["n_pos"], art["classes"][CLASS_B]["n_neg"]) == (10, 19)


def test_row_order_does_not_change_the_fit():
    rows = _rows()
    assert fit_relevance_rows(rows, CLASSES) == fit_relevance_rows(rows[::-1], CLASSES)


def test_unlabelled_or_unscored_images_are_listed_not_silently_used(tmp_path):
    records, labels, manifest = _records(tmp_path, n=31)
    labels["images"] = [i for i in labels["images"] if i["vad_image_id"] != "img30"]
    labels["annotations"] = [a for a in labels["annotations"] if a["image_id"] != 31]
    art = fit_relevance(records, labels, manifest, SCALE_CFG)
    assert art["excluded"]["unlabelled"] == ["img30"]


def test_cli_fit_relevance_writes_the_artifact_and_never_overwrites(tmp_path):
    records, labels, manifest = _records(tmp_path)
    paths = {name: tmp_path / f"{name}.json" for name in ("labels", "split", "cal")}
    paths["labels"].write_text(json.dumps(labels), encoding="utf-8")
    paths["split"].write_text(json.dumps(manifest), encoding="utf-8")
    trace = tmp_path / "route_trace.jsonl"
    trace.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(SCALE_CFG.model_dump(mode="json")), encoding="utf-8")
    args = ["fit-relevance", "--config", str(cfg), "--trace", str(trace), "--labels", str(paths["labels"]),
            "--split-manifest", str(paths["split"]), "--out", str(paths["cal"])]
    assert cli.main(args) == 0
    assert json.loads(paths["cal"].read_text(encoding="utf-8"))["version"] == "relevance-calibrator-v1"
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2
