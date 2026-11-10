"""Stage (3b): `vad-route compare` over run-metrics-v1 files of one series (synthetic fixtures only)."""

from __future__ import annotations

import json

import pytest
import yaml

from routing import cli
from routing.coco_eval import SETTINGS as COCO_SETTINGS
from routing.compare import CompareError, compare_runs
from routing.metrics import ImageResult, evaluate, per_image_outcomes
from routing.schemas import stable_id

from .helpers import CLASS_A, task
from .test_metrics import _p, _set

CLASSES = task().class_ids
IMAGES = [f"g{k}" for k in range(8)]
GT = _set(IMAGES, [(i, 1, [10, 10, 40, 40], {}) for i in IMAGES])
HIT, MISS = (10, 10, 50, 50), (100, 100, 140, 140)
RUN = {"config_hash": "c" * 16, "policy_version": "p1", "primary_model": "prim@1"}


def _doc(hits, *, verified=True, review_rate=0.2, fp_extra=0, edit=None):
    """A run-metrics-v1 document: image k is detected (score 0.9) when hits[k], else a 0.9 miss is reported."""
    results = {}
    for k, image_id in enumerate(IMAGES):
        dets = [_p(CLASS_A, HIT if hits[k] else MISS, 0.9)] + [_p(CLASS_A, MISS, 0.3)] * (fp_extra if k == 0 else 0)
        results[image_id] = ImageResult(tuple(dets), (), False, tuple((d, "accepted") for d in dets))
    metrics = evaluate(GT, CLASSES, results)
    per_image = per_image_outcomes(GT, results, CLASSES)
    for image_id in per_image:
        per_image[image_id]["group_id"] = f"grp-{image_id}" if verified else None
    body = {"version": "run-metrics-v1", "run_id": stable_id("run", [RUN]), "parent_id": None, "runs": [RUN],
            "mixed_runs": False, "seed": "audit-seed-1",
            "series": {"cohort": "development", "class_ids": list(CLASSES), "iou_threshold": 0.5,
                       "coco_settings": dict(COCO_SETTINGS), "cohort_ids_sha256": "a" * 64, "labels_sha256": "b" * 64,
                       "images_sha256": "d" * 64 if verified else None,
                       "evaluators": {"custom": "custom-vad-1", "coco": "pycocotools 2.0.11",
                                      "routed_rule": "routed-score-v1"}},
            "split": {"status": "verified frozen_eval" if verified else "unverified: no --split-manifest/--frozen-lock"},
            **metrics, "cost": {"review_rate": review_rate, "vlm_calls_per_image": 0.0, "latency_ms": {"p90": 10.0}},
            "per_image": per_image}
    if edit:
        edit(body)
    return {"version": "run-metrics-v1", "metrics_id": stable_id("met", body), **body}


def _write(tmp_path, name, doc):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


BASE = [True, True, True, True, False, False, False, False]
BETTER = [True] * 8


# --- deltas, interval and verdict --------------------------------------------------------------------


def test_deltas_in_percentage_points_for_both_evaluators_with_a_custom_interval():
    base = _doc(BASE)
    out = compare_runs([base, _doc(BETTER)], source="routed_all", against="previous", seed=3, resamples=300)
    coco, custom = out["deltas"]["coco"], out["deltas"]["custom"]
    base_map50 = base["routed"]["all_unrejected"]["coco"]["map50"]  # 51/101: 101-point interpolation
    assert coco["map50_pp"] == pytest.approx(100 * (1.0 - base_map50), abs=1e-9) and base_map50 == pytest.approx(51 / 101)
    assert custom["map50_pp"] == pytest.approx(coco["map50_pp"], abs=1e-9)
    low, high = custom["interval"]["map50_pp"]
    assert 0 < low <= custom["map50_pp"] <= high  # every resample improves
    assert custom["interval"]["seed"] == 3 and custom["interval"]["resamples"] == 300
    assert custom["interval"]["method"].startswith("paired group bootstrap") and out["verdict"] == "improved"


def test_an_interval_spanning_zero_is_no_detectable_change():
    mixed = [True, True, True, False, True, False, False, False]  # one image better, one worse
    out = compare_runs([_doc(BASE), _doc(mixed)], source="routed_all", against="previous", seed=1, resamples=300)
    low, high = out["deltas"]["custom"]["interval"]["map50_pp"]
    assert low < 0 < high and out["verdict"] == "no detectable change"


def test_an_unverified_series_is_only_a_descriptive_table_marked_not_a_finding():
    out = compare_runs([_doc(BASE, verified=False), _doc(BETTER, verified=False)], source="routed_all",
                       against="previous", seed=1, resamples=50)
    assert out["status"].startswith("NOT A FINDING") and out["verdict"].startswith("NOT A FINDING")
    assert out["deltas"]["custom"]["interval"].startswith("unavailable: the series is not content-verified")
    assert out["deltas"]["coco"]["map50_pp"] is not None  # still described
    with pytest.raises(CompareError, match="best.*content-verified"):
        compare_runs([_doc(BASE, verified=False), _doc(BETTER, verified=False)], source="routed_all",
                     against="best", guardrails=GUARD, seed=1)


def test_the_same_seed_gives_the_same_interval():
    runs = [_doc(BASE), _doc([True, True, True, False, True, True, False, False])]
    first = compare_runs(runs, source="routed_all", against="previous", seed=7, resamples=200)
    assert first == compare_runs(runs, source="routed_all", against="previous", seed=7, resamples=200)
    assert first["deltas"]["custom"]["interval"]["seed"] == 7


def test_opposite_signs_of_the_coco_and_custom_deltas_raise_a_visible_warning():
    def flip(body):
        body["routed"]["all_unrejected"]["coco"]["map50"] = 0.1  # coco says worse, custom says better
    out = compare_runs([_doc(BASE), _doc(BETTER, edit=flip)], source="routed_all", against="previous", seed=1,
                       resamples=50)
    assert any("opposite signs" in w for w in out["warnings"])


@pytest.mark.parametrize("source, block", [("primary", "detector"), ("routed_accepted", "accepted_only")])
def test_the_source_selects_which_output_is_compared(source, block):
    def boost(body):
        target = body["detector"] if block == "detector" else body["routed"]["accepted_only"]
        target["coco"]["map50"] = 0.75
    out = compare_runs([_doc(BASE), _doc(BASE, edit=boost)], source=source, against="previous", seed=1, resamples=20)
    assert out["source"] == source and out["deltas"]["coco"]["map50_pp"] == pytest.approx(75 - 5100 / 101, abs=1e-9)


# --- refusals --------------------------------------------------------------------------------------------


def test_files_of_another_series_are_refused_naming_the_fields():
    def other(body):
        body["series"]["labels_sha256"] = "e" * 64
        body["series"]["evaluators"]["coco"] = "pycocotools 2.0.12"
    with pytest.raises(CompareError, match=r"series.*evaluators\.coco.*labels_sha256"):
        compare_runs([_doc(BASE), _doc(BETTER, edit=other)], source="routed_all", against="previous", seed=1)


def test_a_file_mixing_runs_is_refused_naming_the_differing_fields():
    def mixed(body):
        body["runs"] = [RUN, {**RUN, "primary_model": "prim@2"}]
        body["mixed_runs"] = True
    with pytest.raises(CompareError, match="mixes runs.*primary_model"):
        compare_runs([_doc(BASE), _doc(BETTER, edit=mixed)], source="routed_all", against="previous", seed=1)


# --- guardrails and best eligible ------------------------------------------------------------------------


GUARD = {"version": "guardrails-v1", "max_recall_drop_pp": {"per_class": 5.0, "per_size": 100.0},
         "max_fp_per_image_increase": 0.5, "max_review_rate_increase_pp": 10.0,
         "max_vlm_calls_per_image_increase": 1.0, "max_latency_p90_increase_ms": 1000.0}


def test_a_candidate_failing_a_guardrail_is_listed_with_the_rule():
    out = compare_runs([_doc(BASE), _doc(BETTER, fp_extra=12)], source="routed_all", against="previous",
                       guardrails=GUARD, seed=1, resamples=20)
    assert out["guardrails"]["candidate"]["passed"] is False
    assert any(f.startswith("max_fp_per_image_increase") for f in out["guardrails"]["candidate"]["failed"])


def test_best_is_the_highest_coco_map_among_runs_that_pass_and_failures_are_listed():
    reference = _doc(BASE)
    strong_but_reviews_more = _doc(BETTER, review_rate=0.9)              # fails the review-rate guardrail
    passing = _doc([True, True, True, True, True, True, False, False])
    candidate = _doc(BETTER)
    out = compare_runs([reference, strong_but_reviews_more, passing, candidate], source="routed_all", against="best",
                       guardrails=GUARD, seed=1, resamples=20)
    assert out["baseline"] == passing["metrics_id"]
    failed = {r["metrics_id"]: r["failed"] for r in out["guardrails"]["runs"] if not r["eligible"]}
    assert list(failed) == [strong_but_reviews_more["metrics_id"]]
    assert failed[strong_but_reviews_more["metrics_id"]][0].startswith("max_review_rate_increase_pp")


def test_best_needs_a_guardrails_file():
    with pytest.raises(CompareError, match="--guardrails"):
        compare_runs([_doc(BASE), _doc(BETTER)], source="routed_all", against="best", seed=1)


# --- CLI ---------------------------------------------------------------------------------------------------


def _cli(tmp_path, *paths, extra=()):
    return cli.main(["compare", *map(str, paths), "--source", "routed_all", "--against", "previous", "--resamples",
                     "50", "--out", str(tmp_path / "cmp.json"), *extra])


def test_cli_writes_the_comparison_records_the_guardrails_digest_and_prints_both_deltas(tmp_path, capsys):
    import hashlib
    guard = tmp_path / "guard.yaml"
    guard.write_text(yaml.safe_dump(GUARD), encoding="utf-8")
    a, b = _write(tmp_path, "a", _doc(BASE)), _write(tmp_path, "b", _doc(BETTER))
    assert _cli(tmp_path, a, b, extra=("--guardrails", str(guard), "--seed", "5")) == 0
    out = json.loads((tmp_path / "cmp.json").read_text(encoding="utf-8"))
    assert out["version"] == "compare-v1" and out["guardrails"]["file_sha256"] == hashlib.sha256(guard.read_bytes()).hexdigest()
    assert out["guardrails"]["version"] == "guardrails-v1" and out["deltas"]["custom"]["interval"]["seed"] == 5
    line = capsys.readouterr().out
    assert "coco" in line and "custom-vad-1" in line and "pp" in line


def test_cli_refuses_a_modified_input_file(tmp_path, capsys):
    tampered = _doc(BETTER)
    tampered["cost"]["review_rate"] = 0.0
    a, b = _write(tmp_path, "a", _doc(BASE)), _write(tmp_path, "b", tampered)
    with pytest.raises(SystemExit) as exc:
        _cli(tmp_path, a, b)
    assert exc.value.code == 2 and "not an unmodified run-metrics-v1 file" in capsys.readouterr().err
    assert not (tmp_path / "cmp.json").exists()


def test_cli_refuses_to_overwrite_its_output(tmp_path, capsys):
    a, b = _write(tmp_path, "a", _doc(BASE)), _write(tmp_path, "b", _doc(BETTER))
    (tmp_path / "cmp.json").write_text("earlier", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _cli(tmp_path, a, b)
    assert exc.value.code == 2 and "already exists" in capsys.readouterr().err


def test_cli_compares_files_written_by_evaluate(tmp_path, capsys):
    from .test_run_metrics import _evaluate, _shadow
    fx, trace = _shadow(tmp_path)
    assert _evaluate(fx, trace, tmp_path / "m1.json") == 0 and _evaluate(fx, trace, tmp_path / "m2.json") == 0
    assert _cli(tmp_path, tmp_path / "m1.json", tmp_path / "m2.json") == 0
    out = json.loads((tmp_path / "cmp.json").read_text(encoding="utf-8"))
    assert out["deltas"]["custom"]["interval"].startswith("unavailable") and out["verdict"].startswith("NOT A FINDING")


# --- R6-4 additions ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("field, value", [("cohort", "frozen_eval"), ("iou_threshold", 0.4),
                                          ("class_ids", ["blue_widget_x9"]), ("coco_settings", {"max_detections": 10})])
def test_runs_differing_in_cohort_task_iou_or_coco_settings_are_not_one_series(field, value):
    def other(body):
        body["series"][field] = value
    with pytest.raises(CompareError, match=f"series.*{field}"):
        compare_runs([_doc(BASE), _doc(BETTER, edit=other)], source="routed_all", against="previous", seed=1)


def test_primary_and_the_selected_routed_variant_are_reported_separately_with_context():
    out = compare_runs([_doc(BASE), _doc(BETTER)], source="routed_accepted", against="previous", seed=1, resamples=20)
    assert set(out["primary"]) == {"coco", "custom"} and "interval" not in out["primary"]["custom"]
    assert "interval" in out["deltas"]["custom"]  # the interval belongs to the selected source only
    assert out["deltas"]["unscored_routed"] == {"baseline": 0, "candidate": 0}
    assert set(out["accepted_annotations"]) == {"baseline", "candidate", "precision_pp", "recall_pp"}


def test_comparing_primary_has_no_separate_primary_block_and_no_unscored_routed():
    out = compare_runs([_doc(BASE), _doc(BETTER)], source="primary", against="previous", seed=1, resamples=20)
    assert "primary" not in out and "unscored_routed" not in out["deltas"]


PLATEAU = {**GUARD, "min_meaningful_gain_pp": 5.0, "patience": 2}


def test_a_plateau_is_reported_as_an_operational_stop_rule():
    six, seven = [True] * 6 + [False] * 2, [True] * 7 + [False]
    runs = [_doc(BASE), _doc(six), _doc(six), _doc(seven)]  # +25 pp, then +0 and +12.5 pp: not a plateau yet
    assert compare_runs(runs, source="routed_all", against="previous", guardrails=PLATEAU, seed=1,
                        resamples=10)["plateau"]["stop_tuning"] is False
    flat = [_doc(BASE), _doc(six), _doc(six), _doc(six)]  # two runs in a row without a 5 pp gain over the best
    plateau = compare_runs(flat, source="routed_all", against="previous", guardrails=PLATEAU, seed=1,
                           resamples=10)["plateau"]
    assert plateau["stop_tuning"] is True and plateau["runs_without_gain"] == 2
    assert plateau["rule"].startswith("plateau: stop tuning") and "operational" in plateau["rule"]


@pytest.mark.parametrize("sequence, expected", [
    (["six", "six"], (False, 1)),                    # one run without a gain is below patience 2
    (["six", "six", "seven", "seven"], (False, 1)),  # a meaningful gain resets the count
])
def test_the_plateau_count_respects_patience_and_resets_on_a_meaningful_gain(sequence, expected):
    hits = {"six": [True] * 6 + [False] * 2, "seven": [True] * 7 + [False]}
    runs = [_doc(BASE)] + [_doc(hits[name]) for name in sequence]
    plateau = compare_runs(runs, source="routed_all", against="previous", guardrails=PLATEAU, seed=1,
                           resamples=10)["plateau"]
    assert (plateau["stop_tuning"], plateau["runs_without_gain"]) == expected


def test_no_plateau_rule_without_the_guardrail_keys():
    out = compare_runs([_doc(BASE), _doc(BETTER)], source="routed_all", against="previous", guardrails=GUARD,
                       seed=1, resamples=10)
    assert out["plateau"] is None


def test_an_unknown_guardrail_key_is_refused():
    with pytest.raises(CompareError, match="guardrails"):
        compare_runs([_doc(BASE), _doc(BETTER)], source="routed_all", against="previous",
                     guardrails={**GUARD, "max_anything": 1}, seed=1)
