"""Stage (3a): `evaluate` writes an immutable, versioned run-metrics-v1 file (synthetic fixtures only)."""

from __future__ import annotations

import importlib.metadata
import json

import pytest

from routing import cli
from routing.metrics import (
    ImageResult, evaluate, map_from_outcomes, mean_average_precision, per_image_outcomes,
)
from routing.schemas import canonical_json, stable_id

from .helpers import CLASS_A, CLASS_B, config, make_fixture, task
from .test_metrics import _labels_for_fixture, _p, _report_fixture, _set

CLASSES = task().class_ids


def _shadow(tmp_path, cfg_path=None):
    fx = make_fixture(tmp_path / "fx")
    out = tmp_path / "run"
    assert cli.main(["shadow", "--config", str(cfg_path or fx["config"]), "--images", str(fx["manifest"]),
                     "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
                     "--relevance-scores", str(fx["relevance"]), "--out", str(out)]) == 0
    return fx, out / "route_trace.jsonl"


def _evaluate(fx, trace, out, *extra):
    return cli.main(["evaluate", "--config", str(fx["config"]), "--labels", str(_labels_for_fixture(fx["root"], "human")),
                     "--trace", str(trace), "--out", str(out), *extra])


def _doc(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- the file -------------------------------------------------------------------------------------------


def test_evaluate_writes_a_versioned_file_whose_id_is_its_own_content_digest(tmp_path, capsys):
    fx, trace = _shadow(tmp_path)
    assert _evaluate(fx, trace, tmp_path / "m.json") == 0
    doc = _doc(tmp_path / "m.json")
    assert doc["version"] == "run-metrics-v1"
    body = {k: v for k, v in doc.items() if k != "metrics_id"}
    assert doc["metrics_id"] == stable_id("met", body)
    assert {"run_id", "parent_id", "runs", "mixed_runs", "series", "seed", "per_image", "split", "detector",
            "routed", "cost"} <= set(doc)
    assert doc["parent_id"] is None and doc["mixed_runs"] is False and doc["seed"] == config().audit.seed


def test_evaluate_refuses_to_overwrite_an_existing_metrics_file(tmp_path, capsys):
    fx, trace = _shadow(tmp_path)
    out = tmp_path / "m.json"
    out.write_text("earlier", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _evaluate(fx, trace, out)
    assert exc.value.code == 2 and "already exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "earlier"


def test_the_same_inputs_give_a_byte_identical_file(tmp_path):
    fx, trace = _shadow(tmp_path)
    assert _evaluate(fx, trace, tmp_path / "a.json") == 0 and _evaluate(fx, trace, tmp_path / "b.json") == 0
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


# --- identity -------------------------------------------------------------------------------------------


def test_run_id_and_artifacts_come_from_the_records_run_info(tmp_path):
    fx, trace = _shadow(tmp_path)
    assert _evaluate(fx, trace, tmp_path / "m.json") == 0
    doc = _doc(tmp_path / "m.json")
    runs = sorted({canonical_json(json.loads(line)["run"]) for line in trace.read_text().splitlines()})
    assert doc["runs"] == [json.loads(r) for r in runs] and len(runs) == 1
    assert doc["run_id"] == stable_id("run", doc["runs"]) and doc["runs"][0]["primary_model"] == "prim@1"


def test_a_cohort_mixing_two_runs_is_marked(tmp_path):
    fx, trace = _shadow(tmp_path)
    lines = trace.read_text().splitlines()
    other = json.loads(lines[0])
    other["run"]["primary_model"] = "prim@2"
    trace.write_text("\n".join([json.dumps(other), *lines[1:]]) + "\n", encoding="utf-8")
    assert _evaluate(fx, trace, tmp_path / "m.json") == 0
    doc = _doc(tmp_path / "m.json")
    assert doc["mixed_runs"] is True and len(doc["runs"]) == 2
    # R6-2: still descriptive, but never eligible as evidence, with the reason named
    assert doc["evidence_status"].startswith("NOT A FINDING") and "mixes 2 runs" in doc["evidence_status"]


# --- R6-2: the config given to evaluate must be the one the shadow run used ---------------------------


def _config_file(tmp_path, fx, change):
    import yaml
    data = yaml.safe_load(fx["config"].read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "other_config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _changed_seed(data):
    data["audit"]["seed"] = "another-seed"


def _changed_prompt(data):  # the same class ids, so the labels still load
    data["task"]["classes"][0]["positive_prompts"] = ["a blue X9 widget, close up"]


def _overlapping_task(data):  # shares a class name with the run's task, adds another
    data["task"]["classes"] = [data["task"]["classes"][0],
                               {**data["task"]["classes"][1], "class_id": "other_class", "display_name": "Other"}]


@pytest.mark.parametrize("change", [_changed_seed, _changed_prompt, _overlapping_task],
                         ids=["seed", "prompt", "overlapping task"])
def test_evaluate_refuses_a_config_the_shadow_run_did_not_use(tmp_path, capsys, change):
    fx, trace = _shadow(tmp_path)
    other = _config_file(tmp_path, fx, change)
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", "--config", str(other), "--labels", str(_labels_for_fixture(fx["root"], "human")),
                  "--trace", str(trace), "--out", str(tmp_path / "m.json")])
    assert exc.value.code == 2 and "does not match the run" in capsys.readouterr().err
    assert not (tmp_path / "m.json").exists()


def test_the_config_file_shadow_overrode_is_accepted(tmp_path):
    fx, trace = _shadow(tmp_path)  # the fixture config has routing_enabled: false; shadow forced it on
    assert _evaluate(fx, trace, tmp_path / "m.json") == 0


def test_parent_records_the_parent_files_metrics_id_after_verifying_it(tmp_path, capsys):
    fx, trace = _shadow(tmp_path)
    assert _evaluate(fx, trace, tmp_path / "parent.json") == 0
    assert _evaluate(fx, trace, tmp_path / "child.json", "--parent", str(tmp_path / "parent.json")) == 0
    assert _doc(tmp_path / "child.json")["parent_id"] == _doc(tmp_path / "parent.json")["metrics_id"]
    tampered = _doc(tmp_path / "parent.json")
    tampered["seed"] = tampered["seed"] + "-edited"
    (tmp_path / "edited.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _evaluate(fx, trace, tmp_path / "c2.json", "--parent", str(tmp_path / "edited.json"))
    assert exc.value.code == 2 and "--parent" in capsys.readouterr().err and not (tmp_path / "c2.json").exists()


def test_the_series_key_pins_cohort_ids_labels_and_evaluators(tmp_path):
    fx, trace = _shadow(tmp_path)
    assert _evaluate(fx, trace, tmp_path / "m.json") == 0
    series = _doc(tmp_path / "m.json")["series"]
    cohort = sorted(["easy", "zero", "small", "corrupt"])
    assert series["cohort_ids_sha256"] == stable_id("ids", cohort, 64).split("-", 1)[1]
    assert len(series["labels_sha256"]) == 64 and series["images_sha256"] is None  # no split: no content binding
    from routing.coco_eval import SETTINGS
    assert (series["class_ids"], series["iou_threshold"], series["coco_settings"]) == (list(CLASSES), 0.5, SETTINGS)
    assert series["evaluators"] == {"custom": "custom-vad-1",
                                    "coco": "pycocotools " + importlib.metadata.version("pycocotools"),
                                    "routed_rule": "routed-score-v1"}


def test_the_series_binds_image_content_through_a_split(tmp_path):
    from .test_split_enforcement import _frozen_and_development_records, _metrics
    paths, frozen, _ = _frozen_and_development_records(tmp_path)
    doc = _metrics(paths, tmp_path, [frozen], "m.json")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    cohort = sorted(i for i, img in manifest["images"].items() if img["split"] == "frozen_eval")
    content = [[i, manifest["images"][i]["sha256"]] for i in cohort]
    assert doc["series"]["images_sha256"] == stable_id("img", content, 64).split("-", 1)[1]
    assert doc["split"]["manifest_sha256"] is not None
    assert doc["per_image"][frozen.image.id]["group_id"] == manifest["images"][frozen.image.id]["group_id"]


# --- per-image outcomes ----------------------------------------------------------------------------------


def test_per_image_outcomes_hold_ids_scores_and_match_codes_only(tmp_path):
    fx, trace = _shadow(tmp_path)
    assert _evaluate(fx, trace, tmp_path / "m.json") == 0
    per_image = _doc(tmp_path / "m.json")["per_image"]
    assert set(per_image) == {"easy", "zero", "small", "corrupt"}
    assert per_image["corrupt"]["result"] is False and per_image["easy"]["result"] is True
    assert set(per_image["easy"]) == {"result", "group_id", "primary", "routed_all", "routed_accepted"}
    text = json.dumps(per_image)
    assert "box" not in text and "uri" not in text and "sha256" not in text


def test_stored_outcomes_reproduce_the_custom_map_exactly():
    gt = _set(["p1", "p2", "neg", "fail"], [
        ("p1", 1, [0, 0, 40, 40], {}), ("p1", 2, [100, 100, 20, 20], {}),
        ("p1", 1, [150, 150, 30, 30], {"ignore": True}), ("p2", 1, [0, 0, 120, 120], {}), ("fail", 1, [50, 50, 40, 40], {})])
    results = {
        "p1": ImageResult((_p(CLASS_A, (0, 0, 40, 40), 0.9), _p(CLASS_A, (150, 150, 180, 180), 0.95),
                           _p(CLASS_B, (100, 100, 121, 121), 0.6), _p(CLASS_A, (60, 60, 90, 90), 0.4)), (), False),
        "p2": ImageResult((_p(CLASS_A, (0, 0, 120, 110), 0.8),), (), True),
        "neg": ImageResult((_p(CLASS_B, (10, 10, 30, 30), 0.7),), (), False),
    }
    outcomes = per_image_outcomes(gt, results, CLASSES)
    expected = mean_average_precision({i: r.detections for i, r in results.items()}, gt, CLASSES)
    got = map_from_outcomes([outcomes[i]["primary"] for i in sorted(outcomes)], CLASSES)
    assert got["map50_95"] == pytest.approx(expected["map50_95"], abs=1e-15)
    assert got["map50"] == pytest.approx(expected["map50"], abs=1e-15)
    assert got["per_class"] == expected["per_class"]


def test_stored_routed_outcomes_reproduce_the_routed_custom_map(tmp_path):
    from .test_routed_score import _crafted
    from routing.metrics import image_results
    rec = _crafted(tmp_path)
    gt = _set([rec.image.id], [(rec.image.id, 1, [100, 100, 40, 40], {})])
    results = image_results([rec])
    routed = evaluate(gt, CLASSES, results)["routed"]
    outcomes = per_image_outcomes(gt, results, CLASSES)
    for source, block in (("routed_all", "all_unrejected"), ("routed_accepted", "accepted_only")):
        got = map_from_outcomes([outcomes[i][source] for i in sorted(outcomes)], CLASSES)
        assert got["map50"] == pytest.approx(routed[block]["custom_map50"], abs=1e-15)
    assert outcomes[rec.image.id]["routed_all"]["unscored"] == 1


def test_primary_outcomes_count_unscored_detections():
    gt, results = _report_fixture()
    results["p2"] = ImageResult(detections=(_p(CLASS_A, (0, 0, 120, 110), None),), accepted=(), flagged_for_review=True)
    assert per_image_outcomes(gt, results, CLASSES)["p2"]["primary"]["unscored"] == 1
