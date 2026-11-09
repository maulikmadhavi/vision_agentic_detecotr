"""A2 stage 3a: the split is enforced at evaluate, shadow export and fitting (synthetic inputs only)."""

from __future__ import annotations

import json

import pytest
import yaml

from routing import cli
from routing.evaluation import IsolationError, assert_split_allows, lock_frozen_set
from routing.sample import make_split

from .helpers import config, make_fixture, task
from .test_split import _inputs


def _split_files(tmp_path, labels=None):
    groups, exclusions, base_labels = _inputs()
    labels = labels or base_labels
    manifest, frozen = make_split(groups, exclusions, labels, task(), seed=11)
    frozen_ids = [i for i, img in manifest["images"].items() if img["split"] == "frozen_eval"]
    frozen_groups = [g for g, v in manifest["groups"].items() if v["split"] == "frozen_eval"]
    lock = lock_frozen_set(frozen_ids, frozen_groups, frozen, tmp_path / "frozen_lock.json")
    manifest["frozen_lock_sha256"] = lock["lock_sha256"]
    paths = {"manifest": tmp_path / "split_manifest.json", "lock": tmp_path / "frozen_lock.json",
             "labels": tmp_path / "labels.json", "trace": tmp_path / "route_trace.jsonl",
             "config": tmp_path / "config.yaml"}
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    paths["labels"].write_text(json.dumps(labels), encoding="utf-8")
    paths["trace"].write_text("", encoding="utf-8")  # no results: enough to check which images are evaluated
    paths["config"].write_text(yaml.safe_dump(config().model_dump(mode="json")), encoding="utf-8")
    return paths, manifest, labels, frozen_ids


def _evaluate(paths, out, *extra):
    return cli.main(["evaluate", "--config", str(paths["config"]), "--labels", str(paths["labels"]),
                     "--trace", str(paths["trace"]), "--out", str(out), *extra])


# --- (1) evaluate on the frozen set -------------------------------------------------------------


def test_evaluate_with_a_verified_lock_scores_only_the_frozen_images(tmp_path):
    paths, manifest, _, frozen_ids = _split_files(tmp_path)
    out = tmp_path / "metrics.json"
    assert _evaluate(paths, out, "--split-manifest", str(paths["manifest"]), "--frozen-lock", str(paths["lock"])) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["images"]["labelled"] == len(frozen_ids) == 10
    assert doc["split"] == {"status": "verified frozen_eval", "frozen_images": 10,
                            "lock_sha256": manifest["frozen_lock_sha256"]}


def test_evaluate_without_a_split_says_it_is_unverified(tmp_path):
    paths, *_ = _split_files(tmp_path)
    out = tmp_path / "metrics.json"
    assert _evaluate(paths, out) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["split"]["status"].startswith("unverified") and doc["images"]["labelled"] == 32


def test_evaluate_refuses_edited_frozen_labels(tmp_path, capsys):
    paths, _, labels, frozen_ids = _split_files(tmp_path)
    edited = json.loads(json.dumps(labels))
    frozen_image = next(i["id"] for i in edited["images"] if i["vad_image_id"] == frozen_ids[0])
    edited["annotations"].append({"id": 999, "image_id": frozen_image, "category_id": 2, "bbox": [1, 1, 5, 5]})
    paths["labels"].write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _evaluate(paths, tmp_path / "m.json", "--split-manifest", str(paths["manifest"]), "--frozen-lock", str(paths["lock"]))
    assert exc.value.code == 2 and "frozen" in capsys.readouterr().err and not (tmp_path / "m.json").exists()


def test_evaluate_refuses_a_lock_that_belongs_to_another_split(tmp_path, capsys):
    paths, manifest, *_ = _split_files(tmp_path)
    paths["manifest"].write_text(json.dumps({**manifest, "frozen_lock_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _evaluate(paths, tmp_path / "m.json", "--split-manifest", str(paths["manifest"]), "--frozen-lock", str(paths["lock"]))
    assert exc.value.code == 2 and "does not belong" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--split-manifest", "--frozen-lock"])
def test_evaluate_needs_both_split_flags_or_neither(tmp_path, capsys, flag):
    paths, *_ = _split_files(tmp_path)
    value = paths["manifest"] if flag == "--split-manifest" else paths["lock"]
    with pytest.raises(SystemExit) as exc:
        _evaluate(paths, tmp_path / "m.json", flag, str(value))
    assert exc.value.code == 2 and "together" in capsys.readouterr().err


# --- (2) shadow tags the export with the split ---------------------------------------------------


def _shadow_split(tmp_path, splits):
    fx = make_fixture(tmp_path / "fx")
    manifest = {"version": "split-v1", "images": {i: {"split": s} for i, s in splits.items()}}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    args = ["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
            "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
            "--split-manifest", str(path), "--out", str(tmp_path / "out")]
    return args


def test_shadow_export_tags_each_image_with_its_split(tmp_path):
    splits = {"easy": "calibration", "zero": "development", "small": "frozen_eval", "corrupt": "excluded"}
    assert cli.main(_shadow_split(tmp_path, splits)) == 0
    doc = json.loads((tmp_path / "out" / "annotations.json").read_text(encoding="utf-8"))
    assert {img["vad_image_id"]: img["vad_split"] for img in doc["images"]} == {
        "easy": "calibration", "zero": "development", "small": "frozen_eval"}


def test_shadow_refuses_images_missing_from_the_split_before_any_output(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(_shadow_split(tmp_path, {"easy": "calibration", "zero": "development", "corrupt": "excluded"}))
    assert exc.value.code == 2 and "small" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_shadow_without_a_split_exports_no_split_tag(tmp_path):
    fx = make_fixture(tmp_path / "fx")
    assert cli.main(["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
                     "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
                     "--out", str(tmp_path / "out")]) == 0
    doc = json.loads((tmp_path / "out" / "annotations.json").read_text(encoding="utf-8"))
    assert all("vad_split" not in img for img in doc["images"])


# --- (3) the fitting guard ----------------------------------------------------------------------


def test_fitting_guard_rejects_a_frozen_image(tmp_path):
    _, manifest, _, frozen_ids = _split_files(tmp_path)
    calibration = [i for i, img in manifest["images"].items() if img["split"] == "calibration"]
    assert_split_allows(manifest, calibration, allowed_splits=("calibration",))
    with pytest.raises(IsolationError, match="frozen"):
        assert_split_allows(manifest, calibration + frozen_ids[:1])


def test_fitting_guard_rejects_unknown_and_disallowed_images(tmp_path):
    _, manifest, *_ = _split_files(tmp_path)
    development = [i for i, img in manifest["images"].items() if img["split"] == "development"]
    with pytest.raises(IsolationError, match="not in the split manifest"):
        assert_split_allows(manifest, ["nope"])
    with pytest.raises(IsolationError, match="development"):
        assert_split_allows(manifest, development[:1], allowed_splits=("calibration",))


def test_fitting_guard_rejects_any_image_of_a_frozen_group(tmp_path):
    _, manifest, *_ = _split_files(tmp_path)
    frozen_group = next(g for g, v in manifest["groups"].items() if v["split"] == "frozen_eval")
    sneaky = {**manifest, "images": {**manifest["images"], "late-copy": {"group_id": frozen_group,
                                                                          "split": "calibration"}}}
    with pytest.raises(IsolationError, match="frozen"):
        assert_split_allows(sneaky, ["late-copy"])
