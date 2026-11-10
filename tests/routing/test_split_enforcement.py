"""A2 stage 3a: the split is enforced at evaluate, shadow export and fitting (synthetic inputs only)."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
import yaml
from PIL import Image

from routing import cli
from routing.evaluation import IsolationError, assert_split_allows, lock_frozen_set
from routing.image_inputs import ManifestEntry
from routing.pipeline import RoutingPipeline
from routing.sample import make_split
from routing.schemas import canonical_json
from routing.store import RouteStore

from .helpers import ScriptedDetector, config, make_fixture, task
from .test_split import _inputs, content_of


def _split_files(tmp_path, labels=None, content=None):
    """``content`` ({image id: {sha256, width, height}}) replaces groups-file entries, as real files would give."""
    groups, exclusions, base_labels = _inputs()
    labels = labels or base_labels
    for image_id, fields in (content or {}).items():
        groups["images"][image_id].update(fields)
    manifest, frozen = make_split(groups, exclusions, labels, task(), seed=11)
    frozen_ids = [i for i, img in manifest["images"].items() if img["split"] == "frozen_eval"]
    frozen_groups = [g for g, v in manifest["groups"].items() if v["split"] == "frozen_eval"]
    lock = lock_frozen_set(frozen_ids, frozen_groups, frozen, tmp_path / "frozen_lock.json",
                           content=content_of(manifest, frozen_ids))
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
                            "lock_sha256": manifest["frozen_lock_sha256"],
                            "image_content": "bound: each frozen result matches the lock's sha256 and size",
                            "manifest_sha256": doc["split"]["manifest_sha256"]}
    assert len(doc["split"]["manifest_sha256"]) == 64


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


# --- R5-2: frozen results are bound to the frozen image content ----------------------------------


def _frozen_image_files(tmp_path):
    """A white 100x100 file for the first frozen image; the split and lock are made with its real sha256."""
    _, _, _, frozen_ids = _split_files(tmp_path / "probe")
    target = sorted(frozen_ids)[0]
    white = tmp_path / "white.png"
    Image.new("RGB", (100, 100), (255, 255, 255)).save(white)
    content = {target: {"sha256": hashlib.sha256(white.read_bytes()).hexdigest(), "width": 100, "height": 100}}
    paths, manifest, labels, frozen_ids = _split_files(tmp_path, content=content)
    return paths, target, white


def _trace(paths, image_path, image_id):
    record = RoutingPipeline(config(), ScriptedDetector({}), RouteStore(":memory:")).process(
        ManifestEntry(str(image_path), image_id=image_id))
    paths["trace"].write_text(record.model_dump_json() + "\n", encoding="utf-8")


def _evaluate_frozen(paths, tmp_path):
    return _evaluate(paths, tmp_path / "m.json", "--split-manifest", str(paths["manifest"]),
                     "--frozen-lock", str(paths["lock"]))


def test_a_frozen_result_for_the_locked_image_bytes_verifies(tmp_path):
    paths, target, white = _frozen_image_files(tmp_path)
    _trace(paths, white, target)
    assert _evaluate_frozen(paths, tmp_path) == 0
    assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))["split"]["status"] == "verified frozen_eval"


def test_identical_bytes_at_another_path_still_verify(tmp_path):
    paths, target, white = _frozen_image_files(tmp_path)
    moved = tmp_path / "elsewhere" / "copy.png"
    moved.parent.mkdir()
    shutil.copy(white, moved)
    _trace(paths, moved, target)
    assert _evaluate_frozen(paths, tmp_path) == 0


def test_a_replaced_frozen_image_with_the_same_id_and_size_is_refused(tmp_path, capsys):
    """The reviewer's case: white replaced by red, same 100x100 size, same image id."""
    paths, target, _ = _frozen_image_files(tmp_path)
    red = tmp_path / "red.png"
    Image.new("RGB", (100, 100), (255, 0, 0)).save(red)
    _trace(paths, red, target)
    with pytest.raises(SystemExit) as exc:
        _evaluate_frozen(paths, tmp_path)
    assert exc.value.code == 2 and f"{target}" in capsys.readouterr().err and not (tmp_path / "m.json").exists()


def test_a_frozen_result_of_another_size_is_refused_even_with_the_same_sha256(tmp_path, capsys):
    _, _, _, frozen_ids = _split_files(tmp_path / "probe")
    target = sorted(frozen_ids)[0]
    tall = tmp_path / "tall.png"
    Image.new("RGB", (100, 120), (255, 255, 255)).save(tall)
    sha = hashlib.sha256(tall.read_bytes()).hexdigest()  # the lock claims these bytes at 100x100
    paths, *_ = _split_files(tmp_path, content={target: {"sha256": sha, "width": 100, "height": 100}})
    _trace(paths, tall, target)
    with pytest.raises(SystemExit) as exc:
        _evaluate_frozen(paths, tmp_path)
    assert exc.value.code == 2 and "height" in capsys.readouterr().err


def test_a_manifest_whose_image_content_differs_from_the_lock_is_refused(tmp_path, capsys):
    paths, manifest, *_ = _split_files(tmp_path)
    frozen = next(i for i, img in manifest["images"].items() if img["split"] == "frozen_eval")
    manifest["images"][frozen]["sha256"] = "0" * 64
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _evaluate_frozen(paths, tmp_path)
    assert exc.value.code == 2 and "does not belong" in capsys.readouterr().err


def test_an_old_lock_without_image_hashes_is_reported_unverified(tmp_path):
    paths, manifest, *_ = _split_files(tmp_path)
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    old = {k: lock[k] for k in ("image_ids", "group_ids", "annotations_sha256")}
    old["lock_sha256"] = hashlib.sha256(canonical_json(old).encode()).hexdigest()
    paths["lock"].write_text(json.dumps(old), encoding="utf-8")
    paths["manifest"].write_text(json.dumps({**manifest, "frozen_lock_sha256": old["lock_sha256"]}), encoding="utf-8")
    assert _evaluate_frozen(paths, tmp_path) == 0
    status = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))["split"]["status"]
    assert status.startswith("unverified") and "image content" in status and "frozen-lock-v2" in status
    assert "version" not in json.loads(paths["lock"].read_text(encoding="utf-8"))  # never upgraded


# --- R6-1: old split files give a descriptive, unverified report (no crash, nothing fabricated) ---------


def _old_files(tmp_path, manifest_version):
    """An old (pre-image-hash) lock, with either a genuine split-v1 manifest or the split-v2 one."""
    paths, manifest, *_ = _split_files(tmp_path)
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    old = {k: lock[k] for k in ("image_ids", "group_ids", "annotations_sha256")}
    old["lock_sha256"] = hashlib.sha256(canonical_json(old).encode()).hexdigest()
    if manifest_version == "split-v1":
        manifest = {**manifest, "version": "split-v1",
                    "images": {i: {k: v for k, v in img.items() if k not in CONTENT}
                               for i, img in manifest["images"].items()}}
    paths["lock"].write_text(json.dumps(old), encoding="utf-8")
    paths["manifest"].write_text(json.dumps({**manifest, "frozen_lock_sha256": old["lock_sha256"]}), encoding="utf-8")
    record = _record(tmp_path, sorted(old["image_ids"])[0])
    paths["trace"].write_text(record.model_dump_json() + "\n", encoding="utf-8")
    return paths, record


CONTENT = ("sha256", "width", "height")


@pytest.mark.parametrize("manifest_version", ["split-v1", "split-v2"])
def test_an_old_lock_gives_an_unverified_report_without_an_image_digest(tmp_path, manifest_version):
    paths, record = _old_files(tmp_path, manifest_version)
    before = {name: paths[name].read_bytes() for name in ("manifest", "lock", "labels", "trace")}
    assert _evaluate_frozen(paths, tmp_path) == 0
    doc = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert doc["split"]["status"].startswith("unverified") and doc["evidence_status"].startswith("NOT A FINDING")
    assert doc["series"]["images_sha256"] is None  # never derived from an unverified split
    manifest = json.loads(before["manifest"])
    assert doc["per_image"][record.image.id]["group_id"] == manifest["images"][record.image.id]["group_id"]
    assert {name: paths[name].read_bytes() for name in before} == before  # inputs untouched


def _record(tmp_path, name, detector=None):
    path = tmp_path / f"{name}.png"
    Image.new("RGB", (100, 100), (255, 255, 255)).save(path)
    return RoutingPipeline(config(), detector or ScriptedDetector({}), RouteStore(":memory:")).process(
        ManifestEntry(str(path), image_id=name))


@pytest.mark.parametrize("how", ["identical", "conflicting"])
def test_image_results_refuse_a_repeated_image_instead_of_keeping_the_last(tmp_path, how):
    from routing.interfaces import RawDetection
    from routing.metrics import LabelError, image_results
    first = _record(tmp_path, "p0-0")
    again = first if how == "identical" else _record(
        tmp_path, "p0-0", ScriptedDetector({"p0-0": [RawDetection("blue_widget_x9", (10, 10, 30, 30), 0.9)]}))
    with pytest.raises(LabelError, match=r"repeated image ids.*\['p0-0'\]"):
        image_results([first, _record(tmp_path, "p1-0"), again])


def test_evaluate_refuses_a_trace_with_a_repeated_image_before_writing(tmp_path, capsys):
    paths, *_ = _split_files(tmp_path)
    record = _record(tmp_path, "p0-0")
    paths["trace"].write_text((record.model_dump_json() + "\n") * 2, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _evaluate(paths, tmp_path / "m.json")
    assert exc.value.code == 2 and "p0-0" in capsys.readouterr().err and not (tmp_path / "m.json").exists()


# --- R5-7: one cohort for accuracy and cost --------------------------------------------------------


def _frozen_and_development_records(tmp_path):
    paths, target, white = _frozen_image_files(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    development = sorted(i for i, img in manifest["images"].items() if img["split"] == "development")[0]
    frozen = RoutingPipeline(config(), ScriptedDetector({}), RouteStore(":memory:")).process(
        ManifestEntry(str(white), image_id=target))
    dev = _record(tmp_path, development)
    slow = dev.model_copy(update={"trace": dev.trace.model_copy(update={"elapsed_ms": 9e6})})
    return paths, frozen, slow


def _metrics(paths, tmp_path, records, name):
    paths["trace"].write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    assert _evaluate(paths, tmp_path / name, "--split-manifest", str(paths["manifest"]),
                     "--frozen-lock", str(paths["lock"])) == 0
    return json.loads((tmp_path / name).read_text(encoding="utf-8"))


def test_a_slow_development_record_changes_no_frozen_metric(tmp_path):
    """The reviewer's case: one frozen and one development result reported two cost images."""
    paths, frozen, slow = _frozen_and_development_records(tmp_path)
    alone = _metrics(paths, tmp_path, [frozen], "alone.json")
    mixed = _metrics(paths, tmp_path, [frozen, slow], "mixed.json")
    assert mixed.pop("out_of_cohort") == {"results": 1, "image_ids": [slow.image.id]}
    assert alone.pop("out_of_cohort") == {"results": 0, "image_ids": []}
    mixed.pop("metrics_id"), alone.pop("metrics_id")  # the file's own digest covers out_of_cohort
    assert mixed == alone and mixed["cost"]["images"] == 1


def test_labelled_images_without_a_result_stay_operational_failures_of_the_cohort(tmp_path):
    paths, frozen, slow = _frozen_and_development_records(tmp_path)
    doc = _metrics(paths, tmp_path, [frozen, slow], "m.json")
    assert doc["images"]["labelled"] == 10 and doc["images"]["with_result"] == 1
    assert len(doc["images"]["operational_failures"]) == 9 and slow.image.id not in doc["images"]["operational_failures"]


def test_without_a_split_the_cohort_is_every_labelled_image(tmp_path):
    paths, *_ = _split_files(tmp_path)
    labelled, unlabelled = _record(tmp_path, "p0-0"), _record(tmp_path, "not-labelled")
    paths["trace"].write_text(labelled.model_dump_json() + "\n" + unlabelled.model_dump_json() + "\n",
                              encoding="utf-8")
    assert _evaluate(paths, tmp_path / "m.json") == 0
    doc = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert doc["cost"]["images"] == 1 and doc["out_of_cohort"] == {"results": 1, "image_ids": ["not-labelled"]}


# --- S6: evidence needs human labels AND a verified frozen split -----------------------------------


def _human(paths):
    labels = json.loads(paths["labels"].read_text(encoding="utf-8"))
    paths["labels"].write_text(json.dumps({**labels, "vad_label_provenance": "human"}), encoding="utf-8")


def test_human_labels_on_a_verified_frozen_split_are_evidence(tmp_path):
    paths, frozen, _ = _frozen_and_development_records(tmp_path)
    _human(paths)
    doc = _metrics(paths, tmp_path, [frozen], "m.json")
    assert doc["evidence_status"] == "evidence: human labels on a verified frozen split"


def test_human_labels_with_an_old_lock_are_not_a_finding_but_metrics_are_kept(tmp_path):
    paths, manifest, *_ = _split_files(tmp_path)
    _human(paths)
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    old = {k: lock[k] for k in ("image_ids", "group_ids", "annotations_sha256")}
    old["lock_sha256"] = hashlib.sha256(canonical_json(old).encode()).hexdigest()
    paths["lock"].write_text(json.dumps(old), encoding="utf-8")
    paths["manifest"].write_text(json.dumps({**manifest, "frozen_lock_sha256": old["lock_sha256"]}), encoding="utf-8")
    assert _evaluate_frozen(paths, tmp_path) == 0
    doc = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert doc["evidence_status"].startswith("NOT A FINDING") and "predates image hashes" in doc["evidence_status"]
    assert "vad_label_provenance" not in doc["evidence_status"] and "detector" in doc and "cost" in doc


@pytest.mark.parametrize("flag", ["--split-manifest", "--frozen-lock"])
def test_evaluate_needs_both_split_flags_or_neither(tmp_path, capsys, flag):
    paths, *_ = _split_files(tmp_path)
    value = paths["manifest"] if flag == "--split-manifest" else paths["lock"]
    with pytest.raises(SystemExit) as exc:
        _evaluate(paths, tmp_path / "m.json", flag, str(value))
    assert exc.value.code == 2 and "together" in capsys.readouterr().err


# --- (2) shadow tags the export with the split ---------------------------------------------------


def _shadow_split(tmp_path, splits, version="split-v1"):
    fx = make_fixture(tmp_path / "fx")
    manifest = {"version": version, "images": {i: {"split": s} for i, s in splits.items()}}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    args = ["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
            "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
            "--split-manifest", str(path), "--out", str(tmp_path / "out")]
    return args


@pytest.mark.parametrize("version", ["split-v1", "split-v2"])
def test_shadow_export_tags_each_image_with_its_split(tmp_path, version):
    splits = {"easy": "calibration", "zero": "development", "small": "frozen_eval", "corrupt": "excluded"}
    assert cli.main(_shadow_split(tmp_path, splits, version)) == 0
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


# --- A7 stage 3: the training-side reader refuses frozen images ---------------------------------


def _export(tmp_path, splits):
    from routing.export import export_routed
    fx = make_fixture(tmp_path / "fx")
    assert cli.main(["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
                     "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
                     "--out", str(tmp_path / "run")]) == 0
    from routing.schemas import RouteRecord
    lines = (tmp_path / "run" / "route_trace.jsonl").read_text(encoding="utf-8").splitlines()
    records = [RouteRecord.model_validate_json(line) for line in lines]
    path = tmp_path / "annotations.json"
    export_routed(records, records[0].task, path, splits)
    return path


def test_training_reader_refuses_images_tagged_frozen(tmp_path):
    from routing.export import load_annotations
    path = _export(tmp_path, {"easy": "calibration", "zero": "development", "small": "frozen_eval"})
    with pytest.raises(IsolationError, match="small"):
        load_annotations(path)


def test_training_reader_accepts_tuning_tags_and_untagged_exports(tmp_path):
    from routing.export import load_annotations
    assert load_annotations(_export(tmp_path / "t", {"easy": "calibration", "zero": "development",
                                                     "small": "calibration"}))
    assert load_annotations(_export(tmp_path / "u", None))


def test_training_reader_with_a_split_manifest_refuses_untagged_frozen_images(tmp_path):
    from routing.export import load_annotations
    path = _export(tmp_path, None)  # exported without --split-manifest: no tags
    manifest = {"version": "split-v1", "groups": {"g-s": {"split": "frozen_eval"}},
                "images": {"easy": {"group_id": "g-e", "split": "calibration"},
                           "small": {"group_id": "g-s", "split": "frozen_eval"}}}
    with pytest.raises(IsolationError, match="small"):
        load_annotations(path, split_manifest=manifest)
    late = {**manifest, "images": {"easy": {"group_id": "g-s", "split": "calibration"}}}  # a frozen group's copy
    with pytest.raises(IsolationError, match="easy"):
        load_annotations(path, split_manifest=late)
