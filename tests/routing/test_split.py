"""A2 stage 2: seeded group-wise split, split manifest and frozen lock (synthetic inputs only)."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from routing import cli
from routing.evaluation import IsolationError, check_isolation, load_frozen, lock_frozen_set
from routing.sample import SampleError, exclusions_digest, frozen_label_subset, make_split
from routing.schemas import canonical_json

from .helpers import CLASS_A, CLASS_B, config, task

CATS = [{"id": 1, "name": CLASS_A}, {"id": 2, "name": CLASS_B}]


def _inputs(n_pos=10, n_neg=6, extra_images=(), excluded=()):
    """Groups g-p0.. (positive) and g-n0.. (negative); each group holds two images."""
    images, labelled, positives = {}, [], []
    for kind, n in (("p", n_pos), ("n", n_neg)):
        for g in range(n):
            for k in range(2):
                image_id = f"{kind}{g}-{k}"
                images[image_id] = {"group_id": f"g-{kind}{g}", "sha256": image_id, "dhash": None, "source": None}
                labelled.append(image_id)
                if kind == "p" and k == 0:
                    positives.append(image_id)
    for image_id, group in extra_images:
        images[image_id] = {"group_id": group, "sha256": image_id, "dhash": None, "source": None}
    groups = {"version": "groups-v1", "max_hamming": 6, "review_margin": 4, "images": images, "pairs": [],
              "unreadable": []}
    labels = {"vad_label_provenance": "synthetic", "categories": [dict(c) for c in CATS],
              "images": [{"id": i + 1, "vad_image_id": name, "width": 100, "height": 100, "vad_exhaustive": True}
                         for i, name in enumerate(labelled)],
              "annotations": []}
    ids = {img["vad_image_id"]: img["id"] for img in labels["images"]}
    for k, name in enumerate(positives):
        labels["annotations"].append({"id": k + 1, "image_id": ids[name], "category_id": 1, "bbox": [10, 10, 20, 20]})
    exclusions = {"vad_exclusions": 1, "recorded_by": "rev-1", "exclusions": [dict(e) for e in excluded]}
    return groups, exclusions, labels


def _groups_by_split(manifest):
    out: dict[str, list[str]] = {}
    for group_id, g in manifest["groups"].items():
        out.setdefault(g["split"], []).append(group_id)
    return out


def _digest(doc):
    return hashlib.sha256(canonical_json(doc).encode()).hexdigest()


def test_every_group_lands_in_exactly_one_split_with_all_its_images():
    manifest, _ = make_split(*_inputs(), task(), seed=11)
    check_isolation({k: v for k, v in _groups_by_split(manifest).items() if k != "excluded"})
    for img in manifest["images"].values():
        assert img["split"] == manifest["groups"][img["group_id"]]["split"]


def test_positive_and_negative_groups_are_stratified_across_all_three_splits():
    counts = make_split(*_inputs(n_pos=10, n_neg=6), task(), seed=11)[0]["counts"]
    # positive 10 -> frozen 3, tuning 7 -> calibration 4, development 3; negative 6 -> 2, 4 -> 2, 2
    assert (counts["frozen_eval"]["positive_groups"], counts["frozen_eval"]["negative_groups"]) == (3, 2)
    assert (counts["calibration"]["positive_groups"], counts["calibration"]["negative_groups"]) == (4, 2)
    assert (counts["development"]["positive_groups"], counts["development"]["negative_groups"]) == (3, 2)
    assert counts["frozen_eval"]["images"] == 10


def test_same_seed_and_inputs_give_a_byte_identical_manifest_regardless_of_input_order():
    groups, exclusions, labels = _inputs()
    first = json.dumps(make_split(groups, exclusions, labels, task(), seed=11)[0], indent=2, sort_keys=True)
    reordered = {**groups, "images": dict(reversed(list(groups["images"].items())))}
    second = json.dumps(make_split(reordered, exclusions, labels, task(), seed=11)[0], indent=2, sort_keys=True)
    other_seed = make_split(groups, exclusions, labels, task(), seed=12)[0]
    assert first == second and json.loads(first)["groups"] != other_seed["groups"]


def test_excluded_and_out_of_domain_images_are_carried_as_excluded():
    excluded = [{"image_id": "p0-1", "reason": "screenshot"},
                {"image_id": "x-0", "reason": "illustration", "out_of_domain_set": True}]
    manifest, _ = make_split(*_inputs(extra_images=[("x-0", "g-x")], excluded=excluded), task(), seed=11)
    assert manifest["images"]["p0-1"]["split"] == "excluded"
    assert manifest["images"]["p0-1"]["exclusion"] == {"reason": "screenshot", "out_of_domain_set": False}
    assert manifest["images"]["p0-0"]["split"] != "excluded"  # the rest of its group stays in its split
    assert manifest["images"]["x-0"]["exclusion"] == {"reason": "illustration", "out_of_domain_set": True}
    assert manifest["groups"]["g-x"]["split"] == "excluded"  # every image excluded -> the group is excluded
    assert manifest["counts"]["excluded"]["images"] == 2


def test_manifest_pins_the_input_digests_and_the_seed():
    groups, exclusions, labels = _inputs()
    manifest, _ = make_split(groups, exclusions, labels, task(), seed=11)
    assert manifest["inputs"] == {"groups_sha256": _digest(groups), "exclusions_sha256": exclusions_digest(exclusions),
                                  "labels_sha256": _digest(labels)}
    assert (manifest["seed"], manifest["version"]) == (11, "split-v1")


def test_halves_round_up_not_to_even():
    counts = make_split(*_inputs(n_pos=7, n_neg=3), task(), seed=11)[0]["counts"]
    # positive 7 -> frozen round(2.1) = 2, tuning 5 -> calibration round_half_up(2.5) = 3, development 2
    assert (counts["calibration"]["positive_groups"], counts["development"]["positive_groups"]) == (3, 2)


def test_a_group_whose_only_box_is_ignore_is_negative():
    groups, exclusions, labels = _inputs(n_pos=3, n_neg=3)
    first_negative = next(i["id"] for i in labels["images"] if i["vad_image_id"] == "n0-0")
    labels["annotations"].append({"id": 900, "image_id": first_negative, "category_id": 1, "bbox": [5, 5, 9, 9],
                                  "ignore": True})
    manifest, _ = make_split(groups, exclusions, labels, task(), seed=11)
    assert manifest["groups"]["g-n0"]["stratum"] == "negative"


def test_the_frozen_lock_still_verifies_after_the_label_file_is_reordered(tmp_path):
    groups, exclusions, labels = _inputs()
    manifest, frozen = make_split(groups, exclusions, labels, task(), seed=11)
    frozen_ids = sorted(i for i, img in manifest["images"].items() if img["split"] == "frozen_eval")
    lock_frozen_set(frozen_ids, _groups_by_split(manifest)["frozen_eval"], frozen, tmp_path / "lock.json")
    reordered = {**labels, "images": labels["images"][::-1], "annotations": labels["annotations"][::-1]}
    assert load_frozen(tmp_path / "lock.json", frozen_label_subset(reordered, frozen_ids))


def test_a_stratum_too_small_for_three_splits_is_refused_not_padded():
    with pytest.raises(SampleError, match="negative stratum has 2 groups.*at least 3"):
        make_split(*_inputs(n_pos=10, n_neg=2), task(), seed=11)


def test_an_empty_stratum_is_allowed_but_stated():
    manifest, _ = make_split(*_inputs(n_pos=6, n_neg=0), task(), seed=11)
    assert manifest["warnings"] == ["no negative groups: no split contains negative images"]


def test_unlabelled_included_images_are_refused():
    groups, exclusions, labels = _inputs()
    labels["images"] = [i for i in labels["images"] if i["vad_image_id"] != "n0-0"]
    with pytest.raises(SampleError, match="n0-0"):
        make_split(groups, exclusions, labels, task(), seed=11)


def test_unreadable_images_must_be_excluded():
    groups, exclusions, labels = _inputs()
    groups["unreadable"] = ["n1-1"]
    with pytest.raises(SampleError, match="unreadable.*n1-1"):
        make_split(groups, exclusions, labels, task(), seed=11)


def test_frozen_subset_verifies_against_the_lock_and_detects_edits(tmp_path):
    groups, exclusions, labels = _inputs()
    manifest, frozen = make_split(groups, exclusions, labels, task(), seed=11)
    frozen_ids = sorted(i for i, img in manifest["images"].items() if img["split"] == "frozen_eval")
    assert frozen == frozen_label_subset(labels, frozen_ids)
    lock_frozen_set(frozen_ids, _groups_by_split(manifest)["frozen_eval"], frozen, tmp_path / "lock.json")
    assert load_frozen(tmp_path / "lock.json", frozen_label_subset(labels, frozen_ids))["image_ids"] == frozen_ids
    edited = json.loads(json.dumps(labels))
    frozen_image = next(i["id"] for i in edited["images"] if i["vad_image_id"] in frozen_ids)
    edited["annotations"].append({"id": 999, "image_id": frozen_image, "category_id": 2, "bbox": [1, 1, 5, 5]})
    with pytest.raises(IsolationError):
        load_frozen(tmp_path / "lock.json", frozen_label_subset(edited, frozen_ids))


def _write_inputs(tmp_path):
    paths = {}
    for name, doc in zip(("groups", "exclusions", "labels"), _inputs()):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(doc), encoding="utf-8")
    paths["config"] = tmp_path / "config.yaml"
    paths["config"].write_text(yaml.safe_dump(config().model_dump(mode="json")), encoding="utf-8")
    return paths


def _args(paths, out_dir, without=()):
    args = ["sample", "split", "--config", str(paths["config"]), "--groups", str(paths["groups"]), "--seed", "11",
            "--out-dir", str(out_dir)]
    for name in ("exclusions", "labels"):
        if name not in without:
            args += [f"--{name}", str(paths[name])]
    return args


def test_cli_split_writes_manifest_and_lock_reruns_identically_and_never_overwrites(tmp_path):
    paths = _write_inputs(tmp_path)
    assert cli.main(_args(paths, tmp_path / "v1")) == 0
    assert cli.main(_args(paths, tmp_path / "v1b")) == 0
    first = (tmp_path / "v1" / "split_manifest.json").read_bytes()
    assert first == (tmp_path / "v1b" / "split_manifest.json").read_bytes()  # byte-identical rerun
    lock = json.loads((tmp_path / "v1" / "frozen_lock.json").read_text(encoding="utf-8"))
    assert json.loads(first)["frozen_lock_sha256"] == lock["lock_sha256"]
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(paths, tmp_path / "v1"))  # a split is versioned: never overwritten
    assert exc.value.code == 2


@pytest.mark.parametrize("missing", ["exclusions", "labels"])
def test_cli_split_refuses_to_run_without_exclusions_or_labels(tmp_path, capsys, missing):
    paths = _write_inputs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(paths, tmp_path / "v1", without=(missing,)))
    assert exc.value.code == 2 and f"--{missing}" in capsys.readouterr().err
    assert not (tmp_path / "v1").exists()
