"""A2 stage 4: second-review queue and label merge (synthetic labels only)."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from routing import cli
from routing.metrics import load_ground_truth
from routing.sample import SampleError, merge_reviews, review_queue

from .helpers import CLASS_A, CLASS_B, config, task

CATS = [{"id": 1, "name": CLASS_A}, {"id": 2, "name": CLASS_B}]


def _first(n_groups=10, ambiguous=("g0-0",), reviewer="ana", provenance="human"):
    """First-pass labels: group gK holds images gK-0 and gK-1; even groups have one class-A box per image."""
    images, anns, groups = [], [], {}
    for g in range(n_groups):
        for k in range(2):
            name = f"g{g}-{k}"
            images.append({"id": len(images) + 1, "vad_image_id": name, "width": 100, "height": 100,
                           "vad_exhaustive": True, "vad_ambiguous": name in ambiguous, "vad_reviewers": [reviewer]})
            groups[name] = {"group_id": f"grp-{g}", "sha256": name, "dhash": None, "source": None}
            if g % 2 == 0:
                anns.append({"id": len(anns) + 1, "image_id": images[-1]["id"], "category_id": 1,
                             "bbox": [10, 10, 30, 30]})
    labels = {"vad_label_provenance": provenance, "categories": [dict(c) for c in CATS], "images": images,
              "annotations": anns}
    groups_doc = {"version": "groups-v1", "max_hamming": 6, "review_margin": 4, "images": groups, "pairs": [],
                  "unreadable": []}
    return labels, groups_doc


def _second(first, image_ids, reviewer="ben", change=None):
    """Second-pass labels for the queued images: a copy of the first pass, optionally changed per image."""
    doc = copy.deepcopy(first)
    keep = {img["id"] for img in doc["images"] if img["vad_image_id"] in image_ids}
    doc["images"] = [dict(img, vad_reviewers=[reviewer]) for img in doc["images"] if img["id"] in keep]
    doc["annotations"] = [a for a in doc["annotations"] if a["image_id"] in keep]
    for image_id, boxes in (change or {}).items():
        key = next(img["id"] for img in doc["images"] if img["vad_image_id"] == image_id)
        doc["annotations"] = [a for a in doc["annotations"] if a["image_id"] != key]
        for cat, bbox in boxes:
            doc["annotations"].append({"id": 1000 + len(doc["annotations"]), "image_id": key, "category_id": cat,
                                       "bbox": bbox})
    return doc


# --- queue --------------------------------------------------------------------------------------


def test_queue_takes_every_ambiguous_image_and_a_seeded_fifth_of_the_other_groups():
    labels, groups = _first(n_groups=10, ambiguous=("g0-0", "g3-1"))
    queue = review_queue(labels, groups, task(), seed=5)
    ambiguous = sorted(i["image_id"] for i in queue["images"] if i["reason"] == "ambiguous")
    sampled_groups = {i["group_id"] for i in queue["images"] if i["reason"] == "sample"}
    assert ambiguous == ["g0-0", "g3-1"] and len(sampled_groups) == 2  # round(0.2 * 10 groups)
    assert all(i["image_id"] not in ambiguous for i in queue["images"] if i["reason"] == "sample")
    assert queue["first_reviewers"] == ["ana"] and queue["sample_fraction"] == 0.2


def test_queue_is_order_independent_and_seeded():
    labels, groups = _first()
    reordered = {**labels, "images": labels["images"][::-1]}
    same = json.dumps(review_queue(labels, groups, task(), seed=5), sort_keys=True)
    assert same == json.dumps(review_queue(reordered, groups, task(), seed=5), sort_keys=True)
    picks = lambda seed: {i["group_id"] for i in review_queue(labels, groups, task(), seed)["images"]}  # noqa: E731
    assert picks(5) != picks(6)


def test_queue_refuses_images_without_a_group_or_a_single_first_reviewer():
    labels, groups = _first()
    del groups["images"]["g1-0"]
    with pytest.raises(SampleError, match="g1-0"):
        review_queue(labels, groups, task(), seed=5)
    labels, groups = _first()
    labels["images"][2]["vad_reviewers"] = []
    with pytest.raises(SampleError, match="first reviewer"):
        review_queue(labels, groups, task(), seed=5)


# --- merge --------------------------------------------------------------------------------------


def _queue_ids(queue):
    return [i["image_id"] for i in queue["images"]]


def test_agreeing_reviews_merge_with_both_reviewers_recorded():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    merged, disagreements = merge_reviews(labels, _second(labels, _queue_ids(queue)), queue, {}, task())
    assert disagreements == []
    reviewed = next(img for img in merged["images"] if img["vad_image_id"] == _queue_ids(queue)[0])
    assert reviewed["vad_reviewers"] == ["ana", "ben"]
    assert reviewed["vad_review"] == {"first": "ana", "second": "ben", "agreement": True, "decision": None,
                                      "adjudicator": None, "note": None, "alternative": []}
    unreviewed = next(img for img in merged["images"] if img["vad_image_id"] not in _queue_ids(queue))
    assert unreviewed["vad_reviewers"] == ["ana"] and unreviewed["vad_review"]["second"] is None
    assert len(load_ground_truth(merged, task())) == 20  # the A3 loader accepts the merged file


def test_a_disagreement_without_adjudication_is_reported_not_overwritten():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    target = _queue_ids(queue)[0]
    second = _second(labels, _queue_ids(queue), change={target: [(2, [50, 50, 20, 20])]})
    with pytest.raises(SampleError, match=target):
        merge_reviews(labels, second, queue, {}, task())


def test_an_adjudicated_disagreement_keeps_both_opinions():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    target = _queue_ids(queue)[0]
    second = _second(labels, _queue_ids(queue), change={target: [(2, [50, 50, 20, 20])]})
    adjudications = {"vad_adjudications": 1, "decisions": {target: {"decision": "second", "adjudicator": "cleo",
                                                                     "note": "class B is right"}}}
    merged, disagreements = merge_reviews(labels, second, queue, adjudications, task())
    assert [d["image_id"] for d in disagreements] == [target]
    img = next(i for i in merged["images"] if i["vad_image_id"] == target)
    assert (img["vad_review"]["decision"], img["vad_review"]["adjudicator"]) == ("second", "cleo")
    assert img["vad_review"]["note"] == "class B is right" and img["vad_review"]["agreement"] is False
    chosen = [a for a in merged["annotations"] if a["image_id"] == img["id"]]
    assert [(a["category_id"], a["bbox"]) for a in chosen] == [(2, [50, 50, 20, 20])]
    first_opinion = [(a["category_id"], a["bbox"]) for a in labels["annotations"]
                     if a["image_id"] == next(i["id"] for i in labels["images"] if i["vad_image_id"] == target)]
    assert [(a["category_id"], a["bbox"]) for a in img["vad_review"]["alternative"]] == first_opinion


def _renumber(doc, ids):
    """The same labels under other numeric category ids ({class name: new id}), as another tool might export."""
    old = {c["id"]: c["name"] for c in doc["categories"]}
    return {**doc, "categories": [{"id": ids[c["name"]], "name": c["name"]} for c in doc["categories"]],
            "annotations": [{**a, "category_id": ids[old[a["category_id"]]]} for a in doc["annotations"]]}


def _classes_and_boxes(doc, image_id):
    gt = load_ground_truth(doc, task())[image_id]
    return [(b.class_id, b.box_xyxy) for b in gt.boxes]


@pytest.mark.parametrize("ids", [{CLASS_A: 2, CLASS_B: 1}, {CLASS_A: 7, CLASS_B: 8}], ids=["swapped", "disjoint"])
def test_the_chosen_second_opinion_keeps_its_class_under_other_category_ids(ids):
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    target = _queue_ids(queue)[0]
    second = _second(labels, _queue_ids(queue), change={target: [(2, [50, 50, 20, 20])]})  # class B in pass 2
    adjudications = {"vad_adjudications": 1, "decisions": {target: {"decision": "second", "adjudicator": "cleo"}}}
    merged, _ = merge_reviews(labels, _renumber(second, ids), queue, adjudications, task())
    assert _classes_and_boxes(merged, target) == [(CLASS_B, (50, 50, 70, 70))]


@pytest.mark.parametrize("ids", [{CLASS_A: 2, CLASS_B: 1}, {CLASS_A: 7, CLASS_B: 8}], ids=["swapped", "disjoint"])
def test_the_kept_alternative_opinion_uses_the_merged_category_ids(ids):
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    target = _queue_ids(queue)[0]
    second = _second(labels, _queue_ids(queue), change={target: [(2, [50, 50, 20, 20])]})
    adjudications = {"vad_adjudications": 1, "decisions": {target: {"decision": "first", "adjudicator": "cleo"}}}
    merged, disagreements = merge_reviews(labels, _renumber(second, ids), queue, adjudications, task())
    names = {c["id"]: c["name"] for c in merged["categories"]}
    img = next(i for i in merged["images"] if i["vad_image_id"] == target)
    assert [(names[a["category_id"]], a["bbox"]) for a in img["vad_review"]["alternative"]] == \
        [(CLASS_B, [50, 50, 20, 20])]
    # the disagreement report uses the same (merged) category ids for both passes
    assert [names[a["category_id"]] for a in disagreements[0]["second_annotations"]] == [CLASS_B]


def test_the_first_pass_may_use_other_ids_than_the_second_and_agreement_still_counts():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    second = _renumber(_second(labels, _queue_ids(queue)), {CLASS_A: 2, CLASS_B: 1})
    merged, disagreements = merge_reviews(labels, second, queue, {}, task())
    assert disagreements == [] and len(load_ground_truth(merged, task())) == 20


@pytest.mark.parametrize("side, size", [("width", "120x100"), ("height", "100x120")])
def test_merge_refuses_passes_that_disagree_on_image_size(side, size):
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    target = _queue_ids(queue)[0]
    second = _second(labels, _queue_ids(queue))
    next(i for i in second["images"] if i["vad_image_id"] == target)[side] = 120
    with pytest.raises(SampleError, match=f"{target}.*100x100.*{size}"):
        merge_reviews(labels, second, queue, {}, task())


def test_boxes_that_overlap_enough_count_as_agreement():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    target = next(i for i in _queue_ids(queue) if int(i[1:].split("-")[0]) % 2 == 0)  # an image with a box
    second = _second(labels, _queue_ids(queue), change={target: [(1, [11, 11, 30, 29])]})  # IoU ~0.9
    assert merge_reviews(labels, second, queue, {}, task())[1] == []


def test_reviewers_who_differ_only_on_an_ignore_box_disagree():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    target = _queue_ids(queue)[0]
    second = _second(labels, _queue_ids(queue))
    key = next(img["id"] for img in second["images"] if img["vad_image_id"] == target)
    second["annotations"].append({"id": 777, "image_id": key, "category_id": 1, "bbox": [60, 60, 20, 20],
                                  "ignore": True})
    with pytest.raises(SampleError, match=target):
        merge_reviews(labels, second, queue, {}, task())


@pytest.mark.parametrize("handle", ["ana", "ANA ", "", "qwen3-vl-4b", "openai-compatible:m@host/v1", "gpt-4o"])
def test_second_reviewer_must_be_a_different_human_handle(handle):
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    with pytest.raises(SampleError, match="reviewer"):
        merge_reviews(labels, _second(labels, _queue_ids(queue), reviewer=handle), queue, {}, task())


def test_a_refused_handle_says_which_token_matched():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    with pytest.raises(SampleError, match="'dino'.*guard against pasted model ids"):
        merge_reviews(labels, _second(labels, _queue_ids(queue), reviewer="Dinoyo"), queue, {}, task())


def test_the_second_pass_must_cover_exactly_the_queue():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    with pytest.raises(SampleError, match="missing"):
        merge_reviews(labels, _second(labels, _queue_ids(queue)[1:]), queue, {}, task())
    with pytest.raises(SampleError, match="not in the queue"):
        merge_reviews(labels, _second(labels, _queue_ids(queue) + ["g9-1"]), queue, {}, task())


def test_merge_refuses_a_queue_made_from_other_first_labels():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    edited = copy.deepcopy(labels)
    edited["annotations"].append({"id": 99, "image_id": 2, "category_id": 2, "bbox": [1, 1, 5, 5]})
    with pytest.raises(SampleError, match="queue"):
        merge_reviews(edited, _second(edited, _queue_ids(queue)), queue, {}, task())


def test_provenance_stays_human_only_when_both_passes_declare_it():
    labels, groups = _first()
    queue = review_queue(labels, groups, task(), seed=5)
    second = _second(labels, _queue_ids(queue))
    second["vad_label_provenance"] = "synthetic"
    assert merge_reviews(labels, second, queue, {}, task())[0]["vad_label_provenance"] == "mixed"


# --- CLI ----------------------------------------------------------------------------------------


def test_cli_queue_then_merge_writes_a_loadable_label_file_or_a_disagreement_report(tmp_path):
    labels, groups = _first()
    paths = {name: tmp_path / f"{name}.json" for name in ("first", "groups", "queue", "second", "merged", "adj")}
    paths["first"].write_text(json.dumps(labels), encoding="utf-8")
    paths["groups"].write_text(json.dumps(groups), encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config().model_dump(mode="json")), encoding="utf-8")
    assert cli.main(["sample", "review-queue", "--config", str(cfg), "--labels", str(paths["first"]),
                     "--groups", str(paths["groups"]), "--seed", "5", "--out", str(paths["queue"])]) == 0
    queue = json.loads(paths["queue"].read_text(encoding="utf-8"))
    target = _queue_ids(queue)[0]
    paths["second"].write_text(json.dumps(_second(labels, _queue_ids(queue), change={target: [(2, [5, 5, 9, 9])]})),
                               encoding="utf-8")
    merge = ["sample", "merge", "--config", str(cfg), "--first", str(paths["first"]), "--second", str(paths["second"]),
             "--queue", str(paths["queue"]), "--out", str(paths["merged"])]
    with pytest.raises(SystemExit) as exc:
        cli.main(merge)
    report = json.loads(paths["merged"].with_suffix(".disagreements.json").read_text(encoding="utf-8"))
    assert exc.value.code == 2 and not paths["merged"].exists() and [d["image_id"] for d in report] == [target]
    paths["adj"].write_text(json.dumps({"vad_adjudications": 1,
                                        "decisions": {target: {"decision": "first", "adjudicator": "cleo"}}}),
                            encoding="utf-8")
    assert cli.main(merge + ["--adjudications", str(paths["adj"])]) == 0
    assert len(load_ground_truth(json.loads(paths["merged"].read_text(encoding="utf-8")), task())) == 20
