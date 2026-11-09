"""Regression tests for the independent review findings R3-1..R3-5 (docs/reviewer_suggestion.md §6)."""

from __future__ import annotations

import json

import yaml

from vad.routing import cli
from vad.routing.adapters import RawDetection
from vad.routing.pipeline import ManifestEntry, RouteStore, RoutingPipeline

from .helpers import CLASS_A, CLASS_B, FakeVLM, ScriptedDetector, ScriptedRelevance, config, draw, make_fixture


def _entry(tmp_path, name="a"):
    return ManifestEntry(str(draw(tmp_path / f"{name}.png", (320, 240), [])), image_id=name)


# R3-1: a reused output directory must not mix earlier runs into the export ---------------------


def test_r3_1_reused_out_dir_exports_only_the_current_run(tmp_path):
    fx = make_fixture(tmp_path / "fx")
    out = tmp_path / "shadow"
    base = ["shadow", "--config", str(fx["config"]), "--primary-version", "prim@1", "--out", str(out)]
    assert cli.main(base + ["--images", str(fx["manifest"]), "--primary-predictions", str(fx["preds"])]) == 0

    # Second run: different manifest, different task (one new class), same --out.
    other = tmp_path / "other"
    draw(other / "x.png", (320, 240), [(CLASS_A, (40, 40, 120, 120))])
    (other / "manifest.jsonl").write_text(json.dumps({"uri": "x.png", "image_id": "x"}) + "\n", encoding="utf-8")
    (other / "preds.jsonl").write_text(json.dumps({"image_id": "x", "detections": [
        {"class_id": "new_part_q1", "box_xyxy": [40, 40, 120, 120], "confidence": 0.9}]}) + "\n", encoding="utf-8")
    cfg = config(routing_enabled=False, task={"task_id": "t2", "prompt_set_version": "p1", "classes": [
        {"class_id": "new_part_q1", "display_name": "New part", "positive_prompts": ["a new part"]}]})
    (other / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    second = ["shadow", "--config", str(other / "config.yaml"), "--primary-version", "prim@2", "--out", str(out),
              "--images", str(other / "manifest.jsonl"), "--primary-predictions", str(other / "preds.jsonl")]
    assert cli.main(second) == 0

    ann = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
    assert [i["vad_image_id"] for i in ann["images"]] == ["x"]
    assert [c["name"] for c in ann["categories"]] == ["new_part_q1"]
    trace = (out / "route_trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(t)["image"]["id"] for t in trace] == ["x"]
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["images"] == 1 and report["errors"] == []  # the first run's corrupt image is not re-reported


# R3-2: cross-class conflicts are never silently rejected ---------------------------------------


def test_r3_2_cross_class_conflict_survives_vlm_absent(tmp_path):
    box = (40, 40, 200, 200)
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, box, 0.95)]})
    secondary = ScriptedDetector({"a": [RawDetection(CLASS_B, box, 0.9)]}, "sec@1")
    rel = ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.4}})  # moderate route -> secondary + VLM
    cfg = config(vlm={"enabled": True, "absent_rejects": True})
    vlm = FakeVLM(default="absent")
    rec = RoutingPipeline(cfg, primary, RouteStore(tmp_path / "r.sqlite"), secondary=secondary,
                          relevance=rel, vlm=vlm).process(_entry(tmp_path))
    assert len(vlm.requests) == 2  # both conflicting candidates were asked about
    assert sorted((c.class_id, c.status) for c in rec.final_candidates) == [
        (CLASS_A, "needs_review"), (CLASS_B, "needs_review")]
    assert all("cross_class_conflict" in c.flags for c in rec.final_candidates)
    assert rec.completeness.status != "complete"


# R3-3: interruption between the VLM call and persisting its answer ------------------------------


class Kill(BaseException):
    """Simulates the process dying (not an ordinary exception)."""


def test_r3_3_crash_after_vlm_answer_before_persist_never_resends(tmp_path):
    entry = _entry(tmp_path)
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, (40, 40, 200, 200), 0.97)]})
    rel = ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.1}})  # moderate, no secondary -> VLM confirms box
    cfg = config(vlm={"enabled": True})
    path = tmp_path / "r.sqlite"

    store = RouteStore(path)
    real_put = store.put_step

    def put_step(key, step, payload):  # the VLM has answered; die before its answer is saved
        if step.startswith("vlm:"):
            raise Kill()
        real_put(key, step, payload)

    store.put_step = put_step
    first = FakeVLM(default="present")
    try:
        RoutingPipeline(cfg, primary, store, relevance=rel, vlm=first).process(entry)
    except Kill:
        pass
    assert len(first.requests) == 1

    again = FakeVLM(default="present")
    rec = RoutingPipeline(cfg, primary, RouteStore(path), relevance=rel, vlm=again).process(entry)
    assert again.requests == []  # at-most-once: the interrupted request is not re-sent
    (c,) = rec.final_candidates
    assert c.status == "needs_review" and c.verification.reason_code == "vlm_interrupted"
    assert rec.completeness.status == "unresolved"
