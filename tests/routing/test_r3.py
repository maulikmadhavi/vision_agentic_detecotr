"""Regression tests for the independent review findings R3-1..R3-5 (docs/reviewer_suggestion.md §6)."""

from __future__ import annotations

import json

import yaml

from vad.routing import cli

from .helpers import CLASS_A, config, draw, make_fixture


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
