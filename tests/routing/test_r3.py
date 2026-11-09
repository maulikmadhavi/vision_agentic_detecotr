"""Regression tests for the independent review findings R3-1..R3-5 (docs/reviewer_suggestion.md §6)."""

from __future__ import annotations

import json

import pytest
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


# R3-4: incomplete calibrated relevance must not yield a low band --------------------------------


class _LowCalibrator:
    version = "cal@test"

    def calibrate(self, raw):
        return {k: 0.01 for k in raw}, 0.02  # confidently "nothing here" for the classes it saw


def test_r3_4_missing_class_with_calibrator_is_not_low_relevance(tmp_path):
    rel = ScriptedRelevance({"a": {CLASS_A: 0.05}})  # CLASS_B not scored at all
    rec = RoutingPipeline(config(), ScriptedDetector({"a": []}), RouteStore(tmp_path / "r.sqlite"),
                          relevance=rel, calibrator=_LowCalibrator()).process(_entry(tmp_path))
    sa = rec.stage_a
    assert sa.relevance_status == "calibrated" and sa.image_relevance_probability == 0.02
    assert sa.relevance_band == "medium"
    assert "incomplete_relevance_low_band_raised_to_medium" in sa.warnings
    assert "low_relevance_zero_boxes" not in rec.route.reason_codes  # not treated as a likely negative

    complete = ScriptedRelevance({"a": {CLASS_A: 0.05, CLASS_B: 0.05}})
    rec2 = RoutingPipeline(config(), ScriptedDetector({"a": []}), RouteStore(tmp_path / "r2.sqlite"),
                           relevance=complete, calibrator=_LowCalibrator()).process(_entry(tmp_path))
    assert rec2.stage_a.relevance_band == "low"  # complete evidence may still gate low


# R3-5: a bare human "complete" cannot clear open findings ---------------------------------------


def test_r3_5_human_complete_does_not_clear_open_findings(tmp_path):
    from vad.routing.completeness import audit
    from vad.routing.schemas import Budget, FinalCandidate, Quality, RouteDecision, StageA

    cfg = config()
    sa = StageA(per_class={}, relevance_status="unavailable", relevance_band="unknown",
                quality=Quality(blur=0.1, exposure=0, min_side_px=500, low_resolution=False))
    dec = RouteDecision(actions=("standard",), priority=0, reason_codes=(), policy_version="v", expected_budget=Budget())
    open_cand = FinalCandidate(id="cand-1", class_id=CLASS_A, box_xyxy=(1, 1, 9, 9), status="needs_review",
                               source_ids=("det-1",), canonical_source_id="det-1")

    unmatched = audit([open_cand], sa, dec, [], cfg, human_outcome="complete")
    assert unmatched.status == "needs_review" and unmatched.established_by == "human"
    assert "human_complete_overridden_by_open_findings" in unmatched.reason_codes
    failed = audit([], sa, dec, [], cfg, failure_reasons=("secondary_failed:RuntimeError",), human_outcome="complete")
    assert failed.status != "complete"
    clean = audit([], sa, dec, [], cfg, human_outcome="complete")
    assert clean.status == "complete" and clean.reviewer_outcome == "complete"


# R4-1: incomplete or inconsistent *calibrator* output must not yield a low band ------------------


class _ScriptedCalibrator:
    version = "cal@scripted"

    def __init__(self, probs, any_p):
        self.probs, self.any_p = probs, any_p

    def calibrate(self, raw):
        return dict(self.probs), self.any_p


def _calibrated(tmp_path, name, calibrator):
    rel = ScriptedRelevance({"a": {CLASS_A: 0.05, CLASS_B: 0.05}})  # scorer covers every class
    return RoutingPipeline(config(), ScriptedDetector({"a": []}), RouteStore(tmp_path / f"{name}.sqlite"),
                           relevance=rel, calibrator=calibrator).process(_entry(tmp_path))


def test_r4_1_calibrator_omitting_a_class_is_not_low_relevance(tmp_path):
    rec = _calibrated(tmp_path, "omit", _ScriptedCalibrator({CLASS_A: 0.01}, 0.02))
    assert rec.stage_a.relevance_band == "medium"
    assert "calibrator_output_incomplete_or_inconsistent" in rec.stage_a.warnings
    assert "low_relevance_zero_boxes" not in rec.route.reason_codes


def test_r4_1_inconsistent_aggregate_is_not_low_relevance(tmp_path):
    # P(any target) = 0.02 < P(CLASS_B) = 0.9 is impossible for a coherent calibrator
    rec = _calibrated(tmp_path, "agg", _ScriptedCalibrator({CLASS_A: 0.01, CLASS_B: 0.9}, 0.02))
    assert rec.stage_a.relevance_band == "medium"
    assert "calibrator_output_incomplete_or_inconsistent" in rec.stage_a.warnings
    assert "low_relevance_zero_boxes" not in rec.route.reason_codes


# T-6: a class the calibrator omitted is unknown, never "not relevant", and never uses raw bands ---



@pytest.mark.parametrize("raw_b", [0.9, 0.05])  # high and low raw score for the omitted class
def test_t6_calibrator_missing_class_is_unknown_not_negative(tmp_path, raw_b):
    rel = ScriptedRelevance({"a": {CLASS_A: 0.05, CLASS_B: raw_b}})
    rec = RoutingPipeline(config(), ScriptedDetector({"a": []}), RouteStore(tmp_path / "r.sqlite"), relevance=rel,
                          calibrator=_ScriptedCalibrator({CLASS_A: 0.01}, 0.02)).process(_entry(tmp_path))
    assert rec.stage_a.relevance_band != "low" and "low_relevance_zero_boxes" not in rec.route.reason_codes
    assert f"calibrator_missing_class:{CLASS_B}" in rec.completeness.reason_codes
    # listed for review as unknown, never judged by raw score against raw bands
    assert f"high_relevance_zero_boxes_for_class:{CLASS_B}" not in rec.completeness.reason_codes
    assert rec.completeness.status == "unresolved"


# P2: the unknown (calibrator-missing) class gets an explicit human handoff, also with a VLM -------


def test_p2_calibrator_missing_class_routes_to_human_review_with_vlm(tmp_path):
    rel = ScriptedRelevance({"a": {CLASS_A: 0.05, CLASS_B: 0.9}})
    vlm = FakeVLM(default="absent")
    rec = RoutingPipeline(config(vlm={"enabled": True}), ScriptedDetector({"a": []}), RouteStore(tmp_path / "r.sqlite"),
                          relevance=rel, calibrator=_ScriptedCalibrator({CLASS_A: 0.01}, 0.02),
                          vlm=vlm).process(_entry(tmp_path))
    assert "human_review" in rec.route.actions
    assert f"calibrator_missing_class:{CLASS_B}" in rec.route.reason_codes
    assert CLASS_B in rec.completeness.suspected_missing_classes
    assert not any(r.class_id == CLASS_B for r, _ in vlm.requests)  # a VLM verdict cannot clear "unknown"
    assert rec.completeness.status == "unresolved"
