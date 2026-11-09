"""One test group per row of the plan's acceptance table."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from routing import cli
from routing.interfaces import RawDetection
from service_detector_vlm.openai_service_call import OpenAICompatibleVerifier
from routing.evaluation import IsolationError, assert_not_frozen, check_isolation, load_frozen, lock_frozen_set
from routing.export import load_annotations
from routing.matching import iou
from routing.image_inputs import ManifestEntry
from routing.pipeline import RoutingPipeline
from routing.store import RouteStore
from routing.schemas import VerificationRequest
from routing.slicing import run_sliced, tile_grid
from routing.task_config import SlicingConfig

from .helpers import CLASS_A, CLASS_B, BlobDetector, FakeVLM, ScriptedDetector, ScriptedRelevance, config, draw, make_fixture, task


def _pipe(cfg, primary, tmp_path, **kw):
    return RoutingPipeline(cfg, primary, RouteStore(tmp_path / "routes.sqlite"), **kw)


def _img(tmp_path, name="a.png", size=(320, 240), rects=()):
    return ManifestEntry(str(draw(tmp_path / name, size, rects)), image_id=name.split(".")[0])


# 1. Arbitrary class registry ----------------------------------------------------------------


def test_arbitrary_class_flows_end_to_end(tmp_path):
    cfg = config(vlm={"enabled": True, "accept_confirmed_detector_proposals": True})
    entry = _img(tmp_path, rects=[(CLASS_B, (100, 80, 140, 120))])
    primary = ScriptedDetector({"a": []})
    secondary = BlobDetector("sec@1")
    vlm = FakeVLM(default="present")
    rel = ScriptedRelevance({"a": {CLASS_A: 0.1, CLASS_B: 0.9}})
    rec = _pipe(cfg, primary, tmp_path, secondary=secondary, relevance=rel, vlm=vlm).process(entry)

    assert rec.task.class_ids == (CLASS_A, CLASS_B)
    assert {c.class_id for c in rec.secondary_candidates} == {CLASS_B}
    # VLM received the project prompt, not a public-taxonomy label
    assert any(req.class_id == CLASS_B and desc == "a red custom thing" for req, desc in vlm.requests)
    accepted = [c for c in rec.final_candidates if c.status == "accepted"]
    assert [c.class_id for c in accepted] == [CLASS_B]

    from routing.export import export_routed
    doc = export_routed([rec], cfg.task, tmp_path / "ann.json")
    assert [c["name"] for c in doc["categories"]] == [CLASS_A, CLASS_B]
    assert doc["annotations"][0]["category_id"] == 2


def test_unknown_class_from_detector_is_dropped_and_reported(tmp_path):
    entry = _img(tmp_path)
    primary = ScriptedDetector({"a": [RawDetection("person", (10, 10, 50, 50), 0.9)]})
    rec = _pipe(config(), primary, tmp_path).process(entry)
    assert rec.primary_candidates == ()
    assert "primary:unknown_class_dropped:person" in rec.trace.warnings


def test_class_mapping_for_secondary(tmp_path):
    cfg = config(matching={"class_mapping": {"widget": CLASS_A}})
    entry = _img(tmp_path, rects=[(CLASS_A, (100, 80, 140, 120))])
    primary = ScriptedDetector({"a": []})
    secondary = BlobDetector("sec@1", label_map={CLASS_A: "widget"})
    rel = ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.1}})
    rec = _pipe(cfg, primary, tmp_path, secondary=secondary, relevance=rel).process(entry)
    assert [d.class_id for d in rec.secondary_candidates] == [CLASS_A]


# 2. Independence -----------------------------------------------------------------------------


def test_relevance_and_difficulty_are_independent(tmp_path):
    cfg = config()
    # high relevance, easy: one large, confident, sharp box
    e1 = _img(tmp_path, "hi.png", rects=[(CLASS_A, (40, 40, 200, 200))])
    # low relevance, hard: tiny low-confidence crowded boxes
    e2 = _img(tmp_path, "lo.png", rects=[])
    primary = ScriptedDetector({
        "hi": [RawDetection(CLASS_A, (40, 40, 200, 200), 0.97)],
        "lo": [RawDetection(CLASS_B, (10, 10, 14, 14), 0.2), RawDetection(CLASS_B, (11, 11, 15, 15), 0.25)],
    })
    rel = ScriptedRelevance({"hi": {CLASS_A: 0.9, CLASS_B: 0.1}, "lo": {CLASS_A: 0.1, CLASS_B: 0.1}})
    pipe = _pipe(cfg, primary, tmp_path, relevance=rel)
    hi, lo = pipe.process(e1), pipe.process(e2)

    assert hi.stage_a.relevance_band == "high" and hi.stage_b.difficulty_pre_secondary.bucket == "easy"
    assert hi.route.actions == ("standard",)
    assert lo.stage_a.relevance_band == "low" and lo.stage_b.difficulty_pre_secondary.bucket == "hard"
    assert "low_relevance_with_detections" in lo.route.reason_codes
    assert "human_review" in lo.route.actions
    # scores stored in separate fields
    assert hi.stage_a.per_class[CLASS_A].raw_similarity == 0.9
    assert hi.stage_a.per_class[CLASS_A].relevance_probability is None
    assert hi.stage_a.relevance_status == "uncalibrated"


# 3. Zero detections --------------------------------------------------------------------------


def test_high_relevance_zero_boxes_is_never_easy_or_negative(tmp_path):
    entry = _img(tmp_path)
    rel = ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.1}})
    rec = _pipe(config(), ScriptedDetector({"a": []}), tmp_path, relevance=rel).process(entry)
    d = rec.stage_b.difficulty_pre_secondary
    assert d.status == "unknown" and d.bucket == "unknown" and d.value is None
    assert "high_relevance_zero_boxes" in rec.route.reason_codes
    # no secondary, SAHI or VLM capacity -> human review
    assert "human_review" in rec.route.actions
    assert rec.completeness.status != "complete"
    assert CLASS_A in rec.completeness.suspected_missing_classes


def test_zero_boxes_with_secondary_escalates(tmp_path):
    entry = _img(tmp_path, rects=[(CLASS_A, (100, 80, 140, 120))])
    rel = ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.1}})
    sec = BlobDetector("sec@1")
    rec = _pipe(config(), ScriptedDetector({"a": []}), tmp_path, relevance=rel, secondary=sec).process(entry)
    assert "secondary" in rec.route.actions and sec.calls == 1
    assert rec.stage_b.difficulty_pre_secondary.status == "unknown"
    # secondary-only box is a proposal, not an accepted annotation
    assert [c.status for c in rec.final_candidates] == ["candidate"]
    assert rec.completeness.status != "complete"


def test_zero_boxes_unknown_relevance_goes_to_review(tmp_path):
    rec = _pipe(config(), ScriptedDetector({"a": []}), tmp_path).process(_img(tmp_path))
    assert rec.stage_a.relevance_status == "unavailable"
    assert "human_review" in rec.route.actions
    assert rec.completeness.status != "complete"


# 4. Stable routing ---------------------------------------------------------------------------

_REPLAY = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
from pathlib import Path
from tests.routing.helpers import make_fixture, config, BlobDetector, ScriptedRelevance
from routing.replay import PrecomputedDetector, PrecomputedRelevance
from routing.image_inputs import load_manifest
from routing.pipeline import RoutingPipeline
from routing.store import RouteStore
from routing.policy import stable_sample
fx = make_fixture(Path(sys.argv[1]) / "fx")
cfg = config(audit={"low_relevance_sample_rate": 0.5, "zero_box_sample_rate": 0.5})
pipe = RoutingPipeline(cfg, PrecomputedDetector(fx["preds"], "prim@1"), RouteStore(Path(sys.argv[1]) / "r.sqlite"),
                       secondary=BlobDetector("sec@1"), relevance=PrecomputedRelevance(fx["relevance"], "rel@1"))
out = []
for r in pipe.run(load_manifest(fx["manifest"])):
    if hasattr(r, "route"):
        out.append([r.record_id, list(r.route.actions), list(r.route.reason_codes),
                    [c.id for c in r.final_candidates], r.completeness.status])
    else:
        out.append(["error", r.record_id])
out.append([stable_sample(f"img{i}", "seed", 0.3) for i in range(50)])
print(json.dumps(out))
"""


def test_routing_replays_identically_across_processes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    results = []
    for seed in ("1", "12345"):
        run_dir = tmp_path / f"run{seed}"
        run_dir.mkdir()
        env = {**os.environ, "PYTHONHASHSEED": seed}
        p = subprocess.run([sys.executable, "-c", _REPLAY, str(run_dir), str(root)], capture_output=True,
                           text=True, env=env, cwd=root)
        assert p.returncode == 0, p.stderr
        results.append(json.loads(p.stdout))
    # record ids embed image hashes; the fixture is regenerated identically in each run
    assert results[0] == results[1]
    assert any(results[0][-1]) and not all(results[0][-1])


# 5. Disagreement -----------------------------------------------------------------------------


def test_directional_disagreement_and_provenance(tmp_path):
    entry = _img(tmp_path, size=(400, 300))
    primary = ScriptedDetector({"a": [
        RawDetection(CLASS_A, (10, 10, 60, 60), 0.9),      # matched tightly
        RawDetection(CLASS_A, (100, 10, 160, 70), 0.8),    # matched but mislocalized
        RawDetection(CLASS_B, (200, 100, 260, 160), 0.7),  # cross-class with secondary
        RawDetection(CLASS_A, (300, 200, 340, 240), 0.6),  # primary-only
    ]}, "prim@1")
    secondary = ScriptedDetector({"a": [
        RawDetection(CLASS_A, (11, 11, 61, 61), 0.85),
        RawDetection(CLASS_A, (12, 10, 62, 60), 0.5),      # duplicate on the first object
        RawDetection(CLASS_A, (110, 18, 170, 78), 0.8),    # IoU ~0.6 with primary #2
        RawDetection(CLASS_A, (200, 100, 260, 160), 0.7),  # same place, other class
        RawDetection(CLASS_B, (20, 200, 60, 240), 0.6),    # secondary-only
    ]}, "sec@1")
    rel = ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.4}})  # medium band -> secondary
    rec = _pipe(config(), primary, tmp_path, secondary=secondary, relevance=rel).process(entry)
    dis = rec.stage_b.disagreement
    assert dis.status == "computed"
    assert dis.matched == 2 and dis.mislocalized == 1
    assert dis.primary_only == 2  # cross-class primary + primary-only
    assert dis.secondary_only == 3  # duplicate + cross-class secondary + secondary-only
    assert dis.cross_class == 1
    # pre-secondary difficulty preserved, final recomputed with disagreement
    assert rec.stage_b.difficulty_pre_secondary != rec.stage_b.difficulty_final

    by_flag = {f: c for c in rec.final_candidates for f in c.flags}
    first = next(c for c in rec.final_candidates if c.box_xyxy == (10, 10, 60, 60))
    assert len(first.source_ids) == 3 and first.status == "accepted"  # primary + secondary + duplicate
    assert by_flag["mislocalized"].status == "needs_review"
    conflicts = [c for c in rec.final_candidates if "cross_class_conflict" in c.flags]
    assert sorted(c.status for c in conflicts) == ["candidate", "needs_review"]
    assert {c.class_id for c in conflicts} == {CLASS_A, CLASS_B}
    assert by_flag["primary_only_disagreement"].status == "needs_review"
    assert by_flag["detector_proposal:secondary"].status == "candidate"
    every_source = {d.id for d in (*rec.primary_candidates, *rec.secondary_candidates)}
    assert {s for c in rec.final_candidates for s in c.source_ids} == every_source


def test_no_secondary_marks_disagreement_unavailable(tmp_path):
    entry = _img(tmp_path)
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, (10, 10, 60, 60), 0.9)]})
    rec = _pipe(config(), primary, tmp_path).process(entry)
    assert rec.stage_b.disagreement.status == "unavailable"


# 6. SAHI coordinates -------------------------------------------------------------------------


def test_sahi_tile_edge_and_overlap_map_once_and_dedupe(tmp_path):
    from PIL import Image
    from routing.image_inputs import load_image
    cfg = SlicingConfig(tile=400, overlap=0.25, include_full_image=False, merge_ios=0.5)
    truth = [
        (CLASS_A, (310, 120, 380, 180)),   # wholly inside the 300..400 x-overlap: seen by two tiles
        (CLASS_B, (940, 640, 1000, 700)),  # flush with bottom-right image corner (edge-aligned last tile)
        (CLASS_A, (500, 330, 506, 336)),   # tiny object
    ]
    entry = ManifestEntry(str(draw(tmp_path / "big.png", (1000, 700), truth)), image_id="big")
    image, ref = load_image(entry)
    tiles = tile_grid(1000, 700, cfg)
    assert all(x2 <= 1000 and y2 <= 700 for _, _, x2, y2 in tiles)
    assert max(x2 for *_, x2, _ in tiles) == 1000 and max(y2 for *_, y2 in tiles) == 700

    res = run_sliced(BlobDetector(), image, ref, task().class_ids, cfg)
    assert res.raw_count > len(truth)  # overlap produced duplicates before merging
    got = sorted((d.class_id, d.box_xyxy) for d in res.detections)
    assert got == sorted((c, tuple(float(v) for v in b)) for c, b in truth)
    for d in res.detections:
        assert d.source == "sahi" and d.slice_origin is not None
        assert 0 <= d.box_xyxy[0] < d.box_xyxy[2] <= 1000 and 0 <= d.box_xyxy[1] < d.box_xyxy[3] <= 700
    merged = next(d for d in res.detections if d.box_xyxy == (310, 120, 380, 180))
    assert len(merged.parent_ids) == 2


def test_sahi_cut_object_is_reassembled_by_full_image_pass(tmp_path):
    from routing.image_inputs import load_image
    cfg = SlicingConfig(tile=400, overlap=0.1, include_full_image=True)
    truth = [(CLASS_A, (250, 100, 550, 200))]  # wider than the overlap: cut by every tile
    entry = ManifestEntry(str(draw(tmp_path / "wide.png", (1000, 400), truth)), image_id="wide")
    image, ref = load_image(entry)
    res = run_sliced(BlobDetector(), image, ref, task().class_ids, cfg)
    assert [d.box_xyxy for d in res.detections] == [(250, 100, 550, 200)]


def test_sahi_recovers_small_object_in_pipeline(tmp_path):
    cfg = config(slicing={"tile": 320, "overlap": 0.2, "include_full_image": False}, budgets={"max_slices": 16})
    entry = ManifestEntry(str(draw(tmp_path / "s.png", (640, 480), [(CLASS_B, (300, 200, 306, 206))])), image_id="s")
    primary = BlobDetector("prim@1", min_frac=0.001)  # misses tiny objects at full resolution...
    crop_det = BlobDetector("prim@1", min_frac=0.0)   # ...the sliced adapter sees them
    rel = ScriptedRelevance({"s": {CLASS_A: 0.1, CLASS_B: 0.9}})
    rec = _pipe(cfg, primary, tmp_path, relevance=rel, sahi_detector=crop_det).process(entry)
    assert "sahi" in rec.route.actions
    assert [d.box_xyxy for d in rec.sahi_candidates] == [(300, 200, 306, 206)]
    assert rec.trace.budget_used.slices == len(tile_grid(640, 480, cfg.slicing))


# 7. VLM failures -----------------------------------------------------------------------------


def _req():
    return VerificationRequest(id="r1", image_id="a", class_id=CLASS_A, candidate_ids=("c1",),
                               region_xyxy=(10, 10, 50, 50), purpose="confirm_candidate")


def _vlm_with(handler):
    calls = []

    def h(request):
        calls.append(request)
        return handler(request)

    return OpenAICompatibleVerifier("http://vlm.test/v1", "qwen3-vl-4b-instruct", transport=httpx.MockTransport(h),
                                    timeout_s=1), calls


def _chat(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.mark.parametrize("handler,reason", [
    (lambda r: _chat("this is not json"), "vlm_malformed"),
    (lambda r: _chat(json.dumps({"class_id": CLASS_A, "verdict": "maybe", "reason": "x"})), "vlm_malformed"),
    (lambda r: _chat(json.dumps({"class_id": CLASS_A, "verdict": "present", "reason": "x", "box": [1, 2, 3, 4]})), "vlm_malformed"),
    (lambda r: _chat(json.dumps({"class_id": "other_class", "verdict": "present", "reason": "x"})), "vlm_contradictory"),
    (lambda r: _chat(None), "vlm_malformed"),
    (lambda r: httpx.Response(500, text="boom"), "vlm_unavailable"),
    (lambda r: httpx.Response(400, json={"error": "response_format json_schema not supported"}), "vlm_unavailable"),
    (lambda r: httpx.Response(200, json={"unexpected": True}), "vlm_unavailable"),
])
def test_vlm_bad_responses_become_uncertain(handler, reason):
    vlm, calls = _vlm_with(handler)
    res = vlm.verify(_img_obj(), _req(), "a blue X9 widget")
    assert res.verdict == "uncertain" and res.reason_code == reason
    assert len(calls) == 1  # exactly one attempt, no retries


def test_vlm_timeout_and_unreachable_become_uncertain():
    def timeout(r):
        raise httpx.ReadTimeout("slow", request=r)

    def refused(r):
        raise httpx.ConnectError("refused", request=r)

    for h, reason in ((timeout, "vlm_timeout"), (refused, "vlm_unavailable")):
        vlm, calls = _vlm_with(h)
        res = vlm.verify(_img_obj(), _req(), "x")
        assert res.verdict == "uncertain" and res.reason_code == reason and len(calls) == 1


def test_vlm_valid_answer_cannot_carry_a_box():
    vlm, _ = _vlm_with(lambda r: _chat(json.dumps({"class_id": CLASS_A, "verdict": "present", "reason": "blue"})))
    res = vlm.verify(_img_obj(), _req(), "x")
    assert res.verdict == "present"
    assert "box_xyxy" not in type(res).model_fields


def test_vlm_call_cap_holds_and_raising_adapter_is_contained(tmp_path):
    cfg = config(vlm={"enabled": True}, budgets={"max_vlm_calls": 2})
    entry = _img(tmp_path, size=(400, 300))
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, (10 + 60 * i, 10, 50 + 60 * i, 50), 0.9) for i in range(5)]})
    rel = ScriptedRelevance({"a": {CLASS_A: 0.05, CLASS_B: 0.05}})  # low relevance + detections -> verify all
    vlm = FakeVLM(raise_exc=RuntimeError("model crashed"))
    rec = _pipe(cfg, primary, tmp_path, relevance=rel, vlm=vlm).process(entry)
    assert len(vlm.requests) == 2
    assert rec.trace.budget_used.vlm_calls == 2
    assert "vlm_call_cap_reached" in rec.completeness.reason_codes
    assert all(v.verdict == "uncertain" for v in rec.verifications)
    assert not any(c.status == "accepted" for c in rec.final_candidates)
    assert rec.completeness.status == "unresolved"


def _img_obj():
    from PIL import Image
    return Image.new("RGB", (100, 100), "white")


# 8. Completeness -----------------------------------------------------------------------------


def test_unmatched_proposal_creates_review_state(tmp_path):
    entry = _img(tmp_path, size=(400, 300))
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, (10, 10, 60, 60), 0.9)]})
    secondary = ScriptedDetector({"a": [RawDetection(CLASS_A, (10, 10, 60, 60), 0.9),
                                        RawDetection(CLASS_A, (200, 150, 250, 200), 0.8)]}, "sec@1")
    rel = ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.1}})
    rec = _pipe(config(), primary, tmp_path, secondary=secondary, relevance=rel).process(entry)
    cand = next(c for c in rec.final_candidates if c.status == "candidate")
    assert cand.id in rec.completeness.unmatched_candidate_ids
    assert rec.completeness.status in ("needs_review", "unresolved")


def test_high_relevance_class_without_box_is_unresolved(tmp_path):
    entry = _img(tmp_path)
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, (10, 10, 120, 120), 0.95)]})
    rel = ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.9}})
    rec = _pipe(config(), primary, tmp_path, relevance=rel).process(entry)
    assert rec.completeness.suspected_missing_classes == (CLASS_B,)
    assert rec.completeness.status == "unresolved"
    assert f"high_relevance_zero_boxes_for_class:{CLASS_B}" in rec.completeness.reason_codes


def test_vlm_present_for_missing_class_never_creates_a_box(tmp_path):
    cfg = config(vlm={"enabled": True})
    entry = _img(tmp_path)
    rel = ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.1}})
    vlm = FakeVLM(default="present")
    rec = _pipe(cfg, ScriptedDetector({"a": []}), tmp_path, relevance=rel, vlm=vlm).process(entry)
    assert rec.final_candidates == ()
    assert any(r.purpose == "suspected_missing_class" for r, _ in vlm.requests)
    assert rec.completeness.status == "unresolved"
    assert f"vlm_reports_unannotated_instance:{CLASS_A}" in rec.completeness.reason_codes


def test_audited_zero_box_low_relevance_image_needs_review(tmp_path):
    cfg = config(audit={"low_relevance_sample_rate": 1.0})
    entry = _img(tmp_path)
    rel = ScriptedRelevance({"a": {CLASS_A: 0.05, CLASS_B: 0.05}})
    rec = _pipe(cfg, ScriptedDetector({"a": []}), tmp_path, relevance=rel).process(entry)
    assert "audit_sample" in rec.route.actions
    assert rec.completeness.status == "needs_review"
    assert "negative_not_verified" in rec.completeness.reason_codes


def test_complete_requires_human_or_calibrated_policy():
    from routing.completeness import audit
    from routing.schemas import AnnotationAudit, Budget, Quality, RouteDecision, StageA
    from pydantic import ValidationError
    cfg = config()
    sa = StageA(per_class={}, relevance_status="unavailable", relevance_band="unknown",
                quality=Quality(blur=0.1, exposure=0, min_side_px=500, low_resolution=False))
    dec = RouteDecision(actions=("standard",), priority=0, reason_codes=(), policy_version="v", expected_budget=Budget())
    assert audit([], sa, dec, [], cfg).status == "needs_review"
    assert audit([], sa, dec, [], cfg, human_outcome="complete").status == "complete"
    calibrated = cfg.model_copy(update={"audit": cfg.audit.model_copy(update={"calibrated": True})})
    assert audit([], sa, dec, [], calibrated).established_by == "calibrated_audit_policy"
    with pytest.raises(ValidationError):
        AnnotationAudit(status="complete")


# 9. Budget and recovery ----------------------------------------------------------------------


def test_slice_budget_blocks_sahi(tmp_path):
    cfg = config(slicing={"tile": 100, "overlap": 0.2}, budgets={"max_slices": 4})
    entry = _img(tmp_path, size=(640, 480))
    rel = ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.1}})
    det = BlobDetector("prim@1")
    rec = _pipe(cfg, det, tmp_path, relevance=rel).process(entry)
    assert "sahi" not in rec.route.actions and "slice_budget_exceeded" in rec.route.reason_codes
    assert det.calls == 1 and rec.trace.budget_used.slices == 0
    assert "human_review" in rec.route.actions


def test_elapsed_and_gpu_caps_stop_escalation(tmp_path):
    entry = _img(tmp_path, rects=[(CLASS_A, (100, 80, 140, 120))])
    rel = ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.1}})

    ticks = iter([0.0] + [1000.0] * 100)  # elapsed jumps past the cap after Stage A
    sec = BlobDetector("sec@1")
    rec = _pipe(config(budgets={"max_elapsed_ms": 500}), ScriptedDetector({"a": []}), tmp_path / "t",
                relevance=rel, secondary=sec, clock=lambda: next(ticks)).process(entry)
    assert sec.calls == 0 and "elapsed_budget_exhausted_before:secondary" in rec.completeness.reason_codes
    assert rec.completeness.status != "complete"

    sec2 = BlobDetector("sec@1")
    rec2 = _pipe(config(budgets={"max_gpu_ms": 10}), ScriptedDetector({"a": []}, gpu_ms=50), tmp_path / "g",
                 relevance=rel, secondary=sec2).process(entry)
    assert sec2.calls == 0 and "gpu_budget_exhausted_before:secondary" in rec2.completeness.reason_codes


class Crash(BaseException):
    pass


def test_resume_after_crash_does_not_repeat_calls(tmp_path):
    cfg = config(vlm={"enabled": True})
    entry = _img(tmp_path, size=(400, 300))
    preds = {"a": [RawDetection(CLASS_A, (10, 10, 60, 60), 0.9)]}
    sec_preds = {"a": [RawDetection(CLASS_A, (200, 150, 250, 200), 0.8)]}
    rel = ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.1}})
    store_path = tmp_path / "routes.sqlite"
    primary, secondary = ScriptedDetector(preds), ScriptedDetector(sec_preds, "sec@1")

    crashing = FakeVLM(raise_exc=Crash())
    with pytest.raises(Crash):
        RoutingPipeline(cfg, primary, RouteStore(store_path), secondary=secondary, relevance=rel,
                        vlm=crashing).process(entry)
    assert primary.calls == 1 and secondary.calls == 1

    vlm = FakeVLM(default="present")
    pipe = RoutingPipeline(cfg, primary, RouteStore(store_path), secondary=secondary, relevance=rel, vlm=vlm)
    rec = pipe.process(entry)
    assert primary.calls == 1 and secondary.calls == 1  # cached, not re-run
    assert "route:resumed" in rec.trace.events
    n_vlm = len(vlm.requests)
    assert n_vlm >= 1

    again = pipe.process(entry)
    assert again == rec and len(vlm.requests) == n_vlm and primary.calls == 1
    assert len(RouteStore(store_path).records()) == 1


def test_corrupt_image_goes_to_error_queue(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG garbage")
    store = RouteStore(tmp_path / "r.sqlite")
    pipe = RoutingPipeline(config(), ScriptedDetector({}), store)
    out = pipe.run([ManifestEntry(str(bad), "bad"), _img(tmp_path)])
    assert type(out[0]).__name__ == "ErrorRecord" and type(out[1]).__name__ == "RouteRecord"
    assert len(store.errors()) == 1


# 10. Data isolation --------------------------------------------------------------------------


def test_split_groups_must_not_overlap():
    check_isolation({"train": ["g1", "g2"], "calibration": ["g3"], "frozen_eval": ["g4"]})
    with pytest.raises(IsolationError):
        check_isolation({"train": ["g1", "g4"], "frozen_eval": ["g4"]})
    with pytest.raises(IsolationError):
        check_isolation({"calibration": ["g3"], "development": ["g3"]})


def test_frozen_set_is_locked_and_not_rewritten_by_threshold_changes(tmp_path):
    ann = {"img1": [[CLASS_A, [1, 2, 3, 4]]]}
    frozen_file = tmp_path / "frozen_annotations.json"
    frozen_file.write_text(json.dumps(ann), encoding="utf-8")
    lock = lock_frozen_set(["img1"], ["g9"], ann, tmp_path / "frozen.lock.json")
    with pytest.raises(FileExistsError):
        lock_frozen_set(["img1"], ["g9"], ann, tmp_path / "frozen.lock.json")
    before = frozen_file.read_bytes()

    entry = _img(tmp_path)
    for thr in (0.3, 0.6):
        cfg = config(relevance={"raw_high": thr + 0.1, "raw_low": thr})
        RoutingPipeline(cfg, ScriptedDetector({}), RouteStore(tmp_path / f"r{thr}.sqlite")).process(entry)
    assert frozen_file.read_bytes() == before
    assert load_frozen(tmp_path / "frozen.lock.json", json.loads(before))["image_ids"] == ["img1"]
    with pytest.raises(IsolationError):
        load_frozen(tmp_path / "frozen.lock.json", {"img1": []})
    with pytest.raises(IsolationError):
        assert_not_frozen({"imgX": "g9"}, lock, ["imgX"])
    assert_not_frozen({"imgY": "g1"}, lock, ["imgY"])


# 11. End-to-end ------------------------------------------------------------------------------


def test_end_to_end_shadow_and_legacy(tmp_path, capsys):
    fx = make_fixture(tmp_path / "fx")
    out = tmp_path / "shadow"
    assert cli.main(["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
                     "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
                     "--relevance-scores", str(fx["relevance"]), "--out", str(out)]) == 0
    doc = load_annotations(out / "annotations.json")
    assert {i["vad_image_id"] for i in doc["images"]} == {"easy", "zero", "small"}
    report = json.loads((out / "report.json").read_text())
    assert report["images"] == 3 and len(report["errors"]) == 1
    traces = [json.loads(line) for line in (out / "route_trace.jsonl").read_text().splitlines()]
    for t in traces:
        assert t["route"]["reason_codes"] and t["route"]["policy_version"] == "route-v1"
    by_id = {t["image"]["id"]: t for t in traces}
    assert by_id["easy"]["route"]["actions"] == ["standard"]
    assert "high_relevance_zero_boxes" in by_id["zero"]["route"]["reason_codes"]
    assert "small_candidate" in by_id["small"]["route"]["reason_codes"]

    # Re-running the same job is idempotent: same annotations version, no duplicate records.
    cli.main(["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
              "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
              "--relevance-scores", str(fx["relevance"]), "--out", str(out)])
    assert load_annotations(out / "annotations.json")["info"]["version_id"] == doc["info"]["version_id"]


def test_feature_flag_off_keeps_legacy_output(tmp_path):
    fx = make_fixture(tmp_path / "fx")
    (fx["root"] / "img_corrupt.png").unlink()
    lines = [json.loads(x) for x in fx["manifest"].read_text().splitlines()]
    fx["manifest"].write_text("".join(json.dumps(r) + "\n" for r in lines if r["image_id"] != "corrupt"))
    out = tmp_path / "legacy"
    assert cli.main(["legacy", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
                     "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
                     "--out", str(out)]) == 0
    doc = load_annotations(out / "annotations.json")
    preds = [json.loads(x) for x in fx["preds"].read_text().splitlines()]
    expected = [(p["image_id"], d["class_id"], d["box_xyxy"], d["confidence"]) for p in preds for d in p["detections"]]
    img_ids = {i["id"]: i["vad_image_id"] for i in doc["images"]}
    cats = {c["id"]: c["name"] for c in doc["categories"]}
    got = [(img_ids[a["image_id"]], cats[a["category_id"]],
            [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]], a["score"])
           for a in doc["annotations"]]
    assert got == expected

    from routing.task_config import load_config
    cfg = load_config(fx["config"])
    assert cfg.routing_enabled is False
    with pytest.raises(RuntimeError):
        RoutingPipeline(cfg, ScriptedDetector({}), RouteStore(":memory:")).process(ManifestEntry("x"))


@pytest.mark.parametrize("absent_rejects,status", [(False, "needs_review"), (True, "rejected")])
def test_hard_route_without_extra_detector_verifies_primary_boxes(tmp_path, absent_rejects, status):
    # Default (absent_rejects=False): an unmeasured VLM "absent" sends the box to review, flagged vlm_absent.
    cfg = config(vlm={"enabled": True, "absent_rejects": absent_rejects})
    entry = _img(tmp_path, size=(640, 480))
    primary = ScriptedDetector({"a": [RawDetection(CLASS_B, (300, 200, 308, 208), 0.4)]})
    rel = ScriptedRelevance({"a": {CLASS_A: 0.1, CLASS_B: 0.9}})
    vlm = FakeVLM(default="absent")
    rec = _pipe(cfg, primary, tmp_path, relevance=rel, vlm=vlm).process(entry)
    assert "high_miss_risk" in rec.route.reason_codes
    assert [r.purpose for r, _ in vlm.requests][0] == "confirm_candidate"
    assert [c.status for c in rec.final_candidates] == [status]
    assert ("vlm_absent" in rec.final_candidates[0].flags) is (not absent_rejects)
    assert rec.completeness.status != "complete"


class _Boom:
    supports_crops = False
    model_version = "boom@1"

    def __init__(self, inner):
        self.inner = inner

    def detect(self, image, image_id, class_ids):
        if image_id == "b1":
            raise RuntimeError("detector crashed")
        return self.inner.detect(image, image_id, class_ids)


def test_one_failing_image_does_not_abort_the_batch(tmp_path):
    entries = [_img(tmp_path, f"b{i}.png") for i in range(3)]
    inner = ScriptedDetector({
        "b0": [RawDetection(CLASS_A, (10, 10, 60, 60), float("nan")),
               RawDetection(CLASS_A, (70, 10, 120, 60), 1.0000001),
               RawDetection(CLASS_A, (130, 10, 130.001, 60), 0.9)],  # rounds to zero width
        "b2": [RawDetection(CLASS_A, (10, 10, 60, 60), 0.9)],
    })
    store = RouteStore(tmp_path / "r.sqlite")
    out = RoutingPipeline(config(), _Boom(inner), store).run(entries)
    assert [type(o).__name__ for o in out] == ["RouteRecord", "ErrorRecord", "RouteRecord"]
    assert out[1].error.startswith("routing_failed_at:primary")
    b0 = out[0]
    assert [d.raw_confidence for d in b0.primary_candidates] == [None, 1.0]
    assert {"primary:nonfinite_confidence_dropped:" + CLASS_A, "primary:confidence_clamped:" + CLASS_A,
            "primary:degenerate_box_dropped:" + CLASS_A} <= set(b0.trace.warnings)
    assert len(store.errors()) == 1


# Reviewer findings (round 1) ------------------------------------------------------------------


def test_r1_sahi_merged_parents_are_persisted(tmp_path):
    from pydantic import ValidationError
    cfg = config(slicing={"tile": 320, "overlap": 0.2, "include_full_image": False}, budgets={"max_slices": 16})
    entry = ManifestEntry(str(draw(tmp_path / "s.png", (640, 480), [(CLASS_B, (300, 200, 306, 206))])), image_id="s")
    rel = ScriptedRelevance({"s": {CLASS_A: 0.1, CLASS_B: 0.9}})
    rec = _pipe(cfg, BlobDetector("prim@1", min_frac=0.001), tmp_path, relevance=rel,
                sahi_detector=BlobDetector("prim@1")).process(entry)
    (merged,) = rec.sahi_candidates
    raw_ids = {d.id for d in rec.sahi_raw_candidates}
    assert len(merged.parent_ids) == 4 and set(merged.parent_ids) <= raw_ids  # seen by 4 overlapping tiles
    assert all(d.slice_origin is not None for d in rec.sahi_raw_candidates)
    data = rec.model_dump(mode="json")
    data["sahi_raw_candidates"] = data["sahi_raw_candidates"][1:]
    with pytest.raises(ValidationError, match="unpersisted parents"):
        type(rec).model_validate(data)


def test_r2_resume_reuses_persisted_stage_a(tmp_path):
    cfg = config(vlm={"enabled": True})
    entry = _img(tmp_path, size=(400, 300))
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, (10, 10, 60, 60), 0.9)]})
    secondary = ScriptedDetector({"a": [RawDetection(CLASS_A, (200, 150, 250, 200), 0.8)]}, "sec@1")
    store_path = tmp_path / "routes.sqlite"
    with pytest.raises(Crash):
        RoutingPipeline(cfg, primary, RouteStore(store_path), secondary=secondary,
                        relevance=ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.1}}),
                        vlm=FakeVLM(raise_exc=Crash())).process(entry)
    # same scorer version, but it would now return different scores: the record must keep the originals
    rec = RoutingPipeline(cfg, primary, RouteStore(store_path), secondary=secondary,
                          relevance=ScriptedRelevance({"a": {CLASS_A: 0.95, CLASS_B: 0.95}}),
                          vlm=FakeVLM()).process(entry)
    assert "route:resumed" in rec.trace.events
    assert rec.stage_a.per_class[CLASS_A].raw_similarity == 0.4
    assert rec.stage_a.relevance_band == "medium" and "moderate_relevance" in rec.route.reason_codes


class _WrongTargetVLM(FakeVLM):
    """Answers 'present' but for different candidates / region than requested."""

    def __init__(self, mutate):
        super().__init__(default="present")
        self.mutate = mutate

    def verify(self, image, request, class_desc):
        return super().verify(image, request, class_desc).model_copy(update=self.mutate)


@pytest.mark.parametrize("mutate", [{"candidate_ids": ("cand-someone-else",)}, {"candidate_ids": ()},
                                    {"region_xyxy": (0.0, 0.0, 5.0, 5.0)}])
def test_r3_vlm_result_must_match_request_candidates_and_region(tmp_path, mutate):
    cfg = config(vlm={"enabled": True})
    entry = _img(tmp_path)
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, (10, 10, 60, 60), 0.9)]})
    rel = ScriptedRelevance({"a": {CLASS_A: 0.05, CLASS_B: 0.05}})  # low relevance + box -> verify
    rec = _pipe(cfg, primary, tmp_path, relevance=rel, vlm=_WrongTargetVLM(mutate)).process(entry)
    (cand,) = rec.final_candidates
    assert cand.status == "needs_review"
    assert cand.verification.verdict == "uncertain" and cand.verification.reason_code == "vlm_contradictory"


class _ClockedBlob(BlobDetector):
    def __init__(self, clock, step_s, **kw):
        super().__init__(**kw)
        self.clock, self.step_s = clock, step_s

    def detect(self, image, image_id, class_ids):
        self.clock[0] += self.step_s
        return super().detect(image, image_id, class_ids)


def _sahi_setup(tmp_path):
    cfg_kw = {"slicing": {"tile": 160, "overlap": 0.0, "include_full_image": False}}
    entry = ManifestEntry(str(draw(tmp_path / "s.png", (640, 320), [(CLASS_B, (600, 300, 606, 306))])), image_id="s")
    rel = ScriptedRelevance({"s": {CLASS_A: 0.1, CLASS_B: 0.9}})
    return cfg_kw, entry, rel  # 8 tiles; the object sits in the last one


def test_r4_sahi_elapsed_budget_enforced_per_tile(tmp_path):
    cfg_kw, entry, rel = _sahi_setup(tmp_path)
    clock = [0.0]
    cfg = config(**cfg_kw, budgets={"max_slices": 16, "max_elapsed_ms": 350})
    det = _ClockedBlob(clock, 0.1, model_version="prim@1")
    rec = _pipe(cfg, ScriptedDetector({"s": []}, "prim@1"), tmp_path, relevance=rel, sahi_detector=det,
                clock=lambda: clock[0]).process(entry)
    assert det.calls == 4 and rec.trace.budget_used.slices == 4  # stopped before tile 5 of 8
    assert "sahi_partial:elapsed_budget_exhausted" in rec.completeness.reason_codes
    assert rec.sahi_candidates == () and rec.completeness.status != "complete"


def test_r4_sahi_gpu_budget_enforced_per_tile(tmp_path):
    cfg_kw, entry, rel = _sahi_setup(tmp_path)
    cfg = config(**cfg_kw, budgets={"max_slices": 16, "max_gpu_ms": 25})
    det = BlobDetector("prim@1", gpu_ms=10.0)
    rec = _pipe(cfg, ScriptedDetector({"s": []}, "prim@1", gpu_ms=0.0), tmp_path, relevance=rel,
                sahi_detector=det).process(entry)
    assert det.calls == 3 and rec.trace.budget_used.slices == 3
    assert "sahi_partial:gpu_budget_exhausted" in rec.completeness.reason_codes
    assert rec.completeness.status != "complete"


def test_r4_sahi_slice_cap_enforced_during_tiling():
    from routing.image_inputs import load_image  # noqa: F401  (import check only)
    from PIL import Image
    from routing.schemas import ImageRef
    img = Image.new("RGB", (640, 320), (200, 200, 200))
    ref = ImageRef(id="x", uri="x", width=640, height=320, sha256="0" * 64)
    det = BlobDetector()
    res = run_sliced(det, img, ref, task().class_ids, SlicingConfig(tile=160, overlap=0.0, include_full_image=False),
                     max_slices=3)
    assert det.calls == 3 and res.slices_run == 3 and res.stopped_reason == "slice_budget_exhausted"


def test_r5_cli_requires_secondary_version(tmp_path, capsys):
    fx = make_fixture(tmp_path / "fx")
    with pytest.raises(SystemExit) as exc:
        cli.main(["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
                  "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
                  "--secondary-predictions", str(fx["preds"]), "--out", str(tmp_path / "o")])
    assert exc.value.code == 2
    assert "--secondary-version is required" in capsys.readouterr().err


# Reviewer findings (round 2) ------------------------------------------------------------------


def _shadow(fx, out, preds=None):
    return cli.main(["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
                     "--primary-predictions", str(preds or fx["preds"]), "--primary-version", "prim@1",
                     "--relevance-scores", str(fx["relevance"]), "--out", str(out)])


def test_r2_missing_prediction_row_goes_to_error_queue_and_batch_continues(tmp_path):
    fx = make_fixture(tmp_path / "fx")
    rows = [json.loads(x) for x in fx["preds"].read_text().splitlines()]
    fx["preds"].write_text("".join(json.dumps(r) + "\n" for r in rows if r["image_id"] != "zero"))
    out = tmp_path / "o"
    assert _shadow(fx, out) == 0
    report = json.loads((out / "report.json").read_text())
    errors = {Path(e["uri"]).name: e["error"] for e in report["errors"]}
    assert errors["img_zero.png"].startswith("routing_failed_at:primary: MissingPrediction")
    assert "img_corrupt.png" in errors
    traces = [json.loads(x) for x in (out / "route_trace.jsonl").read_text().splitlines()]
    assert sorted(t["image"]["id"] for t in traces) == ["easy", "small"]  # unprocessed image is not a zero-box sample


def test_r2_explicit_empty_row_is_a_valid_zero_box_result(tmp_path):
    fx = make_fixture(tmp_path / "fx")
    assert _shadow(fx, tmp_path / "o") == 0
    traces = {json.loads(x)["image"]["id"]: json.loads(x) for x in (tmp_path / "o" / "route_trace.jsonl").read_text().splitlines()}
    assert traces["zero"]["primary_candidates"] == []
    assert "high_relevance_zero_boxes" in traces["zero"]["route"]["reason_codes"]


@pytest.mark.parametrize("rows,match", [
    ([{"image_id": "a"}], "needs 'image_id' and 'detections'"),
    ([{"detections": []}], "needs 'image_id' and 'detections'"),
    ([{"image_id": "a", "detections": []}, {"image_id": "a", "detections": []}], "duplicate image_id"),
])
def test_r2_precomputed_rows_must_be_explicit_and_unique(tmp_path, rows, match):
    from routing.replay import PrecomputedDetector
    p = tmp_path / "p.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match=match):
        PrecomputedDetector(p, "m@1")


def test_r2_changed_precomputed_file_with_same_version_does_not_reuse_records(tmp_path):
    from routing.replay import PrecomputedDetector
    entry = _img(tmp_path)
    preds = tmp_path / "preds.jsonl"
    store_path = tmp_path / "routes.sqlite"

    def run(boxes):
        preds.write_text(json.dumps({"image_id": "a", "detections": [
            {"class_id": CLASS_A, "box_xyxy": b, "confidence": 0.9} for b in boxes]}) + "\n")
        return RoutingPipeline(config(), PrecomputedDetector(preds, "same-version@1"), RouteStore(store_path)).process(entry)

    first = run([[10, 10, 60, 60]])
    again = run([[10, 10, 60, 60]])
    changed = run([[100, 100, 150, 150]])
    assert again.record_id == first.record_id  # identical content -> resumable, same record
    assert changed.record_id != first.record_id
    assert changed.run.primary_artifact_digest != first.run.primary_artifact_digest
    assert [d.box_xyxy for d in changed.primary_candidates] == [(100, 100, 150, 150)]
    assert len(RouteStore(store_path).records()) == 2
