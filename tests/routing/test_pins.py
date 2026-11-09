"""Behavior pins for paths the default suite never executed (cross-check findings U1-U12).

One behavior per test. Line numbers refer to the code at the time the pin was written, so a
refactor that moves the code keeps the behavior, not the line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from routing import cli, pipeline as pipeline_mod
from routing.completeness import audit
from routing.image_inputs import ManifestEntry
from routing.interfaces import RawDetection
from routing.pipeline import RoutingPipeline
from routing.policy import Capabilities, route, stable_sample
from routing.schemas import (
    BoxFeatures, Budget, Difficulty, ErrorRecord, FinalCandidate, Quality, RouteDecision, RouteRecord, StageA,
)
from routing.store import RouteStore

from .helpers import CLASS_A, CLASS_B, BlobDetector, FakeVLM, ScriptedDetector, ScriptedRelevance, config, draw

REPO = Path(__file__).resolve().parents[2]


class Kill(BaseException):
    """A process death: escapes ``except Exception`` like a real crash would."""


def _entry(tmp_path, rects=(), size=(320, 240), name="a"):
    return ManifestEntry(str(draw(tmp_path / f"{name}.png", size, list(rects))), image_id=name)


def _moderate_box():
    # Medium relevance, one primary box, no secondary: route = VLM-then-human, box needs_review.
    return (ScriptedDetector({"a": [RawDetection(CLASS_A, (40, 40, 200, 200), 0.97)]}),
            ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.1}}))


def _crash_before_record(store: RouteStore) -> RouteStore:
    def put_record(rec):
        raise Kill()
    store.put_record = put_record
    return store


# --- U1: VLM recovery, "result exists -> CachedAnswer" -----------------------------------------


def test_u1_resume_reuses_a_persisted_vlm_answer_without_resending(tmp_path):
    """Pins pipeline.py:330-332: a saved ``vlm:<id>`` answer is returned on resume, never re-sent."""
    primary, rel = _moderate_box()
    cfg, path = config(vlm={"enabled": True}), tmp_path / "r.sqlite"
    first = FakeVLM(default="present")
    with pytest.raises(Kill):
        RoutingPipeline(cfg, primary, _crash_before_record(RouteStore(path)), relevance=rel, vlm=first).process(
            _entry(tmp_path))
    assert len(first.requests) == 1
    again = FakeVLM(default="absent")  # would change the outcome if it were asked
    rec = RoutingPipeline(cfg, primary, RouteStore(path), relevance=rel, vlm=again).process(_entry(tmp_path))
    (c,) = rec.final_candidates
    assert again.requests == [] and c.verification.verdict == "present" and c.status == "accepted"


def test_u1_cached_vlm_answers_count_toward_the_call_budget(tmp_path):
    """Pins pipeline.py:329 (``calls += 1`` before the cache lookup): resumed calls still count."""
    primary, rel = _moderate_box()
    cfg, path = config(vlm={"enabled": True}), tmp_path / "r.sqlite"
    with pytest.raises(Kill):
        RoutingPipeline(cfg, primary, _crash_before_record(RouteStore(path)), relevance=rel,
                        vlm=FakeVLM()).process(_entry(tmp_path))
    rec = RoutingPipeline(cfg, primary, RouteStore(path), relevance=rel, vlm=FakeVLM()).process(_entry(tmp_path))
    assert rec.trace.budget_used.vlm_calls == 1


# --- U2 / U3: slicing failure and cached slicing on resume --------------------------------------


class _CropBoom(BlobDetector):
    def detect(self, image, image_id, class_ids):
        raise RuntimeError("tile inference failed")


def _small_object(tmp_path):
    entry = _entry(tmp_path, [(CLASS_A, (150, 110, 158, 118))])  # tiny box -> high_miss_risk -> sahi
    return entry, ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.05}})


def test_u2_slicing_failure_is_a_review_reason_not_an_error(tmp_path):
    """Pins pipeline.py:245-246: an exception from sliced inference becomes ``sahi_failed:<Type>``."""
    entry, rel = _small_object(tmp_path)
    rec = RoutingPipeline(config(), BlobDetector(), RouteStore(":memory:"), relevance=rel,
                          sahi_detector=_CropBoom()).process(entry)
    assert isinstance(rec, RouteRecord) and "sahi" in rec.route.actions
    assert rec.sahi_candidates == () and "sahi_failed:RuntimeError" in rec.completeness.reason_codes


def test_u3_resume_reuses_the_cached_slicing_result(tmp_path):
    """Pins pipeline.py:230-231 (cached ``sahi`` step): resume does not re-run tiles."""
    entry, rel = _small_object(tmp_path)
    path = tmp_path / "r.sqlite"
    crop = BlobDetector()
    with pytest.raises(Kill):
        RoutingPipeline(config(), BlobDetector(), _crash_before_record(RouteStore(path)), relevance=rel,
                        sahi_detector=crop).process(entry)
    assert crop.calls > 0
    crop_again = BlobDetector()
    rec = RoutingPipeline(config(), BlobDetector(), RouteStore(path), relevance=rel,
                          sahi_detector=crop_again).process(entry)
    assert crop_again.calls == 0 and len(rec.sahi_candidates) == 1


# --- U4 / U5 / U6: VLM loop skips ---------------------------------------------------------------


def test_u4_an_already_accepted_candidate_is_not_re_verified(tmp_path):
    """Pins pipeline.py:352-354: only candidate/needs_review boxes are sent to the VLM."""
    entry, rel = _small_object(tmp_path)
    vlm = FakeVLM(default="absent")
    rec = RoutingPipeline(config(vlm={"enabled": True}), BlobDetector(), RouteStore(":memory:"), relevance=rel,
                          sahi_detector=BlobDetector(), vlm=vlm).process(entry)
    (c,) = rec.final_candidates  # primary + sahi agree -> accepted before verification
    assert "vlm_verify" in rec.route.actions and c.status == "accepted" and c.verification is None
    assert not [r for r, _ in vlm.requests if r.purpose == "confirm_candidate"]


def test_u5_missing_class_check_skipped_at_the_call_cap(tmp_path):
    """Pins pipeline.py:372-374: a missing-class check past the cap is skipped and recorded."""
    rel = ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.05}})  # high relevance, zero boxes
    vlm = FakeVLM()
    rec = RoutingPipeline(config(vlm={"enabled": True}, budgets={"max_vlm_calls": 0}), ScriptedDetector({}),
                          RouteStore(":memory:"), relevance=rel, vlm=vlm).process(_entry(tmp_path))
    assert "vlm_verify" in rec.route.actions and vlm.requests == []
    assert "vlm_call_cap_reached" in rec.completeness.reason_codes


def test_u6_candidate_check_skipped_when_the_elapsed_budget_is_spent(tmp_path):
    """Pins pipeline.py:355-357: an exhausted budget skips the VLM, leaves the box unchanged, records why."""
    primary, rel = _moderate_box()
    ticks = iter([0.0])  # first read is t0; every later read is far past the budget
    vlm = FakeVLM()
    rec = RoutingPipeline(config(vlm={"enabled": True}), primary, RouteStore(":memory:"), relevance=rel, vlm=vlm,
                          clock=lambda: next(ticks, 1e6)).process(_entry(tmp_path))
    (c,) = rec.final_candidates
    assert vlm.requests == [] and c.status == "needs_review" and c.verification is None
    assert "vlm_skipped_budget" in rec.completeness.reason_codes


# --- U7: errors after the primary step are queued with their step ------------------------------


def test_u7_stage_a_failure_is_queued_with_its_step(tmp_path, monkeypatch):
    """Pins pipeline.py:122-127 for step ``stage_a``."""
    def boom(*a, **k):
        raise RuntimeError("stage a broke")
    monkeypatch.setattr(pipeline_mod, "stage_a", boom)
    out = RoutingPipeline(config(), ScriptedDetector({}), RouteStore(":memory:")).process(_entry(tmp_path))
    assert isinstance(out, ErrorRecord) and out.error.startswith("routing_failed_at:stage_a: RuntimeError")


def test_u7_audit_failure_is_queued_with_the_record_step(tmp_path, monkeypatch):
    """Pins pipeline.py:122-127 for step ``record`` (completeness audit / record construction)."""
    def boom(*a, **k):
        raise RuntimeError("audit broke")
    monkeypatch.setattr(pipeline_mod, "audit", boom)
    out = RoutingPipeline(config(), ScriptedDetector({}), RouteStore(":memory:")).process(_entry(tmp_path))
    assert isinstance(out, ErrorRecord) and out.error.startswith("routing_failed_at:record: RuntimeError")


def test_u7_failed_record_write_is_queued_as_an_error(tmp_path):
    """Pins pipeline.py:282 + 122-127: a failing ``put_record`` becomes a persisted error, not a crash."""
    store = RouteStore(tmp_path / "r.sqlite")

    def put_record(rec):
        raise OSError("disk full")
    store.put_record = put_record
    out = RoutingPipeline(config(), ScriptedDetector({}), store).process(_entry(tmp_path))
    assert isinstance(out, ErrorRecord) and out.error.startswith("routing_failed_at:record: OSError")
    assert [e.record_id for e in RouteStore(tmp_path / "r.sqlite").errors()] == [out.record_id]


# --- U8: unreadable file ------------------------------------------------------------------------


def test_u8_missing_image_file_goes_to_the_error_queue(tmp_path):
    """Pins pipeline.py:104-108: a file that cannot be read (content digest unavailable) is queued."""
    out = RoutingPipeline(config(), ScriptedDetector({}), RouteStore(":memory:")).process(
        ManifestEntry(str(tmp_path / "missing.png"), image_id="m"))
    assert isinstance(out, ErrorRecord) and out.error.startswith("FileNotFoundError")


# --- U9: export-schema through the CLI ----------------------------------------------------------


def test_u9_cli_export_schema_writes_the_committed_schemas(tmp_path):
    """Pins cli.py:89-91: ``export-schema`` needs no config and reproduces schemas/ byte for byte."""
    assert cli.main(["export-schema", "--out", str(tmp_path)]) == 0
    written = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    committed = {p.name: p.read_bytes() for p in (REPO / "schemas").glob("*.json")}
    assert written == committed


# --- U10: route branches ------------------------------------------------------------------------


def _sa_high():
    return StageA(per_class={}, relevance_status="uncalibrated", relevance_band="high",
                  quality=Quality(blur=0.1, exposure=0.0, min_side_px=480, low_resolution=False))


def _features(crowding=0.0):
    return BoxFeatures(num_boxes=2, per_class_counts={CLASS_A: 2}, min_box_area_fraction=0.1, crowding=crowding,
                       class_ambiguity=0, mean_confidence=0.9, min_confidence=0.9, high_relevance_zero_boxes=False)


EASY = Difficulty(value=0.1, status="uncalibrated", bucket="easy")
HARD = Difficulty(value=0.9, status="uncalibrated", bucket="hard")


def test_u10_crowding_alone_escalates():
    """Pins policy.py:96-98: severe crowding without small boxes or a hard bucket is high miss risk."""
    d = route("i", 640, 480, _sa_high(), _features(crowding=0.9), EASY, Capabilities(), config())
    assert {"high_miss_risk", "severe_crowding"} <= set(d.reason_codes) and "small_candidate" not in d.reason_codes


def test_u10_hard_difficulty_alone_escalates():
    """Pins policy.py:92-94 (branch 94->96): a hard bucket escalates without small/crowded reasons."""
    d = route("i", 640, 480, _sa_high(), _features(), HARD, Capabilities(), config())
    assert "high_miss_risk" in d.reason_codes
    assert not {"small_candidate", "severe_crowding"} & set(d.reason_codes)


def test_u10_zero_sample_rate_never_samples():
    """Pins policy.py:27-28: a rate of 0 never samples, for any image id."""
    assert not any(stable_sample(f"img{i}", "seed", 0.0) for i in range(200))


# --- U11: completeness branches -----------------------------------------------------------------


DEC = RouteDecision(actions=("standard",), priority=0, reason_codes=(), policy_version="v", expected_budget=Budget())


def _accepted(box=(10, 10, 50, 50)):
    return FinalCandidate(id="cand-1", class_id=CLASS_A, box_xyxy=box, status="accepted", source_ids=("d1",),
                          canonical_source_id="d1")


def test_u11_calibrated_audit_policy_with_nothing_open_is_complete():
    """Pins completeness.py:104-105: authority from a calibrated audit policy and no findings -> complete."""
    cfg = config(audit={"calibrated": True})
    assert audit([_accepted()], _sa_high(), DEC, [], cfg).status == "complete"


def test_u11_accepted_box_not_in_the_reference_set_is_unmatched():
    """Pins completeness.py:40-45: reference comparison marks uncovered accepted boxes unmatched."""
    a = audit([_accepted()], _sa_high(), DEC, [], config(), reference_boxes=[(CLASS_A, (200, 200, 260, 260))])
    assert a.unmatched_candidate_ids == ("cand-1",) and "compared_against_reference_annotations" in a.reason_codes


# --- U12: VLM "present" on a detector proposal with acceptance disabled -------------------------


def test_u12_present_on_a_detector_proposal_stays_in_review_by_default(tmp_path):
    """Pins pipeline.py:399-402: without accept_confirmed_detector_proposals, "present" does not accept."""
    entry = _entry(tmp_path, [(CLASS_B, (100, 80, 140, 120))])
    rel = ScriptedRelevance({"a": {CLASS_A: 0.1, CLASS_B: 0.9}})
    rec = RoutingPipeline(config(vlm={"enabled": True}), ScriptedDetector({}), RouteStore(":memory:"),
                          secondary=BlobDetector("sec@1"), relevance=rel, vlm=FakeVLM(default="present")).process(entry)
    (c,) = [c for c in rec.final_candidates if c.class_id == CLASS_B]
    assert "detector_proposal:secondary" in c.flags and c.verification.verdict == "present"
    assert c.status == "needs_review"
