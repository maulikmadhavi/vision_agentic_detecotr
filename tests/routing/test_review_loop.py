"""A7 stage 2: applying human adjudications to routed records, re-audit, findings queue and export."""

from __future__ import annotations

import json

import pytest
import yaml

from routing import cli
from routing.image_inputs import ManifestEntry
from routing.interfaces import RawDetection
from routing.pipeline import RoutingPipeline
from routing.review import ReviewError, append_adjudications, apply_adjudications, open_findings
from routing.schemas import Adjudication, RouteRecord
from routing.store import RouteStore

from .helpers import CLASS_A, CLASS_B, BlobDetector, FakeVLM, ScriptedDetector, ScriptedRelevance, config, draw

WHEN = "2026-11-10T10:00:00+08:00"
SMALL_A = (100, 100, 108, 108)  # tiny box -> high miss risk; no slicer/secondary -> candidate needs review


def _entry(tmp_path, name):
    return ManifestEntry(str(draw(tmp_path / f"{name}.png", (320, 240), [])), image_id=name)


def _run(tmp_path, name, cfg=None, scores=None, detector=None, **kw):
    cfg = cfg or config()
    scores = scores or {CLASS_A: 0.4, CLASS_B: 0.9}  # class B: high relevance, never detected
    detector = detector or ScriptedDetector({name: [RawDetection(CLASS_A, SMALL_A, 0.97)]})
    rec = RoutingPipeline(cfg, detector, RouteStore(":memory:"), relevance=ScriptedRelevance({name: scores}),
                          **kw).process(_entry(tmp_path, name))
    assert isinstance(rec, RouteRecord)
    return rec


class _CropBoom(BlobDetector):
    def detect(self, image, image_id, class_ids):
        raise RuntimeError("tile failed")


def _scenarios(tmp_path):
    vlm_cfg = config(vlm={"enabled": True})
    return {
        "R": (_run(tmp_path, "r"), config()),
        "F": (_run(tmp_path, "f", scores={CLASS_A: 0.9, CLASS_B: 0.05}, sahi_detector=_CropBoom()), config()),
        "cap0": (_run(tmp_path, "c", cfg=config(vlm={"enabled": True}, budgets={"max_vlm_calls": 0}),
                      vlm=FakeVLM()), config(vlm={"enabled": True}, budgets={"max_vlm_calls": 0})),
        "uncertain": (_run(tmp_path, "u", cfg=vlm_cfg, vlm=FakeVLM(default="uncertain")), vlm_cfg),
        "vlm_absent": (_run(tmp_path, "v", cfg=vlm_cfg, vlm=FakeVLM(default="absent")), vlm_cfg),
        "accepted": (_run(tmp_path, "e", scores={CLASS_A: 0.9, CLASS_B: 0.05},
                          detector=ScriptedDetector({"e": [RawDetection(CLASS_A, (40, 40, 200, 200), 0.97)]})),
                     config()),
    }


def _adj(rec, decision, **kw):
    return Adjudication(record_id=rec.record_id, image_id=rec.image.id, decision=decision, reviewer="ana",
                        decided_at=WHEN, **kw)


def _candidate(rec):
    (c,) = [c for c in rec.final_candidates if c.class_id == CLASS_A]
    return c


# --- re-audit invariant -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["R", "F", "cap0", "uncertain", "vlm_absent", "accepted"])
def test_no_adjudications_reproduce_the_original_audit(tmp_path, name):
    rec, cfg = _scenarios(tmp_path)[name]
    if name == "vlm_absent":  # the missing-class check answered absent: it must be re-used, not lost
        assert f"vlm_absent_for_suspected_class:{CLASS_B}" in rec.completeness.reason_codes
    (reviewed,) = apply_adjudications([rec], [], cfg)
    assert reviewed.completeness == rec.completeness and reviewed.final_candidates == rec.final_candidates
    assert reviewed.human_detections == () and reviewed.adjudications == ()


# --- findings -----------------------------------------------------------------------------------


def test_findings_queue_for_an_open_record(tmp_path):
    rec = _run(tmp_path, "r")
    findings = open_findings(rec)
    kinds = [(f["finding"], f.get("candidate_id") or f.get("class_id")) for f in findings]
    assert kinds == [("candidate", _candidate(rec).id), ("class", CLASS_B), ("image", None)]
    assert findings[-1]["decide_with"] == ["complete"] and findings[0]["decide_with"] == ["accept", "reject", "reclassify"]
    assert findings[1]["decide_with"] == ["class_absent", "add_box"]


def test_a_complete_record_has_no_findings(tmp_path):
    rec = _run(tmp_path, "e", scores={CLASS_A: 0.9, CLASS_B: 0.05},
               detector=ScriptedDetector({"e": [RawDetection(CLASS_A, (40, 40, 200, 200), 0.97)]}),
               cfg=config(audit={"calibrated": True}))
    assert rec.completeness.status == "complete" and open_findings(rec) == []


# --- closing findings ---------------------------------------------------------------------------


def test_a_bare_complete_cannot_close_open_findings(tmp_path):
    rec = _run(tmp_path, "r")
    (reviewed,) = apply_adjudications([rec], [_adj(rec, "complete")], config())
    assert reviewed.completeness.status != "complete"
    assert "human_complete_overridden_by_open_findings" in reviewed.completeness.reason_codes


def test_every_finding_closed_by_name_then_complete_closes_the_image(tmp_path):
    rec = _run(tmp_path, "r")
    adjs = [_adj(rec, "accept", candidate_id=_candidate(rec).id), _adj(rec, "class_absent", class_id=CLASS_B),
            _adj(rec, "complete")]
    (reviewed,) = apply_adjudications([rec], adjs, config())
    assert (reviewed.completeness.status, reviewed.completeness.established_by) == ("complete", "human")
    assert f"human_confirmed_absent:{CLASS_B}" in reviewed.completeness.reason_codes


def test_complete_without_naming_the_class_leaves_it_open(tmp_path):
    rec = _run(tmp_path, "r")
    adjs = [_adj(rec, "accept", candidate_id=_candidate(rec).id), _adj(rec, "complete")]
    (reviewed,) = apply_adjudications([rec], adjs, config())
    assert reviewed.completeness.status != "complete" and CLASS_B in reviewed.completeness.suspected_missing_classes


def test_add_box_closes_the_class_with_a_human_detection(tmp_path):
    rec = _run(tmp_path, "r")
    adjs = [_adj(rec, "accept", candidate_id=_candidate(rec).id),
            _adj(rec, "add_box", class_id=CLASS_B, box_xyxy=(10, 10, 60, 60)), _adj(rec, "complete")]
    (reviewed,) = apply_adjudications([rec], adjs, config())
    (human,) = reviewed.human_detections
    assert (human.source, human.class_id, human.raw_confidence) == ("human", CLASS_B, None)
    added = [c for c in reviewed.final_candidates if c.source_ids == (human.id,)]
    assert added and added[0].status == "accepted" and reviewed.completeness.status == "complete"


def test_reject_and_reclassify_change_only_the_reviewed_candidate(tmp_path):
    rec = _run(tmp_path, "r")
    cand = _candidate(rec)
    (rejected,) = apply_adjudications([rec], [_adj(rec, "reject", candidate_id=cand.id)], config())
    assert next(c for c in rejected.final_candidates if c.id == cand.id).status == "rejected"
    (moved,) = apply_adjudications([rec], [_adj(rec, "reclassify", candidate_id=cand.id, class_id=CLASS_B)], config())
    changed = next(c for c in moved.final_candidates if c.id == cand.id)
    assert (changed.class_id, changed.status) == (CLASS_B, "accepted")
    assert _candidate(rec).status == "needs_review"  # the model's record is untouched


def test_failed_escalation_closes_only_with_complete_after_everything_else(tmp_path):
    rec, cfg = _scenarios(tmp_path)["F"]
    assert any(r.startswith("sahi_failed:") for r in rec.completeness.reason_codes)
    accept = _adj(rec, "accept", candidate_id=_candidate(rec).id)
    (without_complete,) = apply_adjudications([rec], [accept], cfg)
    assert without_complete.completeness.status == "needs_review"
    (closed,) = apply_adjudications([rec], [accept, _adj(rec, "complete")], cfg)
    assert closed.completeness.status == "complete"
    assert "escalation_failures_closed_by_human_complete" in closed.completeness.reason_codes


def test_complete_does_not_close_failed_escalation_while_a_candidate_is_open(tmp_path):
    rec, cfg = _scenarios(tmp_path)["F"]
    (reviewed,) = apply_adjudications([rec], [_adj(rec, "complete")], cfg)  # the candidate is not decided
    assert reviewed.completeness.status != "complete"
    assert "escalation_failures_closed_by_human_complete" not in reviewed.completeness.reason_codes


def test_a_decided_candidate_no_longer_counts_as_an_uncertain_verification(tmp_path):
    rec, cfg = _scenarios(tmp_path)["uncertain"]
    assert "verification_uncertain" in rec.completeness.reason_codes
    cand = _candidate(rec)
    adjs = [_adj(rec, "reject", candidate_id=cand.id)]
    adjs += [_adj(rec, "class_absent", class_id=c) for c in rec.completeness.suspected_missing_classes]
    (reviewed,) = apply_adjudications([rec], adjs + [_adj(rec, "complete")], cfg)
    assert reviewed.completeness.status == "complete"


def test_a_later_decision_on_the_same_finding_supersedes_the_earlier_one(tmp_path):
    rec = _run(tmp_path, "r")
    cand = _candidate(rec)
    adjs = [_adj(rec, "reject", candidate_id=cand.id), _adj(rec, "accept", candidate_id=cand.id, reason="second look")]
    (reviewed,) = apply_adjudications([rec], adjs, config())
    assert next(c for c in reviewed.final_candidates if c.id == cand.id).status == "accepted"
    assert [a.decision for a in reviewed.adjudications] == ["reject", "accept"]  # history kept


# --- R5-3: a class cannot be both present (an accepted box) and absent -------------------------


def test_accepting_a_box_and_declaring_its_class_absent_is_a_conflict(tmp_path):
    rec = _run(tmp_path, "r")
    cand = _candidate(rec)
    adjs = [_adj(rec, "accept", candidate_id=cand.id), _adj(rec, "class_absent", class_id=CLASS_A),
            _adj(rec, "class_absent", class_id=CLASS_B), _adj(rec, "complete")]
    with pytest.raises(ReviewError, match=f"{cand.id}.*{CLASS_A}"):
        apply_adjudications([rec], adjs, config())


def test_reclassifying_a_box_into_a_class_declared_absent_is_a_conflict(tmp_path):
    rec = _run(tmp_path, "r")
    cand = _candidate(rec)
    adjs = [_adj(rec, "reclassify", candidate_id=cand.id, class_id=CLASS_B), _adj(rec, "class_absent", class_id=CLASS_B)]
    with pytest.raises(ReviewError, match=f"{cand.id}.*{CLASS_B}"):
        apply_adjudications([rec], adjs, config())


def test_declaring_absent_the_class_of_a_box_the_model_already_accepted_is_a_conflict(tmp_path):
    rec, cfg = _scenarios(tmp_path)["accepted"]
    cand = _candidate(rec)
    assert cand.status == "accepted"  # accepted by the routing policy, no human decision
    with pytest.raises(ReviewError, match=f"{cand.id}.*{CLASS_A}"):
        apply_adjudications([rec], [_adj(rec, "class_absent", class_id=CLASS_A)], cfg)


@pytest.mark.parametrize("decide", [
    lambda rec, cand: [_adj(rec, "reject", candidate_id=cand.id)],
    lambda rec, cand: [_adj(rec, "accept", candidate_id=cand.id), _adj(rec, "reject", candidate_id=cand.id)],
    lambda rec, cand: [_adj(rec, "reclassify", candidate_id=cand.id, class_id=CLASS_B)],
], ids=["reject", "accept-then-reject", "reclassified-away"])
def test_absence_is_allowed_once_no_accepted_box_of_that_class_remains(tmp_path, decide):
    rec = _run(tmp_path, "r")
    cand = _candidate(rec)
    adjs = decide(rec, cand) + [_adj(rec, "class_absent", class_id=CLASS_A), _adj(rec, "complete")]
    (reviewed,) = apply_adjudications([rec], adjs, config())
    assert not [c for c in reviewed.final_candidates if c.status == "accepted" and c.class_id == CLASS_A]


# --- refusals -----------------------------------------------------------------------------------


@pytest.mark.parametrize("mutate,match", [
    (lambda rec: {"record_id": "rec-unknown"}, "not in the trace"),
    (lambda rec: {"image_id": "other"}, "image"),
    (lambda rec: {"decision": "accept", "candidate_id": "cand-nope"}, "cand-nope"),
    (lambda rec: {"decision": "class_absent", "class_id": "dog"}, "dog"),
    (lambda rec: {"decision": "add_box", "class_id": CLASS_B, "box_xyxy": (10, 10, 400, 60)}, "exceeds image"),
])
def test_adjudications_that_do_not_fit_the_trace_are_refused(tmp_path, mutate, match):
    rec = _run(tmp_path, "r")
    fields = {"record_id": rec.record_id, "image_id": rec.image.id, "decision": "complete", **mutate(rec)}
    adj = Adjudication(reviewer="ana", decided_at=WHEN, **fields)
    with pytest.raises(ReviewError, match=match):
        apply_adjudications([rec], [adj], config())


def test_the_run_config_must_match(tmp_path):
    rec = _run(tmp_path, "r")
    with pytest.raises(ReviewError, match="config"):
        apply_adjudications([rec], [], config(relevance={"raw_high": 0.8, "raw_low": 0.2}))


# --- CLI ----------------------------------------------------------------------------------------


def _cli_files(tmp_path, adjs_for):
    rec = _run(tmp_path, "r")
    trace = tmp_path / "route_trace.jsonl"
    trace.write_text(rec.model_dump_json() + "\n", encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config().model_dump(mode="json")), encoding="utf-8")
    log = tmp_path / "adjudications.jsonl"
    append_adjudications(log, adjs_for(rec))
    return rec, trace, cfg, log


def _apply(cfg, trace, log, out, *extra):
    return cli.main(["review", "apply", "--config", str(cfg), "--trace", str(trace), "--adjudications", str(log),
                     "--out-dir", str(out), *extra])


def _full_review(rec):
    return [_adj(rec, "reject", candidate_id=_candidate(rec).id),
            _adj(rec, "add_box", class_id=CLASS_B, box_xyxy=(10, 10, 60, 60)), _adj(rec, "complete")]


def test_cli_apply_exports_accepted_and_human_boxes_only_and_replays_identically(tmp_path):
    rec, trace, cfg, log = _cli_files(tmp_path, _full_review)
    before = trace.read_bytes()
    assert _apply(cfg, trace, log, tmp_path / "a") == 0
    assert _apply(cfg, trace, log, tmp_path / "b") == 0
    for name in ("reviewed.jsonl", "annotations.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert trace.read_bytes() == before  # model suggestions are never rewritten
    doc = json.loads((tmp_path / "a" / "annotations.json").read_text(encoding="utf-8"))
    (ann,) = doc["annotations"]  # the rejected model box is gone; the human box is exported
    assert ann["vad_source"] == "human" and "score" not in ann and ann["bbox"] == [10, 10, 50, 50]
    assert doc["images"][0]["vad_completeness"] == "complete" and doc["info"]["kind"] == "reviewed"


def test_cli_findings_writes_one_line_per_open_finding(tmp_path):
    rec, trace, cfg, log = _cli_files(tmp_path, lambda r: [])
    out = tmp_path / "findings.jsonl"
    assert cli.main(["review", "findings", "--trace", str(trace), "--out", str(out)]) == 0
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [f["finding"] for f in lines] == ["candidate", "class", "image"]
    assert {f["record_id"] for f in lines} == {rec.record_id}


@pytest.mark.parametrize("version", ["split-v1", "split-v2"])
def test_cli_apply_tags_frozen_images_and_warns(tmp_path, capsys, version):
    rec, trace, cfg, log = _cli_files(tmp_path, _full_review)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"version": version, "images": {rec.image.id: {"split": "frozen_eval"}}}),
                     encoding="utf-8")
    assert _apply(cfg, trace, log, tmp_path / "out", "--split-manifest", str(split)) == 0
    doc = json.loads((tmp_path / "out" / "annotations.json").read_text(encoding="utf-8"))
    assert doc["images"][0]["vad_split"] == "frozen_eval"
    assert "3 adjudications on frozen_eval images" in capsys.readouterr().err
