"""A7 stage 1: Adjudication / ReviewedRecord schemas and the append-only adjudications log."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from routing.review import AdjudicationLogError, append_adjudications, read_adjudications
from routing.schemas import (
    EXPORTED_MODELS, Adjudication, AnnotationAudit, Detection, FinalCandidate, ImageRef, ReviewedRecord,
    check_reviewer_handle,
)

from .helpers import CLASS_A, CLASS_B, task

WHEN = "2026-11-10T09:30:00+08:00"
BASE = {"record_id": "rec-1", "image_id": "a", "reviewer": "ana", "decided_at": WHEN}


def _adj(**kw):
    return Adjudication(**{**BASE, **kw})


# --- decisions ----------------------------------------------------------------------------------


@pytest.mark.parametrize("fields", [
    {"decision": "accept", "candidate_id": "cand-1"},
    {"decision": "reject", "candidate_id": "cand-1", "reason": "a shadow, not an object"},
    {"decision": "reclassify", "candidate_id": "cand-1", "class_id": CLASS_B},
    {"decision": "add_box", "class_id": CLASS_A, "box_xyxy": (10, 10, 40, 40)},
    {"decision": "class_absent", "class_id": CLASS_B},
    {"decision": "complete"},
])
def test_each_decision_kind_validates(fields):
    adj = _adj(**fields)
    assert adj.source == "human" and adj.id.startswith("adj-")


@pytest.mark.parametrize("fields,match", [
    ({"decision": "accept"}, "candidate_id"),
    ({"decision": "reject", "candidate_id": "cand-1", "class_id": CLASS_A}, "class_id"),
    ({"decision": "reclassify", "candidate_id": "cand-1"}, "class_id"),
    ({"decision": "add_box", "class_id": CLASS_A}, "box_xyxy"),
    ({"decision": "add_box", "box_xyxy": (1, 1, 5, 5)}, "class_id"),
    ({"decision": "add_box", "class_id": CLASS_A, "box_xyxy": (1, 1, 5, 5), "candidate_id": "cand-1"}, "candidate_id"),
    ({"decision": "class_absent"}, "class_id"),
    ({"decision": "complete", "class_id": CLASS_A}, "class_id"),
    ({"decision": "accept", "candidate_id": "cand-1", "box_xyxy": (1, 1, 5, 5)}, "box_xyxy"),
    ({"decision": "add_box", "class_id": CLASS_A, "box_xyxy": (5, 5, 1, 1)}, "box"),
    ({"decision": "complete", "decided_at": "yesterday"}, "decided_at"),
    ({"decision": "complete", "decided_at": "2026-11-10T09:30:00"}, "time zone"),
    ({"decision": "maybe"}, "decision"),
    ({"decision": "complete", "source": "vlm"}, "source"),
    ({"decision": "complete", "confidence": 0.9}, "confidence"),
])
def test_invalid_adjudications_are_refused(fields, match):
    with pytest.raises(ValidationError, match=match):
        _adj(**fields)


@pytest.mark.parametrize("handle", ["", "qwen3-vl-4b", "openai-compatible:m@host", "rev/1"])
def test_reviewer_must_be_a_human_handle(handle):
    with pytest.raises(ValidationError, match="reviewer"):
        _adj(decision="complete", reviewer=handle)


def test_the_handle_error_names_the_matched_token_and_why():
    with pytest.raises(ValueError) as exc:
        check_reviewer_handle("Eclipse Lee", "first")
    assert "'clip'" in str(exc.value) and "guard against pasted model ids" in str(exc.value)


def test_id_is_content_derived_and_tamper_evident():
    a = _adj(decision="reject", candidate_id="cand-1")
    assert a.id == _adj(decision="reject", candidate_id="cand-1").id != _adj(decision="accept", candidate_id="cand-1").id
    assert a.id != _adj(decision="reject", candidate_id="cand-1", reviewer="ben").id
    with pytest.raises(ValidationError, match="id"):
        Adjudication(**{**a.model_dump(), "reason": "edited after the fact"})


# --- the append-only log ------------------------------------------------------------------------


def test_log_round_trips_and_only_ever_appends(tmp_path):
    log = tmp_path / "adjudications.jsonl"
    first = [_adj(decision="reject", candidate_id="cand-1"), _adj(decision="class_absent", class_id=CLASS_B)]
    append_adjudications(log, first)
    before = log.read_bytes()
    append_adjudications(log, [_adj(decision="complete")])
    assert log.read_bytes().startswith(before)  # earlier lines untouched
    assert [a.decision for a in read_adjudications(log)] == ["reject", "class_absent", "complete"]


def test_repeating_the_same_adjudication_is_read_once(tmp_path):
    log = tmp_path / "adjudications.jsonl"
    adj = _adj(decision="accept", candidate_id="cand-1")
    append_adjudications(log, [adj])
    append_adjudications(log, [adj])
    assert read_adjudications(log) == [adj]


def test_a_malformed_line_names_its_line_number(tmp_path):
    log = tmp_path / "adjudications.jsonl"
    append_adjudications(log, [_adj(decision="complete")])
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({**BASE, "decision": "accept"}) + "\n")  # accept without a candidate
    with pytest.raises(AdjudicationLogError, match="line 2"):
        read_adjudications(log)


def test_a_missing_log_reads_as_empty(tmp_path):
    assert read_adjudications(tmp_path / "none.jsonl") == []


# --- ReviewedRecord -----------------------------------------------------------------------------


REF = ImageRef(id="a", uri="a.png", width=100, height=100, sha256="0" * 64)


def _human_box(**kw):
    return Detection(**{"id": "hum-1", "class_id": CLASS_A, "box_xyxy": (10, 10, 40, 40), "source": "human",
                        "model_version": "human:ana", **kw})


def _reviewed(**kw):
    adj = _adj(decision="add_box", class_id=CLASS_A, box_xyxy=(10, 10, 40, 40))
    fields = {"record_id": "rec-1", "image": REF, "task": task(), "adjudications": (adj,),
              "human_detections": (_human_box(),),
              "final_candidates": (FinalCandidate(id="cand-h", class_id=CLASS_A, box_xyxy=(10, 10, 40, 40),
                                                  status="accepted", source_ids=("hum-1",),
                                                  canonical_source_id="hum-1"),),
              "completeness": AnnotationAudit(status="needs_review")}
    return ReviewedRecord(**{**fields, **kw})


def test_reviewed_record_validates():
    assert _reviewed().human_detections[0].source == "human"


def test_reviewed_record_refuses_model_detections_as_human_labels():
    with pytest.raises(ValidationError, match="human"):
        _reviewed(human_detections=(_human_box(source="primary"),))


def test_reviewed_record_refuses_adjudications_for_another_record_or_image():
    with pytest.raises(ValidationError, match="rec-2"):
        _reviewed(adjudications=(_adj(decision="complete", record_id="rec-2"),))
    with pytest.raises(ValidationError, match="image"):
        _reviewed(adjudications=(_adj(decision="complete", image_id="b"),))


def test_reviewed_record_checks_boxes_and_classes_against_the_task():
    with pytest.raises(ValidationError, match="exceeds image"):
        _reviewed(human_detections=(_human_box(box_xyxy=(10, 10, 140, 40)),))
    with pytest.raises(ValidationError, match="unknown class_id"):
        _reviewed(human_detections=(_human_box(class_id="dog"),))


def test_both_models_are_exported_as_json_schemas():
    assert Adjudication in EXPORTED_MODELS and ReviewedRecord in EXPORTED_MODELS
