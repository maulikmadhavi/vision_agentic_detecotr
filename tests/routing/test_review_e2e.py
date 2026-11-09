"""A7 stage 3: the human review loop end to end on the checked-in fixture (no services, no real images).

shadow run -> `review findings` -> adjudications closing every finding -> image complete;
the same run with only bare `complete` decisions -> images with open findings stay open;
the reviewed export holds accepted model boxes plus human boxes and never a rejected box;
replaying the same adjudication file gives byte-identical output.
"""

from __future__ import annotations

import json

from routing import cli
from routing.export import load_annotations
from routing.review import append_adjudications
from routing.schemas import Adjudication, ReviewedRecord

from .helpers import make_fixture

WHEN = "2026-11-10T11:00:00+08:00"


def _shadow(tmp_path):
    fx = make_fixture(tmp_path / "fx")
    out = tmp_path / "run"
    assert cli.main(["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
                     "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1",
                     "--relevance-scores", str(fx["relevance"]), "--out", str(out)]) == 0
    return fx, out / "route_trace.jsonl"


def _findings(tmp_path, trace):
    path = tmp_path / "findings.jsonl"
    assert cli.main(["review", "findings", "--trace", str(trace), "--out", str(path)]) == 0
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _decide(finding, rejected_so_far):
    """A reviewer's answer: reject the first open candidate, accept the rest; draw a box for each
    suspected class; then mark each image complete."""
    base = {"record_id": finding["record_id"], "image_id": finding["image_id"], "reviewer": "ana",
            "decided_at": WHEN}
    if finding["finding"] == "candidate":
        decision = "reject" if not rejected_so_far else "accept"
        return Adjudication(**base, decision=decision, candidate_id=finding["candidate_id"])
    if finding["finding"] == "class":
        return Adjudication(**base, decision="add_box", class_id=finding["class_id"], box_xyxy=(5, 5, 45, 45))
    return Adjudication(**base, decision="complete")


def _apply(fx, trace, log, out):
    assert cli.main(["review", "apply", "--config", str(fx["config"]), "--trace", str(trace),
                     "--adjudications", str(log), "--out-dir", str(out)]) == 0
    lines = (out / "reviewed.jsonl").read_text(encoding="utf-8").splitlines()
    return {r.image.id: r for r in (ReviewedRecord.model_validate_json(line) for line in lines)}


def test_review_loop_end_to_end(tmp_path):
    fx, trace = _shadow(tmp_path)
    trace_before = trace.read_bytes()
    findings = _findings(tmp_path, trace)
    kinds = {f["finding"] for f in findings}
    assert {"candidate", "class", "image"} <= kinds  # the fixture exercises every kind of finding

    # 1. Every finding closed by name, then complete -> every reviewed image is complete.
    decisions, rejected = [], []
    for finding in findings:
        adj = _decide(finding, rejected)
        rejected += [adj.candidate_id] if adj.decision == "reject" else []
        decisions.append(adj)
    log = tmp_path / "full.jsonl"
    append_adjudications(log, decisions)
    reviewed = _apply(fx, trace, log, tmp_path / "full")
    open_images = {f["image_id"] for f in findings}
    assert all(reviewed[i].completeness.status == "complete" for i in open_images)

    # 2. Only bare `complete` decisions -> images with candidate or class findings stay open.
    bare = tmp_path / "bare.jsonl"
    append_adjudications(bare, [_decide(f, rejected) for f in findings if f["finding"] == "image"])
    bare_reviewed = _apply(fx, trace, bare, tmp_path / "bare")
    blocked = {f["image_id"] for f in findings if f["finding"] in ("candidate", "class")}
    assert blocked and all(bare_reviewed[i].completeness.status != "complete" for i in blocked)
    assert all("human_complete_overridden_by_open_findings" in bare_reviewed[i].completeness.reason_codes
               for i in blocked)

    # 3. The reviewed export: accepted model boxes plus human boxes, never the rejected one.
    doc = load_annotations(tmp_path / "full" / "annotations.json")  # the training-side reader accepts it
    exported = {a["vad_candidate_id"] for a in doc["annotations"]}
    assert rejected and not set(rejected) & exported
    sources = {a["vad_source"] for a in doc["annotations"]}
    assert sources == {"human", "model"} and all("score" not in a for a in doc["annotations"])
    human_boxes = sum(len(r.human_detections) for r in reviewed.values())
    assert sum(a["vad_source"] == "human" for a in doc["annotations"]) == human_boxes > 0

    # 4. Replay: the same adjudication file gives byte-identical output; the trace is untouched.
    _apply(fx, trace, log, tmp_path / "again")
    for name in ("reviewed.jsonl", "annotations.json"):
        assert (tmp_path / "full" / name).read_bytes() == (tmp_path / "again" / name).read_bytes()
    assert trace.read_bytes() == trace_before
