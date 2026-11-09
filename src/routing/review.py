"""Human review loop (reviewer work order A7; board T-7).

Stage 2: :func:`open_findings` lists what a reviewer must decide; :func:`apply_adjudications`
turns decisions into ReviewedRecords and re-audits each image (a finding closes only through an
adjudication naming it; a bare ``complete`` cannot close open findings).
Stage 1: the append-only adjudications log. Every human decision is an
:class:`~routing.schemas.Adjudication`, written as one JSON line. The file is only ever
appended to (never rewritten), so the history of decisions is replayable; reading verifies
every line (including its content-derived id) and drops exact repeats.
Model suggestions stay in the routed records; this log holds human decisions only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from .completeness import audit, failure_reasons_of
from .schemas import Adjudication, Detection, FinalCandidate, ReviewedRecord, RouteRecord, stable_id
from .task_config import RoutingConfig


class AdjudicationLogError(ValueError):
    """A line of the adjudications log is not a valid Adjudication."""


def append_adjudications(path: str | Path, adjudications: Iterable[Adjudication]) -> None:
    """Append decisions to the log, one JSON line each; earlier lines are never touched."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for adjudication in adjudications:
            f.write(adjudication.model_dump_json() + "\n")


def read_adjudications(path: str | Path) -> list[Adjudication]:
    """All decisions in log order; a missing log is empty; exact repeats (same id) are read once."""
    p = Path(path)
    if not p.exists():
        return []
    seen: set[str] = set()
    out: list[Adjudication] = []
    for number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            adjudication = Adjudication.model_validate_json(line)
        except ValidationError as e:
            raise AdjudicationLogError(f"{p}: line {number}: {e}") from None
        if adjudication.id not in seen:
            seen.add(adjudication.id)
            out.append(adjudication)
    return out


# --------------------------------------------------------------------------- apply and re-audit


class ReviewError(ValueError):
    """Adjudications that do not fit the routed records (unknown record, image, candidate or class)."""


def open_findings(record: RouteRecord) -> list[dict]:
    """What a reviewer must decide for one routed image, in order: candidates, classes, then the image.

    A candidate closes with accept / reject / reclassify naming it; a suspected class with
    class_absent / add_box naming it; the image (including failed escalations) with complete,
    after everything above it is decided.
    """
    if record.completeness.status == "complete":
        return []
    base = {"record_id": record.record_id, "image_id": record.image.id}
    out = [{**base, "finding": "candidate", "candidate_id": c.id, "class_id": c.class_id,
            "box_xyxy": list(c.box_xyxy), "status": c.status, "flags": list(c.flags),
            "decide_with": ["accept", "reject", "reclassify"]}
           for c in sorted(record.final_candidates, key=lambda c: c.id) if c.status in ("candidate", "needs_review")]
    out += [{**base, "finding": "class", "class_id": class_id, "decide_with": ["class_absent", "add_box"]}
            for class_id in record.completeness.suspected_missing_classes]
    out.append({**base, "finding": "image", "status": record.completeness.status,
                "failures": list(failure_reasons_of(record.completeness)),
                "reasons": list(record.completeness.reason_codes), "decide_with": ["complete"],
                "after": "every candidate and class finding of this image is decided"})
    return out


def _check_fits(adjudication: Adjudication, record: RouteRecord | None) -> None:
    if record is None:
        raise ReviewError(f"adjudication {adjudication.id} is for record {adjudication.record_id!r}, "
                          "which is not in the trace")
    if adjudication.image_id != record.image.id:
        raise ReviewError(f"adjudication {adjudication.id} names image {adjudication.image_id!r}, but record "
                          f"{record.record_id} is image {record.image.id!r}")
    if adjudication.candidate_id and adjudication.candidate_id not in {c.id for c in record.final_candidates}:
        raise ReviewError(f"adjudication {adjudication.id} names candidate {adjudication.candidate_id!r}, "
                          f"which record {record.record_id} does not have")
    if adjudication.class_id and adjudication.class_id not in record.task.class_ids:
        raise ReviewError(f"adjudication {adjudication.id} names class {adjudication.class_id!r}, "
                          "which is not in the task")


def _review_one(record: RouteRecord, adjudications: Sequence[Adjudication], cfg: RoutingConfig) -> ReviewedRecord:
    by_candidate = {a.candidate_id: a for a in adjudications if a.candidate_id}  # later decisions supersede
    added = [a for a in adjudications if a.decision == "add_box"]
    absent = {a.class_id: a for a in adjudications if a.decision == "class_absent"}
    if both := sorted(set(absent) & {a.class_id for a in added}):
        raise ReviewError(f"record {record.record_id}: class_absent and add_box both given for {both}")
    candidates = []
    for c in record.final_candidates:
        a = by_candidate.get(c.id)
        if a is None:
            candidates.append(c)
        elif a.decision == "reject":
            candidates.append(c.model_copy(update={"status": "rejected", "flags": c.flags + ("human_rejected",)}))
        elif a.decision == "accept":
            candidates.append(c.model_copy(update={"status": "accepted", "flags": c.flags + ("human_accepted",)}))
        else:  # reclassify
            candidates.append(c.model_copy(update={"status": "accepted", "class_id": a.class_id,
                                                   "flags": c.flags + (f"human_reclassified_from:{c.class_id}",)}))
    humans = []
    for a in added:
        box = Detection(id=stable_id("hum", a.id), class_id=a.class_id, box_xyxy=a.box_xyxy, source="human",
                        model_version=f"human:{a.reviewer}")
        try:
            box.check_in(record.image, record.task.class_ids)
        except ValueError as e:
            raise ReviewError(f"adjudication {a.id}: {e}") from None
        humans.append(box)
        candidates.append(FinalCandidate(id=stable_id("cand", [box.id]), class_id=box.class_id, box_xyxy=box.box_xyxy,
                                         status="accepted", source_ids=(box.id,), canonical_source_id=box.id,
                                         flags=("human_added",)))
    # R5-3: an accepted box (accepted by the policy, by a human or by reclassification) says its class
    # is present, so class_absent for that class is a contradiction; the box is never dropped silently.
    if clashes := sorted(f"{c.id} ({c.class_id})" for c in candidates if c.status == "accepted" and c.class_id in absent):
        raise ReviewError(f"record {record.record_id}: class_absent contradicts accepted candidate(s) {clashes}; "
                          "reject or reclassify the candidate, or do not declare its class absent")
    effective = [*by_candidate.values(), *added, *absent.values(),
                 *(a for a in adjudications if a.decision == "complete")]
    completeness = audit(candidates, record.stage_a, record.route, [v for v in record.verifications if not v.candidate_ids],
                         cfg, failure_reasons=failure_reasons_of(record.completeness), adjudications=effective)
    return ReviewedRecord(record_id=record.record_id, image=record.image, task=record.task,
                          adjudications=tuple(adjudications), human_detections=tuple(humans),
                          final_candidates=tuple(sorted(candidates, key=lambda c: c.id)), completeness=completeness)


def apply_adjudications(records: Sequence[RouteRecord], adjudications: Sequence[Adjudication],
                        cfg: RoutingConfig) -> list[ReviewedRecord]:
    """Apply the decisions (in log order; a later decision on a finding supersedes an earlier one) and re-audit.

    Records are never modified: model suggestions stay in the RouteRecords, and each result is
    a separate ReviewedRecord. With no adjudications the re-audit reproduces the original one.
    Contradictions are refused (ReviewError): class_absent together with add_box for that class, or
    with any accepted candidate of that class (policy-accepted, accepted or reclassified into it).
    """
    by_id = {r.record_id: r for r in records}
    for r in records:
        if r.run.config_hash != cfg.config_hash():
            raise ReviewError(f"record {r.record_id} was routed with config {r.run.config_hash}, but the given "
                              f"config hashes to {cfg.config_hash()}; the re-audit needs the run's config")
    grouped: dict[str, list[Adjudication]] = {r.record_id: [] for r in records}
    for a in adjudications:
        _check_fits(a, by_id.get(a.record_id))
        grouped[a.record_id].append(a)
    return [_review_one(r, grouped[r.record_id], cfg) for r in records]
