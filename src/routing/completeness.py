"""Annotation completeness audit.

Completeness is a separate question from difficulty: is there evidence of a
target instance without an accepted annotation? "All model boxes accepted" is
never enough to claim ``complete``; only a human or a calibrated audit policy
can establish it.
"""

from __future__ import annotations

from collections.abc import Sequence

from .matching import iou
from .schemas import Adjudication, AnnotationAudit, FinalCandidate, RouteDecision, StageA, VerificationResult
from .scoring import calibrated_probability_missing, class_is_high_relevance
from .task_config import RoutingConfig

# Reason codes the pipeline passes in as failure_reasons (failed or partial escalation). A re-audit
# recovers them from a stored record's reason codes.
FAILURE_PREFIXES = ("elapsed_budget_exhausted_before:", "gpu_budget_exhausted_before:", "secondary_failed:",
                    "sahi_partial:", "sahi_failed:", "vlm_call_cap_reached", "vlm_skipped_budget")


def failure_reasons_of(completeness: AnnotationAudit) -> tuple[str, ...]:
    return tuple(r for r in completeness.reason_codes if r.startswith(FAILURE_PREFIXES))


def audit(
    candidates: Sequence[FinalCandidate],
    sa: StageA,
    decision: RouteDecision,
    missing_class_checks: Sequence[VerificationResult],
    cfg: RoutingConfig,
    *,
    failure_reasons: Sequence[str] = (),
    reference_boxes: Sequence[tuple[str, tuple[float, float, float, float]]] = (),
    human_outcome: str | None = None,
    adjudications: Sequence[Adjudication] | None = None,
) -> AnnotationAudit:
    """Audit one image.

    ``reference_boxes`` are accepted annotations from another source (e.g. an
    existing human label set); proposals not covered by them are unmatched.
    ``human_outcome`` ("complete" or free text) records a reviewer decision.
    ``adjudications`` (human review) close findings one by one: a decided candidate no longer
    counts as unresolved through its verification, ``class_absent`` closes that class's suspected
    or calibrator-missing evidence, and ``complete`` establishes human authority and closes failed
    escalations only once no other finding is open. Without them the audit is unchanged.
    """
    adjudications = adjudications or ()
    decided = {a.candidate_id for a in adjudications if a.candidate_id}
    absent = {a.class_id for a in adjudications if a.decision == "class_absent"}
    human_complete = any(a.decision == "complete" for a in adjudications)
    accepted = [c for c in candidates if c.status == "accepted"]
    reasons: list[str] = []

    unmatched = [c.id for c in candidates if c.status in ("needs_review", "candidate")]
    if reference_boxes:
        for c in accepted:
            if not any(cls == c.class_id and iou(box, c.box_xyxy) >= cfg.matching.iou_threshold
                       for cls, box in reference_boxes):
                unmatched.append(c.id)
        reasons.append("compared_against_reference_annotations")
    if unmatched:
        reasons.append("unmatched_candidates")

    accepted_classes = {c.class_id for c in accepted} | {cls for cls, _ in reference_boxes}
    suspected = sorted(
        cid for cid in cfg.task.class_ids
        if cid not in accepted_classes and class_is_high_relevance(sa, cid, cfg)
    )
    reasons.extend(f"human_confirmed_absent:{cid}" for cid in suspected if cid in absent)
    suspected = [cid for cid in suspected if cid not in absent]
    unresolved = False
    checks = {v.class_id: v for v in missing_class_checks}
    for cid in suspected:
        reasons.append(f"high_relevance_zero_boxes_for_class:{cid}")
        v = checks.get(cid)
        if v is None or v.verdict == "uncertain":
            unresolved = True
        elif v.verdict == "present":
            reasons.append(f"vlm_reports_unannotated_instance:{cid}")
            unresolved = True
        else:
            reasons.append(f"vlm_absent_for_suspected_class:{cid}")
    for cid in cfg.task.class_ids:
        if cid not in accepted_classes and cid not in absent and calibrated_probability_missing(sa, cid):
            reasons.append(f"calibrator_missing_class:{cid}")
            suspected.append(cid)
            unresolved = True  # relevance unknown: never a negative, needs a human
    if any(c.verification is not None and c.verification.verdict == "uncertain" and c.id not in decided
           for c in candidates):
        reasons.append("verification_uncertain")
        unresolved = True

    if "audit_sample" in decision.actions:
        reasons.append("audit_sampled")
    if "human_review" in decision.actions:
        reasons.append("routed_to_human_review")
    reasons.extend(failure_reasons)
    if "low_relevance_zero_boxes" in decision.reason_codes:
        reasons.append("negative_not_verified")

    if human_complete and human_outcome is None:
        human_outcome = "complete"
    if human_outcome is not None:
        established = "human"
    elif cfg.audit.calibrated:
        established = "calibrated_audit_policy"
    else:
        established = "none"

    # A bare human "complete" may only close an image with no open evidence. It cannot clear
    # unmatched candidates, unresolved/uncertain verifications or failed escalations: those
    # need explicit per-finding adjudications; failed escalations close only with a complete
    # adjudication once nothing else is open.
    failures_open = bool(failure_reasons) and not (human_complete and not unresolved and not unmatched)
    if failure_reasons and not failures_open:
        reasons.append("escalation_failures_closed_by_human_complete")
    open_findings = unresolved or bool(unmatched) or failures_open
    if human_outcome == "complete" and open_findings:
        reasons.append("human_complete_overridden_by_open_findings")
    if human_outcome == "complete" and not open_findings:
        status = "complete"
    elif unresolved:
        status = "unresolved"
    elif unmatched or failure_reasons or established == "none" or "human_review" in decision.actions:
        status = "needs_review"
        if established == "none":
            reasons.append("completeness_not_established")
    else:
        status = "complete"

    return AnnotationAudit(
        status=status, unmatched_candidate_ids=tuple(sorted(set(unmatched))),
        suspected_missing_classes=tuple(sorted(suspected)), reason_codes=tuple(dict.fromkeys(reasons)),
        established_by=established, reviewer_outcome=human_outcome,
    )
