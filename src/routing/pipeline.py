"""Bounded, resumable orchestration of the routing graph.

Order per image: validate -> Stage A -> primary -> Stage B -> route (persisted)
-> optional secondary / SAHI -> match + fuse -> bounded VLM -> completeness ->
persisted RouteRecord. Every expensive step result is cached under the record
key once it returns, so a restarted job reuses it. A crash after a detector call but
before its result is persisted can repeat that call (sliced inference is cached only
after the whole sliced pass). VLM requests are at-most-once: an attempt marker is
persisted before sending, and an interrupted request is recorded as uncertain, never re-sent.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from PIL import Image

from .completeness import audit
from .detection_mapping import to_detections
from .image_inputs import ManifestEntry, load_image
from .interfaces import Detector, RelevanceCalibrator, RelevanceScorer, VLMVerifier
from .matching import disagreement, final_candidate, fuse, match
from .policy import Capabilities, route
from .schemas import (
    Budget, Detection, Disagreement, ErrorRecord, FinalCandidate, ImageRef, RouteDecision, RouteRecord,
    RunInfo, Source, StageA, StageB, Trace, VerificationRequest, VerificationResult, stable_id,
)
from .scoring import box_features, class_is_high_relevance, estimate_difficulty, stage_a
from .slicing import run_sliced
from .store import RouteStore
from .task_config import RoutingConfig
from .verification_results import uncertain


# --------------------------------------------------------------------------- pipeline


def _dets(payload) -> list[Detection]:
    return [Detection.model_validate(d) for d in payload]


def _dump(dets: Sequence[Detection]) -> list[dict]:
    return [d.model_dump(mode="json") for d in dets]


class RoutingPipeline:
    def __init__(
        self,
        cfg: RoutingConfig,
        primary: Detector,
        store: RouteStore,
        *,
        secondary: Detector | None = None,
        sahi_detector: Detector | None = None,
        relevance: RelevanceScorer | None = None,
        calibrator: RelevanceCalibrator | None = None,
        vlm: VLMVerifier | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        scale = getattr(relevance, "score_scale", None)
        if scale is not None and cfg.relevance.raw_scale != scale:
            raise ValueError(f"relevance scorer reports scale {scale!r} but config relevance.raw_scale is "
                             f"{cfg.relevance.raw_scale!r}; declare raw bands for this scale before using it")
        self.cfg, self.primary, self.store = cfg, primary, store
        self._step = "idle"
        self.secondary, self.relevance, self.calibrator, self.clock = secondary, relevance, calibrator, clock
        crop_capable = sahi_detector or (primary if primary.supports_crops else None)
        self.sahi_detector = crop_capable if crop_capable is not None and crop_capable.supports_crops else None
        self.vlm = vlm if cfg.vlm.enabled else None
        self.caps = Capabilities(secondary=secondary is not None, sahi=self.sahi_detector is not None,
                                 vlm=self.vlm is not None)
        self.run_info = RunInfo(
            config_hash=cfg.config_hash(), policy_version=cfg.policy_version,
            primary_model=primary.model_version,
            secondary_model=secondary.model_version if secondary else None,
            relevance_scorer=relevance.model_version if relevance else None,
            relevance_calibrator=calibrator.version if calibrator else None,
            vlm_model=self.vlm.model_version if self.vlm else None,
            primary_artifact_digest=getattr(primary, "artifact_digest", None),
            secondary_artifact_digest=getattr(secondary, "artifact_digest", None),
            relevance_artifact_digest=getattr(relevance, "artifact_digest", None),
        )

    # -- legacy path (feature flag off) -------------------------------------------------

    def legacy_predict(self, image: Image.Image, ref: ImageRef) -> list[Detection]:
        res = self.primary.detect(image, ref.id, self.cfg.task.class_ids)
        dets, _ = to_detections(res.detections, ref, self.cfg.task.class_ids, "primary", self.primary.model_version)
        return dets

    # -- routed path ----------------------------------------------------------------------

    def record_key(self, ref: ImageRef) -> str:
        return stable_id("rec", [ref.sha256, ref.id, self.run_info.model_dump(mode="json")])

    def run(self, entries: Iterable[ManifestEntry]) -> list[RouteRecord | ErrorRecord]:
        return [self.process(e) for e in entries]

    def process(self, entry: ManifestEntry) -> RouteRecord | ErrorRecord:
        if not self.cfg.routing_enabled:
            raise RuntimeError("routing_enabled is false; use legacy_predict() for the existing path")
        try:
            image, ref = load_image(entry)
        except Exception as e:  # corrupt/unreadable -> explicit error queue, batch continues
            try:
                content = hashlib.sha256(Path(entry.uri).read_bytes()).hexdigest()
            except OSError:
                content = None
            ident = [entry.image_id or entry.uri, content, self.run_info.config_hash]
            err = ErrorRecord(record_id=stable_id("err", ident),
                              uri=entry.uri, error=f"{type(e).__name__}: {e}"[:500],
                              config_hash=self.run_info.config_hash)
            self.store.put_error(err)
            return err
        key = self.record_key(ref)
        done = self.store.get_record(key)
        if done is not None:
            return done
        self._step = "start"
        try:
            return self._route_image(image, ref, key)
        except Exception as e:  # model/adapter failure: queue the image, keep the batch going
            err = ErrorRecord(record_id=stable_id("err", [key, self._step]), uri=entry.uri,
                              error=f"routing_failed_at:{self._step}: {type(e).__name__}: {e}"[:500],
                              config_hash=self.run_info.config_hash)
            self.store.put_error(err)
            return err

    def _load_or_compute_stage_a(self, image: Image.Image, ref: ImageRef, key: str) -> StageA:
        cached = self.store.get_step(key, "stage_a")
        if cached is not None:
            return StageA.model_validate(cached)
        scores = stage_a(image, ref, self.cfg, self.relevance, self.calibrator)
        self.store.put_step(key, "stage_a", scores.model_dump(mode="json"))
        return scores

    def _run_cached_detector(
        self, key: str, step: str, detector: Detector, image: Image.Image, ref: ImageRef, source: Source,
        class_mapping: Mapping[str, str] | None = None,
    ) -> tuple[list[Detection], float | None, list[str]]:
        """Detections for ``step``, from the step cache or one detector call (then cached). Errors propagate."""
        cached = self.store.get_step(key, step)
        if cached is None:
            class_ids = self.cfg.task.class_ids
            result = detector.detect(image, ref.id, class_ids)
            detections, mapping_warnings = to_detections(result.detections, ref, class_ids, source,
                                                         detector.model_version, class_mapping)
            cached = {"detections": _dump(detections), "gpu_ms": result.gpu_ms, "warnings": mapping_warnings}
            self.store.put_step(key, step, cached)
        return _dets(cached["detections"]), cached["gpu_ms"], list(cached["warnings"])

    def _run_sliced_detection(self, key: str, image: Image.Image, ref: ImageRef, elapsed_ms: Callable[[], float],
                              gpu_before: float | None) -> dict:
        """The ``sahi`` step payload, from the step cache or one sliced run (then cached). Errors propagate.

        Tiling stops early when the image's elapsed budget is spent, or when the GPU time
        measured before slicing plus the tiles' measured GPU time reaches the GPU budget.
        """
        cached = self.store.get_step(key, "sahi")
        if cached is not None:
            return cached
        budgets = self.cfg.budgets

        def stop_check(sahi_gpu: float | None) -> str | None:
            if elapsed_ms() >= budgets.max_elapsed_ms:
                return "elapsed_budget_exhausted"
            if gpu_before is not None and sahi_gpu is not None and gpu_before + sahi_gpu >= budgets.max_gpu_ms:
                return "gpu_budget_exhausted"
            return None

        sliced = run_sliced(self.sahi_detector, image, ref, self.cfg.task.class_ids, self.cfg.slicing,
                            self.cfg.matching.class_mapping, max_slices=budgets.max_slices, stop_check=stop_check)
        cached = {"detections": _dump(sliced.detections), "raw": _dump(sliced.raw), "gpu_ms": sliced.gpu_ms,
                  "warnings": sliced.warnings, "slices_run": sliced.slices_run, "stopped_reason": sliced.stopped_reason}
        self.store.put_step(key, "sahi", cached)
        return cached

    def _route_image(self, image: Image.Image, ref: ImageRef, key: str) -> RouteRecord:
        cfg = self.cfg
        start = self.clock()
        events: list[str] = []
        warnings: list[str] = []
        failures: list[str] = []
        gpu_total: float | None = 0.0
        slices_used = 0

        def add_gpu(ms: float | None) -> None:
            nonlocal gpu_total
            if ms is None or gpu_total is None:
                gpu_total = None  # one unmeasured call makes the image's GPU total unknown
            else:
                gpu_total += ms

        def elapsed_ms() -> float:
            return (self.clock() - start) * 1000.0

        def within_budget(step: str) -> bool:
            if elapsed_ms() >= cfg.budgets.max_elapsed_ms:
                failures.append(f"elapsed_budget_exhausted_before:{step}")
                return False
            if gpu_total is not None and gpu_total >= cfg.budgets.max_gpu_ms:
                failures.append(f"gpu_budget_exhausted_before:{step}")
                return False
            return True

        self._step = "stage_a"
        stage_a_scores = self._load_or_compute_stage_a(image, ref, key)

        self._step = "primary"
        # A primary failure propagates: process() queues the image as an ErrorRecord.
        primary, gpu_ms, detector_warnings = self._run_cached_detector(key, "primary", self.primary, image, ref,
                                                                       "primary")
        add_gpu(gpu_ms)
        warnings += detector_warnings
        events.append(f"primary:{len(primary)}")

        self._step = "stage_b"
        features = box_features(primary, ref, stage_a_scores, cfg)
        unavailable = Disagreement(status="unavailable")
        difficulty_pre = estimate_difficulty(stage_a_scores, features, unavailable, cfg)

        stored = self.store.get_step(key, "route")
        if stored is None:
            decision = route(ref.id, ref.width, ref.height, stage_a_scores, features, difficulty_pre, self.caps, cfg)
            self.store.put_step(key, "route", decision.model_dump(mode="json"))  # persist before expensive calls
        else:
            decision = RouteDecision.model_validate(stored)
            events.append("route:resumed")

        self._step = "escalation"
        secondary: list[Detection] = []
        sahi: list[Detection] = []
        sahi_raw: list[Detection] = []
        ran_secondary = False
        ran_sahi = False
        if "secondary" in decision.actions and self.secondary is not None and within_budget("secondary"):
            try:
                secondary, gpu_ms, detector_warnings = self._run_cached_detector(
                    key, "secondary", self.secondary, image, ref, "secondary", cfg.matching.class_mapping)
                add_gpu(gpu_ms)
                warnings += detector_warnings
                ran_secondary = True
                events.append(f"secondary:{len(secondary)}")
            except Exception as e:  # an optional detector failing is a review reason, not an error
                failures.append(f"secondary_failed:{type(e).__name__}")

        if "sahi" in decision.actions and self.sahi_detector is not None and within_budget("sahi"):
            try:
                sliced = self._run_sliced_detection(key, image, ref, elapsed_ms, gpu_total)
                sahi, sahi_raw = _dets(sliced["detections"]), _dets(sliced["raw"])
                add_gpu(sliced["gpu_ms"])
                slices_used = sliced["slices_run"]
                warnings += sliced["warnings"]
                if sliced["stopped_reason"]:  # budget hit mid-way: a partial pass is never claimed complete
                    failures.append(f"sahi_partial:{sliced['stopped_reason']}")
                ran_sahi = True
                events.append(f"sahi:{len(sahi)}/{slices_used}_slices")
            except Exception as e:  # optional slicing failing is a review reason, not an error
                failures.append(f"sahi_failed:{type(e).__name__}")

        self._step = "fusion"
        # Stage B after secondary: class-aware matching and directional disagreement.
        detector_disagreement = unavailable
        mislocalized_ids: set[str] = set()
        difficulty_final = difficulty_pre
        if ran_secondary:
            matched = match(primary, secondary, cfg.matching.iou_threshold)
            detector_disagreement = disagreement(matched, cfg.matching.mislocalized_iou)
            mislocalized_ids = {det.id for primary_det, secondary_det, pair_iou in matched.pairs
                                if pair_iou < cfg.matching.mislocalized_iou for det in (primary_det, secondary_det)}
            difficulty_final = estimate_difficulty(stage_a_scores, features, detector_disagreement, cfg)

        candidates = self._candidates(primary, secondary, sahi, decision, ran_secondary or ran_sahi, mislocalized_ids)

        self._step = "vlm_verify"
        verifications: list[VerificationResult] = []
        missing_checks: list[VerificationResult] = []
        vlm_used = 0
        if "vlm_verify" in decision.actions and self.vlm is not None:
            candidates, verifications, missing_checks, vlm_used = self._verify(
                image, ref, key, stage_a_scores, candidates, within_budget, failures)

        self._step = "record"
        completeness = audit(candidates, stage_a_scores, decision, missing_checks, cfg, failure_reasons=failures)
        budget_used = Budget(slices=slices_used, vlm_calls=vlm_used, elapsed_ms=round(elapsed_ms(), 2),
                             gpu_ms=gpu_total)
        record = RouteRecord(
            record_id=key, image=ref, task=cfg.task, run=self.run_info, stage_a=stage_a_scores,
            primary_candidates=tuple(primary), secondary_candidates=tuple(secondary), sahi_candidates=tuple(sahi),
            sahi_raw_candidates=tuple(sahi_raw),
            stage_b=StageB(difficulty_pre_secondary=difficulty_pre, difficulty_final=difficulty_final,
                           features=features, disagreement=detector_disagreement),
            route=decision, verifications=tuple(verifications + missing_checks),
            final_candidates=tuple(candidates), completeness=completeness,
            trace=Trace(elapsed_ms=budget_used.elapsed_ms, gpu_ms=gpu_total, budget_used=budget_used,
                        events=tuple(events), warnings=tuple(dict.fromkeys(warnings + failures))),
        )
        self.store.put_record(record)
        return record

    def _candidates(
        self, primary: Sequence[Detection], secondary: Sequence[Detection], sahi: Sequence[Detection],
        decision: RouteDecision, ran_extra: bool, mislocalized_ids: set[str],
    ) -> list[FinalCandidate]:
        out = []
        reasons = decision.reason_codes
        low_relevance = "low_relevance_with_detections" in reasons
        risky_unconfirmed = "high_miss_risk" in reasons and not ran_extra
        # The route wanted a second detector (moderate route, or secondary planned but skipped/failed)
        # and none ran: primary boxes are not confirmed by anything, so they must not be auto-accepted.
        moderate_unconfirmed = (not ran_extra and not risky_unconfirmed
                                and ("secondary" in decision.actions or "secondary_unavailable" in reasons))
        for cluster in fuse([*primary, *secondary, *sahi], self.cfg.matching.iou_threshold):
            flags: list[str] = []
            has_primary = "primary" in cluster.sources
            if risky_unconfirmed:
                flags.append("high_miss_risk_unconfirmed")
            if moderate_unconfirmed:
                flags.append("secondary_unconfirmed")
            if cluster.cross_class_conflict:
                flags.append("cross_class_conflict")
            if any(det.id in mislocalized_ids for det in cluster.members):
                flags.append("mislocalized")
            if has_primary and ran_extra and cluster.sources == {"primary"}:
                flags.append("primary_only_disagreement")
            if not has_primary:
                flags.append("detector_proposal:" + "+".join(sorted(cluster.sources)))
            if low_relevance:
                flags.append("low_relevance_image")
            if not has_primary:
                status = "candidate"
            elif flags:
                status = "needs_review"
            else:
                status = "accepted"
            out.append(final_candidate(cluster, status, flags))
        return sorted(out, key=lambda candidate: candidate.id)

    def _verify(
        self, image: Image.Image, ref: ImageRef, key: str, stage_a_scores: StageA, candidates: list[FinalCandidate],
        within_budget: Callable[[str], bool], failures: list[str],
    ) -> tuple[list[FinalCandidate], list[VerificationResult], list[VerificationResult], int]:
        """Bounded VLM pass: reviewable candidates first, then high-relevance classes without an accepted box.

        Every request counts toward the per-image cap, including answers reused from the
        step cache on resume, so the logical cap holds across restarts.
        """
        cfg = self.cfg
        descriptions = {spec.class_id: spec.positive_prompts[0] for spec in cfg.task.classes}
        cap = cfg.budgets.max_vlm_calls
        calls = 0
        verifications: list[VerificationResult] = []

        updated = []
        for candidate in candidates:
            if candidate.status not in ("needs_review", "candidate"):
                updated.append(candidate)
                continue
            if calls >= cap or not within_budget("vlm_verify"):
                failures.append("vlm_call_cap_reached" if calls >= cap else "vlm_skipped_budget")
                updated.append(candidate)
                continue
            request = VerificationRequest(id=stable_id("vreq", [key, "confirm", candidate.id]), image_id=ref.id,
                                          class_id=candidate.class_id, candidate_ids=(candidate.id,),
                                          region_xyxy=candidate.box_xyxy, purpose="confirm_candidate")
            calls += 1
            result = self._verify_request(image, key, request, descriptions[request.class_id])
            verifications.append(result)
            flags = candidate.flags
            if result.verdict == "absent" and not cfg.vlm.absent_rejects:
                flags = flags + ("vlm_absent",)
            status = self._candidate_status_after_verification(candidate, result)
            updated.append(candidate.model_copy(update={"status": status, "verification": result, "flags": flags}))

        accepted_classes = {candidate.class_id for candidate in updated if candidate.status == "accepted"}
        missing: list[VerificationResult] = []
        for class_id in cfg.task.class_ids:
            if class_id in accepted_classes or not class_is_high_relevance(stage_a_scores, class_id, cfg):
                continue
            if calls >= cap or not within_budget("vlm_missing_class"):
                failures.append("vlm_call_cap_reached" if calls >= cap else "vlm_skipped_budget")
                continue
            request = VerificationRequest(id=stable_id("vreq", [key, "missing", class_id]), image_id=ref.id,
                                          class_id=class_id, purpose="suspected_missing_class")
            calls += 1
            missing.append(self._verify_request(image, key, request, descriptions[request.class_id]))
        return sorted(updated, key=lambda candidate: candidate.id), verifications, missing, calls

    def _verify_request(self, image: Image.Image, key: str, request: VerificationRequest,
                        class_description: str) -> VerificationResult:
        """One VLM request with at-most-once delivery; the result is persisted under ``vlm:<id>``."""
        vlm = self.vlm
        cached = self.store.get_step(key, f"vlm:{request.id}")
        if cached is not None:
            return VerificationResult.model_validate(cached)
        # At-most-once: the attempt is persisted *before* the external call. If a previous
        # process died between sending and saving the answer, the request is not re-sent
        # (the VLM has no idempotency key); the lost answer becomes `uncertain` -> review.
        if self.store.get_step(key, f"vlm_attempt:{request.id}") is not None:
            result = uncertain(request, "vlm_interrupted", vlm.model_version,
                               "call started before an interruption; answer was not persisted")
        else:
            self.store.put_step(key, f"vlm_attempt:{request.id}", {"request_id": request.id})
            try:
                result = vlm.verify(image, request, class_description)
            except Exception as e:  # adapters should not raise, but never trust that
                result = uncertain(request, "vlm_unavailable", vlm.model_version, type(e).__name__)
            if not self._verification_matches_request(result, request):
                result = uncertain(request, "vlm_contradictory", vlm.model_version)
        self.store.put_step(key, f"vlm:{request.id}", result.model_dump(mode="json"))
        return result

    @staticmethod
    def _verification_matches_request(res: object, req: VerificationRequest) -> bool:
        """True only if ``res`` is a verification of exactly this request (class, candidates, region)."""
        return (isinstance(res, VerificationResult) and res.request_id == req.id and res.class_id == req.class_id
                and tuple(res.candidate_ids) == tuple(req.candidate_ids) and res.region_xyxy == req.region_xyxy)

    def _candidate_status_after_verification(self, c: FinalCandidate, res: VerificationResult) -> str:
        """VLM confirms or rejects detector-authored boxes; it never creates or moves one.

        A cross-class conflict is checked first: the VLM answers per class and cannot
        adjudicate which class (or geometry) is right, so it may neither accept nor
        reject a conflicted candidate while ``cross_class_always_review`` is on.
        """
        if "cross_class_conflict" in c.flags and self.cfg.policy.cross_class_always_review:
            return "needs_review"
        if res.verdict == "absent":
            return "rejected" if self.cfg.vlm.absent_rejects else "needs_review"
        if res.verdict == "uncertain":
            return "needs_review"
        proposal = any(f.startswith("detector_proposal:") for f in c.flags)
        if not proposal or self.cfg.vlm.accept_confirmed_detector_proposals:
            return "accepted"
        return "needs_review"
