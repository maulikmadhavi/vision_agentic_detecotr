"""Bounded, resumable orchestration of the routing graph.

Order per image: validate -> Stage A -> primary -> Stage B -> route (persisted)
-> optional secondary / SAHI -> match + fuse -> bounded VLM -> completeness ->
persisted RouteRecord. Every expensive step result is cached under the record
key, so a restarted job never repeats a detector or VLM call.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .adapters import Detector, RelevanceCalibrator, RelevanceScorer, VLMVerifier, to_detections, uncertain
from .completeness import audit
from .matching import disagreement, final_candidate, fuse, match
from .policy import Capabilities, route
from .schemas import (
    Budget, Detection, Disagreement, ErrorRecord, FinalCandidate, ImageRef, RouteDecision, RouteRecord,
    RunInfo, StageA, StageB, Trace, VerificationRequest, VerificationResult, stable_id,
)
from .scoring import box_features, class_is_high_relevance, estimate_difficulty, stage_a
from .slicing import run_sliced
from .task_config import RoutingConfig


# --------------------------------------------------------------------------- manifest


@dataclass(frozen=True)
class ManifestEntry:
    uri: str
    image_id: str | None = None
    group_id: str | None = None


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """A directory of images, or a JSONL manifest of {"uri", "image_id"?, "group_id"?}."""
    p = Path(path)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return [ManifestEntry(str(f)) for f in sorted(p.iterdir()) if f.suffix.lower() in exts]
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    base = p.parent
    return [ManifestEntry(str((base / r["uri"]) if not Path(r["uri"]).is_absolute() else r["uri"]),
                          r.get("image_id"), r.get("group_id")) for r in rows]


def load_image(entry: ManifestEntry) -> tuple[Image.Image, ImageRef]:
    data = Path(entry.uri).read_bytes()
    with Image.open(io.BytesIO(data)) as probe:
        probe.verify()
    img = Image.open(io.BytesIO(data))
    img.load()
    img = img.convert("RGB")
    sha = hashlib.sha256(data).hexdigest()
    ref = ImageRef(id=entry.image_id or f"sha256-{sha[:16]}", uri=entry.uri, width=img.width,
                   height=img.height, sha256=sha, group_id=entry.group_id)
    return img, ref


# --------------------------------------------------------------------------- persistence


class StoreLockedError(RuntimeError):
    """Another routing job holds the output directory."""


@contextmanager
def job_lock(out_dir: str | Path):
    """Exclusive per-output-directory lock for a writing job (shadow run).

    Uses an OS byte-range lock (``msvcrt`` on Windows, ``flock`` elsewhere), so the
    lock disappears with the process: no stale-lock cleanup and no PID probing.
    Readers (reports, tests reopening a store) do not need it.
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    f = open(p / ".vad-route.lock", "a+", encoding="utf-8")
    try:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise StoreLockedError(f"{p} is in use by another routing job; run one job per output directory") from e
        try:
            yield p
        finally:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


class RouteStore:
    """SQLite store of step results, final records and the error queue."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS steps (record_key TEXT, step TEXT, payload TEXT, PRIMARY KEY (record_key, step));
            CREATE TABLE IF NOT EXISTS records (record_key TEXT PRIMARY KEY, image_id TEXT, status TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS errors (record_key TEXT PRIMARY KEY, payload TEXT);
        """)

    def get_step(self, key: str, step: str):
        row = self.db.execute("SELECT payload FROM steps WHERE record_key=? AND step=?", (key, step)).fetchone()
        return json.loads(row[0]) if row else None

    def put_step(self, key: str, step: str, payload) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO steps VALUES (?,?,?)", (key, step, json.dumps(payload, sort_keys=True)))

    def get_record(self, key: str) -> RouteRecord | None:
        row = self.db.execute("SELECT payload FROM records WHERE record_key=?", (key,)).fetchone()
        return RouteRecord.model_validate_json(row[0]) if row else None

    def put_record(self, rec: RouteRecord) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
                            (rec.record_id, rec.image.id, rec.completeness.status, rec.model_dump_json()))

    def put_error(self, err: ErrorRecord) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO errors VALUES (?,?)", (err.record_id, err.model_dump_json()))

    def records(self) -> list[RouteRecord]:
        return [RouteRecord.model_validate_json(r[0]) for r in self.db.execute("SELECT payload FROM records ORDER BY image_id")]

    def errors(self) -> list[ErrorRecord]:
        return [ErrorRecord.model_validate_json(r[0]) for r in self.db.execute("SELECT payload FROM errors ORDER BY record_key")]


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

    def _cached_detect(self, key: str, step: str, fn) -> tuple[list[Detection], float | None, list[str]]:
        cached = self.store.get_step(key, step)
        if cached is None:
            dets, gpu, warns = fn()
            cached = {"detections": _dump(dets), "gpu_ms": gpu, "warnings": warns}
            self.store.put_step(key, step, cached)
        return _dets(cached["detections"]), cached["gpu_ms"], list(cached["warnings"])

    def _route_image(self, image: Image.Image, ref: ImageRef, key: str) -> RouteRecord:
        cfg, t0 = self.cfg, self.clock()
        class_ids = cfg.task.class_ids
        events: list[str] = []
        warnings: list[str] = []
        failures: list[str] = []
        gpu_total: float | None = 0.0
        slices_used = 0

        def add_gpu(v: float | None) -> None:
            nonlocal gpu_total
            gpu_total = None if (v is None or gpu_total is None) else gpu_total + v

        def elapsed_ms() -> float:
            return (self.clock() - t0) * 1000.0

        def within_budget(step: str) -> bool:
            if elapsed_ms() >= cfg.budgets.max_elapsed_ms:
                failures.append(f"elapsed_budget_exhausted_before:{step}")
                return False
            if gpu_total is not None and gpu_total >= cfg.budgets.max_gpu_ms:
                failures.append(f"gpu_budget_exhausted_before:{step}")
                return False
            return True

        self._step = "stage_a"
        cached_sa = self.store.get_step(key, "stage_a")
        if cached_sa is None:
            sa = stage_a(image, ref, cfg, self.relevance, self.calibrator)
            self.store.put_step(key, "stage_a", sa.model_dump(mode="json"))
        else:
            sa = StageA.model_validate(cached_sa)

        def run_primary():
            res = self.primary.detect(image, ref.id, class_ids)
            dets, w = to_detections(res.detections, ref, class_ids, "primary", self.primary.model_version)
            return dets, res.gpu_ms, w

        self._step = "primary"
        primary, g, w = self._cached_detect(key, "primary", run_primary)
        add_gpu(g)
        warnings += w
        events.append(f"primary:{len(primary)}")

        self._step = "stage_b"
        features = box_features(primary, ref, sa, cfg)
        unavailable = Disagreement(status="unavailable")
        diff_pre = estimate_difficulty(sa, features, unavailable, cfg)

        stored = self.store.get_step(key, "route")
        if stored is None:
            decision = route(ref.id, ref.width, ref.height, sa, features, diff_pre, self.caps, cfg)
            self.store.put_step(key, "route", decision.model_dump(mode="json"))  # persist before expensive calls
        else:
            decision = RouteDecision.model_validate(stored)
            events.append("route:resumed")

        self._step = "escalation"
        secondary: list[Detection] = []
        sahi: list[Detection] = []
        sahi_raw: list[Detection] = []
        ran_secondary = ran_sahi = False
        if "secondary" in decision.actions and self.secondary is not None and within_budget("secondary"):
            sec = self.secondary

            def run_secondary():
                res = sec.detect(image, ref.id, class_ids)
                dets, w2 = to_detections(res.detections, ref, class_ids, "secondary", sec.model_version,
                                         cfg.matching.class_mapping)
                return dets, res.gpu_ms, w2

            try:
                secondary, g, w = self._cached_detect(key, "secondary", run_secondary)
                add_gpu(g)
                warnings += w
                ran_secondary = True
                events.append(f"secondary:{len(secondary)}")
            except Exception as e:
                failures.append(f"secondary_failed:{type(e).__name__}")

        if "sahi" in decision.actions and self.sahi_detector is not None and within_budget("sahi"):
            det = self.sahi_detector

            gpu_before = gpu_total

            def stop_check(sahi_gpu: float | None) -> str | None:
                if elapsed_ms() >= cfg.budgets.max_elapsed_ms:
                    return "elapsed_budget_exhausted"
                if gpu_before is not None and sahi_gpu is not None and gpu_before + sahi_gpu >= cfg.budgets.max_gpu_ms:
                    return "gpu_budget_exhausted"
                return None

            try:
                cached = self.store.get_step(key, "sahi")
                if cached is None:
                    r = run_sliced(det, image, ref, class_ids, cfg.slicing, cfg.matching.class_mapping,
                                   max_slices=cfg.budgets.max_slices, stop_check=stop_check)
                    cached = {"detections": _dump(r.detections), "raw": _dump(r.raw), "gpu_ms": r.gpu_ms,
                              "warnings": r.warnings, "slices_run": r.slices_run, "stopped_reason": r.stopped_reason}
                    self.store.put_step(key, "sahi", cached)
                sahi, sahi_raw = _dets(cached["detections"]), _dets(cached["raw"])
                add_gpu(cached["gpu_ms"])
                slices_used = cached["slices_run"]
                warnings += cached["warnings"]
                if cached["stopped_reason"]:
                    failures.append(f"sahi_partial:{cached['stopped_reason']}")
                ran_sahi = True
                events.append(f"sahi:{len(sahi)}/{slices_used}_slices")
            except Exception as e:
                failures.append(f"sahi_failed:{type(e).__name__}")

        self._step = "fusion"
        # Stage B after secondary: class-aware matching and directional disagreement.
        dis = unavailable
        mislocalized_ids: set[str] = set()
        if ran_secondary:
            m = match(primary, secondary, cfg.matching.iou_threshold)
            dis = disagreement(m, cfg.matching.mislocalized_iou)
            mislocalized_ids = {i for p, s, v in m.pairs if v < cfg.matching.mislocalized_iou for i in (p.id, s.id)}
        diff_final = estimate_difficulty(sa, features, dis, cfg) if ran_secondary else diff_pre

        candidates = self._candidates(primary, secondary, sahi, decision, ran_secondary or ran_sahi, mislocalized_ids)

        self._step = "vlm_verify"
        verifications: list[VerificationResult] = []
        missing_checks: list[VerificationResult] = []
        vlm_used = 0
        if "vlm_verify" in decision.actions and self.vlm is not None:
            candidates, verifications, missing_checks, vlm_used = self._verify(
                image, ref, key, sa, candidates, within_budget, failures)

        self._step = "record"
        completeness = audit(candidates, sa, decision, missing_checks, cfg, failure_reasons=failures)
        used = Budget(slices=slices_used, vlm_calls=vlm_used, elapsed_ms=round(elapsed_ms(), 2), gpu_ms=gpu_total)
        rec = RouteRecord(
            record_id=key, image=ref, task=cfg.task, run=self.run_info, stage_a=sa,
            primary_candidates=tuple(primary), secondary_candidates=tuple(secondary), sahi_candidates=tuple(sahi),
            sahi_raw_candidates=tuple(sahi_raw),
            stage_b=StageB(difficulty_pre_secondary=diff_pre, difficulty_final=diff_final,
                           features=features, disagreement=dis),
            route=decision, verifications=tuple(verifications + missing_checks),
            final_candidates=tuple(candidates), completeness=completeness,
            trace=Trace(elapsed_ms=used.elapsed_ms, gpu_ms=gpu_total, budget_used=used,
                        events=tuple(events), warnings=tuple(dict.fromkeys(warnings + failures))),
        )
        self.store.put_record(rec)
        return rec

    def _candidates(self, primary, secondary, sahi, decision, ran_extra, mislocalized_ids) -> list[FinalCandidate]:
        out = []
        rc = decision.reason_codes
        low_rel = "low_relevance_with_detections" in rc
        risky_unconfirmed = "high_miss_risk" in rc and not ran_extra
        # The route wanted a second detector (moderate route, or secondary planned but skipped/failed)
        # and none ran: primary boxes are not confirmed by anything, so they must not be auto-accepted.
        moderate_unconfirmed = (not ran_extra and not risky_unconfirmed
                                and ("secondary" in decision.actions or "secondary_unavailable" in rc))
        for fc in fuse([*primary, *secondary, *sahi], self.cfg.matching.iou_threshold):
            flags: list[str] = []
            has_primary = "primary" in fc.sources
            if risky_unconfirmed:
                flags.append("high_miss_risk_unconfirmed")
            if moderate_unconfirmed:
                flags.append("secondary_unconfirmed")
            if fc.cross_class_conflict:
                flags.append("cross_class_conflict")
            if any(d.id in mislocalized_ids for d in fc.members):
                flags.append("mislocalized")
            if has_primary and ran_extra and fc.sources == {"primary"}:
                flags.append("primary_only_disagreement")
            if not has_primary:
                flags.append("detector_proposal:" + "+".join(sorted(fc.sources)))
            if low_rel:
                flags.append("low_relevance_image")
            if not has_primary:
                status = "candidate"
            elif flags:
                status = "needs_review"
            else:
                status = "accepted"
            out.append(final_candidate(fc, status, flags))
        return sorted(out, key=lambda c: c.id)

    def _verify(self, image, ref, key, sa, candidates, within_budget, failures):
        cfg, vlm = self.cfg, self.vlm
        names = {c.class_id: c.positive_prompts[0] for c in cfg.task.classes}
        cap = cfg.budgets.max_vlm_calls
        calls = 0
        verifications: list[VerificationResult] = []

        def ask(req: VerificationRequest) -> VerificationResult:
            nonlocal calls
            calls += 1
            cached = self.store.get_step(key, f"vlm:{req.id}")
            if cached is not None:
                return VerificationResult.model_validate(cached)
            try:
                res = vlm.verify(image, req, names[req.class_id])
            except Exception as e:  # adapters should not raise, but never trust that
                res = uncertain(req, "vlm_unavailable", vlm.model_version, type(e).__name__)
            if not self._answers(res, req):
                res = uncertain(req, "vlm_contradictory", vlm.model_version)
            self.store.put_step(key, f"vlm:{req.id}", res.model_dump(mode="json"))
            return res

        updated = []
        for c in candidates:
            if c.status not in ("needs_review", "candidate"):
                updated.append(c)
                continue
            if calls >= cap or not within_budget("vlm_verify"):
                failures.append("vlm_call_cap_reached" if calls >= cap else "vlm_skipped_budget")
                updated.append(c)
                continue
            req = VerificationRequest(id=stable_id("vreq", [key, "confirm", c.id]), image_id=ref.id,
                                      class_id=c.class_id, candidate_ids=(c.id,), region_xyxy=c.box_xyxy,
                                      purpose="confirm_candidate")
            res = ask(req)
            verifications.append(res)
            flags = c.flags + ("vlm_absent",) if res.verdict == "absent" and not cfg.vlm.absent_rejects else c.flags
            updated.append(c.model_copy(update={"status": self._apply(c, res), "verification": res, "flags": flags}))

        accepted_classes = {c.class_id for c in updated if c.status == "accepted"}
        missing: list[VerificationResult] = []
        for cid in cfg.task.class_ids:
            if cid in accepted_classes or not class_is_high_relevance(sa, cid, cfg):
                continue
            if calls >= cap or not within_budget("vlm_missing_class"):
                failures.append("vlm_call_cap_reached" if calls >= cap else "vlm_skipped_budget")
                continue
            req = VerificationRequest(id=stable_id("vreq", [key, "missing", cid]), image_id=ref.id,
                                      class_id=cid, purpose="suspected_missing_class")
            missing.append(ask(req))
        return sorted(updated, key=lambda c: c.id), verifications, missing, calls

    @staticmethod
    def _answers(res: object, req: VerificationRequest) -> bool:
        """True only if ``res`` is a verification of exactly this request (class, candidates, region)."""
        return (isinstance(res, VerificationResult) and res.request_id == req.id and res.class_id == req.class_id
                and tuple(res.candidate_ids) == tuple(req.candidate_ids) and res.region_xyxy == req.region_xyxy)

    def _apply(self, c: FinalCandidate, res: VerificationResult) -> str:
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
