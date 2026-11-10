"""Command line entry point.

    vad-route export-schema --out schemas/
    vad-route shadow  --config configs/routing/example.yaml --images fixtures/manifest.jsonl \
                      --primary-predictions preds.jsonl --primary-version mymodel@abc --out runs/shadow1
    vad-route legacy  --config ... --images ... --primary-predictions ... --primary-version ... --out runs/legacy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from service_detector_gdino.client import RemoteDetector
from service_detector_vlm.openai_service_call import OpenAICompatibleVerifier
from services_embedding.client import WEMM_TEXT_SCALE, WeMMMCPRelevanceScorer

from .calibration import CalibrationError, FittedRelevanceCalibrator, fit_relevance
from .evaluation import (
    CONTENT_FIELDS, IsolationError, check_image_content, load_frozen, lock_frozen_set, route_report,
)
from .export import export_legacy, export_reviewed, export_routed
from .image_inputs import ManifestEntry, RepeatedImageError, image_id, load_image, load_manifest, repeated_ids
from .interfaces import Detector, RelevanceScorer
from .metrics import (
    ROUTED_RULE, LabelError, evaluate, evidence_status, image_results, load_ground_truth, per_image_outcomes,
)
from .pipeline import RoutingPipeline
from .replay import PrecomputedDetector, PrecomputedRelevance
from .review import AdjudicationLogError, ReviewError, apply_adjudications, open_findings, read_adjudications
from .sample import (
    SampleError, UnresolvedReviews, frozen_label_subset, group_manifest, label_digest, make_split, merge_reviews,
    review_queue,
)
from .store import RouteStore, StoreLockedError, job_lock
from .schemas import ErrorRecord, RouteRecord, TaskSpec, canonical_json, export_json_schemas, stable_id
from .task_config import RoutingConfig, load_config, shadow_run_config


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vad-route")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export-schema")
    s.add_argument("--out", default="schemas")
    e = sub.add_parser("evaluate", help="detection metrics of a shadow run against ground-truth labels")
    e.add_argument("--config", required=True)
    e.add_argument("--labels", required=True, help="COCO JSON with vad_* fields (see routing.metrics)")
    e.add_argument("--trace", required=True, help="route_trace.jsonl of the shadow run")
    e.add_argument("--out", required=True, help="run-metrics-v1 JSON to write (never overwritten)")
    e.add_argument("--parent", help="the run-metrics file of the run this one follows (its metrics_id is recorded)")
    e.add_argument("--iou", type=float, default=0.5, help="working IoU threshold for recall, false positives, misses")
    e.add_argument("--split-manifest", help="with --frozen-lock: score only the frozen_eval images, after verifying")
    e.add_argument("--frozen-lock", help="the frozen lock written by `vad-route sample split`")
    sample = sub.add_parser("sample", help="tooling for the human-checked trial sample")
    sample_sub = sample.add_subparsers(dest="sample_cmd", required=True)
    g = sample_sub.add_parser("group", help="group exact and near-duplicate images; list pairs for confirmation")
    g.add_argument("--images", required=True, help="image directory or JSONL manifest")
    g.add_argument("--out", required=True, help="groups JSON to write")
    g.add_argument("--max-hamming", type=int, default=6, help="near-duplicate dHash distance that merges two images")
    sp = sample_sub.add_parser("split", help="seeded group-wise split; writes a versioned manifest and the frozen lock")
    sp.add_argument("--config", required=True, help="routing config (for the task's class ids)")
    sp.add_argument("--groups", required=True, help="groups JSON from `vad-route sample group`")
    sp.add_argument("--exclusions", required=True, help="exclusions JSON, recorded before the split")
    sp.add_argument("--labels", required=True, help="ground-truth labels (COCO JSON with vad_* fields)")
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--out-dir", required=True, help="new directory for split_manifest.json and frozen_lock.json")
    rq = sample_sub.add_parser("review-queue", help="images for an independent second review")
    rq.add_argument("--config", required=True)
    rq.add_argument("--labels", required=True, help="first-pass labels (one reviewer per image)")
    rq.add_argument("--groups", required=True, help="groups JSON from `vad-route sample group`")
    rq.add_argument("--seed", type=int, required=True)
    rq.add_argument("--out", required=True)
    mg = sample_sub.add_parser("merge", help="merge first and second review; disagreements need adjudication")
    mg.add_argument("--config", required=True)
    mg.add_argument("--first", required=True, help="first-pass labels")
    mg.add_argument("--second", required=True, help="second-pass labels for exactly the queued images")
    mg.add_argument("--queue", required=True, help="queue JSON from `vad-route sample review-queue`")
    mg.add_argument("--adjudications", help='{"vad_adjudications": 1, "decisions": {image_id: {decision, adjudicator}}}')
    mg.add_argument("--out", required=True, help="merged label file (A3 format)")
    review = sub.add_parser("review", help="human review of a shadow run: findings to decide, then apply decisions")
    review_sub = review.add_subparsers(dest="review_cmd", required=True)
    rf = review_sub.add_parser("findings", help="list the open findings of a shadow run")
    rf.add_argument("--trace", required=True, help="route_trace.jsonl of the shadow run")
    rf.add_argument("--out", required=True, help="findings JSONL to write")
    ra = review_sub.add_parser("apply", help="apply adjudications; write reviewed records and the reviewed export")
    ra.add_argument("--config", required=True, help="the config the shadow run used")
    ra.add_argument("--trace", required=True, help="route_trace.jsonl of the shadow run (never modified)")
    ra.add_argument("--adjudications", required=True, help="append-only adjudications JSONL")
    ra.add_argument("--out-dir", required=True, help="directory for reviewed.jsonl and annotations.json")
    ra.add_argument("--split-manifest", help="split manifest; tags each exported image with vad_split")
    fr = sub.add_parser("fit-relevance", help="fit a relevance calibrator on the calibration split")
    fr.add_argument("--config", required=True, help="the config file the shadow run used (its hash must match the "
                                                    "trace, allowing shadow's routing/--vlm overrides)")
    fr.add_argument("--trace", required=True, help="route_trace.jsonl of a shadow run over calibration images")
    fr.add_argument("--labels", required=True, help="ground-truth labels (COCO JSON with vad_* fields)")
    fr.add_argument("--split-manifest", required=True, help="only calibration-split images may be used")
    fr.add_argument("--out", required=True, help="calibrator JSON to write (never overwritten)")
    for name in ("shadow", "legacy"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--images", required=True, help="image directory or JSONL manifest")
        version_help = "model identifier only (not verified); the file's SHA-256 is bound into the run key separately"
        if name == "legacy":
            p.add_argument("--primary-predictions", required=True)
            p.add_argument("--primary-version", required=True, help=version_help)
        else:
            src = p.add_mutually_exclusive_group(required=True)
            src.add_argument("--primary-predictions")
            src.add_argument("--primary-url", help="svc-1.0 detector server base URL (RemoteDetector)")
            p.add_argument("--primary-version", help=version_help + "; with --primary-predictions only")
            p.add_argument("--primary-model-id", help="expected model id reported by --primary-url")
            p.add_argument("--primary-revision", help="expected model revision reported by --primary-url")
            p.add_argument("--primary-weights-sha256", help="expected weights digest (verified per response)")
            p.add_argument("--primary-timeout", type=float, default=60.0, help="seconds per request to --primary-url")
        p.add_argument("--out", required=True)
        if name == "shadow":
            p.add_argument("--secondary-predictions")
            p.add_argument("--secondary-version", help="model identifier only (not verified); the file's SHA-256 is bound into the run key separately")
            rel = p.add_mutually_exclusive_group()
            rel.add_argument("--relevance-scores")
            rel.add_argument("--wemm", action="store_true", help="Stage A relevance from the WeMM embedding server at "
                                                                 "$VAD_WEMM_URL (needs relevance.raw_scale: wemm_text_cosine_v1)")
            p.add_argument("--wemm-model", default="tencent/WeMM-Embedding-4B", help="expected meta.model (verified)")
            p.add_argument("--wemm-dim", type=int, default=2560, help="expected meta.dim (verified)")
            p.add_argument("--wemm-build", help="expected meta.build; without it the version is recorded as unverified")
            p.add_argument("--wemm-max-side", type=int, default=448, help="downscale longest image side before sending")
            p.add_argument("--relevance-version", default="precomputed", help="scorer identifier only (not verified); the file's SHA-256 is bound into the run key separately")
            p.add_argument("--relevance-calibrator", help="calibrator JSON from `vad-route fit-relevance`; refused "
                                                          "unless its scale, scorer and prompts match this run")
            p.add_argument("--vlm", action="store_true", help="enable VLM verification at vlm.base_url (overrides config)")
            p.add_argument("--split-manifest", help="split manifest; tags each exported image with vad_split")
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.cmd == "export-schema":
        for path in export_json_schemas(args.out):
            print(path)
        return 0
    if args.cmd == "evaluate":
        return _run_evaluate(parser, args)
    if args.cmd == "sample":
        return _run_sample(parser, args)
    if args.cmd == "review":
        return _run_human_review(parser, args)
    if args.cmd == "fit-relevance":
        return _run_fit_relevance(parser, args)

    cfg = load_config(args.config)
    out = Path(args.out)
    entries = load_manifest(args.images)
    if getattr(args, "primary_url", None):
        primary = RemoteDetector(base_url=args.primary_url, task=cfg.task, model_id=args.primary_model_id,
                                 model_revision=args.primary_revision, weights_sha256=args.primary_weights_sha256,
                                 timeout_s=args.primary_timeout)
    else:
        primary = PrecomputedDetector(args.primary_predictions, args.primary_version)

    if args.cmd == "legacy":
        return _run_legacy(cfg, primary, entries, out)
    return _run_shadow(parser, args, cfg, primary, entries, out)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Flag combinations argparse cannot express; each failure exits with code 2."""
    if getattr(args, "secondary_predictions", None) and not args.secondary_version:
        parser.error("--secondary-version is required with --secondary-predictions (model versions are part of the record hash)")
    if args.cmd != "shadow":
        return
    remote_flags = (args.primary_model_id, args.primary_revision, args.primary_weights_sha256)
    if args.primary_url and args.primary_version:
        parser.error("--primary-version is not used with --primary-url (the version comes from "
                     "--primary-model-id/--primary-revision)")
    if args.primary_url and not (args.primary_model_id and args.primary_revision):
        parser.error("--primary-url needs --primary-model-id and --primary-revision")
    if not args.primary_url and not args.primary_version:
        parser.error("--primary-version is required with --primary-predictions")
    if not args.primary_url and any(remote_flags):
        parser.error("--primary-model-id/--primary-revision/--primary-weights-sha256 are only with --primary-url")


RUN_METRICS_VERSION = "run-metrics-v1"


def _content_sha256(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def _run_evaluate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists():
        parser.error(f"{out} already exists; run metrics are immutable, so write each evaluation to a new --out")
    parent_id = None
    if args.parent:
        parent = json.loads(Path(args.parent).read_text(encoding="utf-8"))
        body = {k: v for k, v in parent.items() if k != "metrics_id"}
        if parent.get("version") != RUN_METRICS_VERSION or parent.get("metrics_id") != stable_id("met", body):
            parser.error(f"--parent {args.parent} is not an unmodified {RUN_METRICS_VERSION} file")
        parent_id = parent["metrics_id"]
    cfg = load_config(args.config)
    task = cfg.task
    label_doc = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    try:
        ground_truth = load_ground_truth(label_doc, task)
    except LabelError as e:
        parser.error(f"--labels: {e}")
    if bool(args.split_manifest) != bool(args.frozen_lock):
        parser.error("--split-manifest and --frozen-lock go together")
    split = {"status": "unverified: no --split-manifest/--frozen-lock; every labelled image is scored, "
                       "whatever its split", "manifest_sha256": None}
    manifest = None
    lines = Path(args.trace).read_text(encoding="utf-8").splitlines()
    records = [RouteRecord.model_validate_json(line) for line in lines if line.strip()]
    try:
        results = image_results(records)  # refuses repeated images before anything is scored or written
    except LabelError as e:
        parser.error(f"--trace: {e}")
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        lock = json.loads(Path(args.frozen_lock).read_text(encoding="utf-8"))
        frozen_ids = sorted(i for i, img in manifest["images"].items() if img["split"] == "frozen_eval")
        if manifest.get("frozen_lock_sha256") != lock.get("lock_sha256") or lock.get("image_ids") != frozen_ids:
            parser.error("--frozen-lock does not belong to --split-manifest (lock hash or frozen images differ)")
        try:
            load_frozen(args.frozen_lock, frozen_label_subset(label_doc, frozen_ids))
        except IsolationError as e:
            parser.error(f"the frozen labels do not verify against --frozen-lock: {e}")
        ground_truth = {i: ground_truth[i] for i in frozen_ids}
        split = {"frozen_images": len(frozen_ids), "lock_sha256": lock["lock_sha256"],
                 "manifest_sha256": _content_sha256(manifest)}
        if "images" not in lock:  # a lock from before image hashes: labels verified, image content not
            split["status"] = ("unverified: the frozen labels match --frozen-lock, but the lock predates image "
                               "hashes, so image content is not checked; re-lock (frozen-lock-v2) to bind it")
        else:
            if any({k: manifest["images"][i].get(k) for k in CONTENT_FIELDS} != lock["images"][i] for i in frozen_ids):
                parser.error("--frozen-lock does not belong to --split-manifest (frozen image content differs)")
            try:
                check_image_content(lock["images"], [r.image for r in records if r.image.id in lock["images"]],
                                    "--frozen-lock")
            except IsolationError as e:
                parser.error(f"a frozen result was routed on other image content: {e}")
            split = {"status": "verified frozen_eval", **split,
                     "image_content": "bound: each frozen result matches the lock's sha256 and size"}
    # One cohort for accuracy AND cost: the labelled images in scope (the frozen ones when a split is
    # given). Labelled images without a result stay in it as operational failures; other results
    # are only listed, so a development or unlabelled record cannot change any cohort metric.
    cohort = [r for r in records if r.image.id in ground_truth]
    outside = sorted(r.image.id for r in records if r.image.id not in ground_truth)
    cohort_results = {r.image.id: results[r.image.id] for r in cohort}
    metrics = {"evidence_status": evidence_status(label_doc, split["status"]), "split": split,
               **evaluate(ground_truth, task.class_ids, cohort_results, args.iou),
               "cost": route_report(cohort), "out_of_cohort": {"results": len(outside), "image_ids": outside}}
    # run-metrics-v1: immutable, identified by its own content; ids, scores and match codes only.
    cohort_ids = sorted(ground_truth)
    runs = [json.loads(r) for r in sorted({canonical_json(rec.run) for rec in cohort})]
    per_image = per_image_outcomes(ground_truth, cohort_results, task.class_ids)
    for image_id, outcome in per_image.items():
        outcome["group_id"] = manifest["images"][image_id].get("group_id") if manifest else None
    labels = {**frozen_label_subset(label_doc, cohort_ids), "vad_label_provenance": label_doc.get("vad_label_provenance")}
    body = {
        "version": RUN_METRICS_VERSION, "run_id": stable_id("run", runs), "parent_id": parent_id,
        "runs": runs, "mixed_runs": len(runs) > 1, "seed": cfg.audit.seed,
        "series": {"cohort_ids_sha256": _content_sha256(cohort_ids), "labels_sha256": label_digest(labels),
                   # only a verified split binds image content (frozen-lock-v2 == manifest == records);
                   # an old lock or a split-v1 manifest gives null, never a derived or upgraded hash
                   "images_sha256": _content_sha256([[i, manifest["images"][i]["sha256"]] for i in cohort_ids])
                   if split["status"] == "verified frozen_eval" else None,
                   "evaluators": {"custom": metrics["detector"]["evaluator"],
                                  "coco": f"pycocotools {metrics['detector']['coco']['version']}",
                                  "routed_rule": ROUTED_RULE.split(":", 1)[0]}},
        **metrics, "per_image": per_image,
    }
    doc = {"version": RUN_METRICS_VERSION, "metrics_id": stable_id("met", body), **body}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"evidence_status: {metrics['evidence_status']}")
    print(f"metrics written to {out}")
    return 0


def _run_sample(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.sample_cmd == "split":
        return _run_split(parser, args)
    if args.sample_cmd in ("review-queue", "merge"):
        return _run_review(parser, args)
    doc = group_manifest(load_manifest(args.images), args.max_hamming)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    groups = {v["group_id"] for v in doc["images"].values()}
    to_check = sum(1 for p in doc["pairs"] if p["reason"] == "near_duplicate")
    print(f"{len(doc['images'])} images in {len(groups)} groups; {to_check} near-duplicate pairs to confirm; "
          f"{len(doc['unreadable'])} unreadable; written to {out}")
    return 0


def _run_fit_relevance(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists():
        parser.error(f"{out} already exists; calibrators are versioned, write a new --out")
    docs = {name: json.loads(Path(getattr(args, name)).read_text(encoding="utf-8"))
            for name in ("labels", "split_manifest")}
    try:
        artifact = fit_relevance(_read_trace(args.trace), docs["labels"], docs["split_manifest"],
                                 load_config(args.config))
    except (CalibrationError, IsolationError, LabelError) as e:
        parser.error(str(e))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact["label_provenance"] != "human":
        print("NOT A FINDING: the labels do not declare vad_label_provenance: human; this calibrator only "
              "tests the tooling", file=sys.stderr)
    separable = [c for c, p in artifact["classes"].items() if p["separable"]]
    print(f"calibrator written to {out}" + (f"; separable (ridge-limited): {separable}" if separable else ""))
    return 0


def _read_trace(path: str) -> list[RouteRecord]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [RouteRecord.model_validate_json(line) for line in lines if line.strip()]


def _run_config(parser: argparse.ArgumentParser, cfg: RoutingConfig, records: list[RouteRecord]) -> RoutingConfig:
    """The config a shadow run actually hashed: shadow enables routing, and --vlm enables the VLM."""
    hashes = {r.run.config_hash for r in records}
    if (variant := shadow_run_config(cfg, hashes)) is not None:
        return variant
    parser.error(f"--config does not match the run (run config hashes {sorted(hashes)}); "
                 "the re-audit needs the config the shadow run used")


def _run_human_review(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    records = _read_trace(args.trace)
    if args.review_cmd == "findings":
        findings = [f for r in records for f in open_findings(r)]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(f, sort_keys=True) + "\n" for f in findings), encoding="utf-8")
        print(f"{len(findings)} open findings in {len({f['record_id'] for f in findings})} images; written to {out}")
        return 0
    cfg = _run_config(parser, load_config(args.config), records)
    splits = None
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        if manifest.get("version") not in ("split-v1", "split-v2"):
            parser.error("--split-manifest must be a split manifest from `vad-route sample split`")
        splits = {i: img["split"] for i, img in manifest["images"].items()}
        if missing := sorted(r.image.id for r in records if r.image.id not in splits):
            parser.error(f"images not in --split-manifest: {missing[:5]}")
    try:
        reviewed = apply_adjudications(records, read_adjudications(args.adjudications), cfg)
    except (ReviewError, AdjudicationLogError) as e:
        parser.error(str(e))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reviewed.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in reviewed), encoding="utf-8")
    export_reviewed(reviewed, cfg.task, out / "annotations.json", splits)
    if splits and (frozen := sum(len(r.adjudications) for r in reviewed if splits[r.image.id] == "frozen_eval")):
        print(f"warning: {frozen} adjudications on frozen_eval images: kept, and those images are exported with "
              "vad_split=frozen_eval so fitting and training refuse them", file=sys.stderr)
    done = sum(r.completeness.status == "complete" for r in reviewed)
    print(f"{len(reviewed)} images reviewed, {done} complete; written to {out}")
    return 0


def _run_review(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    task, out = load_config(args.config).task, Path(args.out)
    try:
        if args.sample_cmd == "review-queue":
            doc = review_queue(read(args.labels), read(args.groups), task, args.seed)
            summary = f"{len(doc['images'])} images queued for a second review"
        else:
            adjudications = read(args.adjudications) if args.adjudications else {}
            doc, disagreements = merge_reviews(read(args.first), read(args.second), read(args.queue), adjudications,
                                               task)
            summary = f"merged {len(doc['images'])} images; {len(disagreements)} adjudicated disagreements"
    except UnresolvedReviews as e:
        report = out.with_suffix(".disagreements.json")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(e.disagreements, indent=2), encoding="utf-8")
        parser.error(f"{e}; details in {report}; nothing merged")
    except (SampleError, LabelError) as e:
        parser.error(str(e))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{summary}; written to {out}")
    return 0


def _run_split(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    if (out / "split_manifest.json").exists() or (out / "frozen_lock.json").exists():
        parser.error(f"{out} already holds a split; a split is versioned, so write it to a new --out-dir")
    docs = {name: json.loads(Path(getattr(args, name)).read_text(encoding="utf-8"))
            for name in ("groups", "exclusions", "labels")}
    try:
        manifest, frozen = make_split(docs["groups"], docs["exclusions"], docs["labels"],
                                      load_config(args.config).task, args.seed)
    except (SampleError, LabelError) as e:
        parser.error(str(e))
    frozen_ids = [i for i, img in manifest["images"].items() if img["split"] == "frozen_eval"]
    frozen_groups = [g for g, v in manifest["groups"].items() if v["split"] == "frozen_eval"]
    lock = lock_frozen_set(frozen_ids, frozen_groups, frozen, out / "frozen_lock.json",
                           content={i: manifest["images"][i] for i in frozen_ids})
    manifest["frozen_lock_sha256"] = lock["lock_sha256"]
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for warning in manifest["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    summary = ", ".join(f"{name} {c['groups']} groups / {c['images']} images" for name, c in manifest["counts"].items())
    print(f"split written to {out}: {summary}")
    return 0


def _run_legacy(cfg: RoutingConfig, primary: Detector, entries: list[ManifestEntry], out: Path) -> int:
    predictions = []
    pipe = RoutingPipeline(cfg, primary, RouteStore(":memory:"))
    for entry in entries:
        image, ref = load_image(entry)
        predictions.append(({"file_name": ref.uri, "width": ref.width, "height": ref.height, "vad_image_id": ref.id},
                            pipe.legacy_predict(image, ref)))
    export_legacy(predictions, cfg.task, out / "annotations.json")
    print(f"legacy primary-only export: {out / 'annotations.json'}")
    return 0


def _build_relevance_scorer(parser: argparse.ArgumentParser, args: argparse.Namespace,
                            cfg: RoutingConfig) -> RelevanceScorer | None:
    if args.relevance_scores:
        return PrecomputedRelevance(args.relevance_scores, args.relevance_version)
    if not args.wemm:
        return None
    url = os.environ.get("VAD_WEMM_URL")
    if not url:
        parser.error("--wemm reads the server endpoint from $VAD_WEMM_URL, which is not set")
    if cfg.relevance.raw_scale != WEMM_TEXT_SCALE:
        parser.error(f"--wemm needs relevance.raw_scale: {WEMM_TEXT_SCALE} "
                     "(with bands chosen for that scale)")
    return WeMMMCPRelevanceScorer(url=url, model=args.wemm_model, dim=args.wemm_dim, build=args.wemm_build,
                                  max_side=args.wemm_max_side)


def _run_shadow(parser: argparse.ArgumentParser, args: argparse.Namespace, cfg: RoutingConfig, primary: Detector,
                entries: list[ManifestEntry], out: Path) -> int:
    # Shadow mode: force-enable routing for this run only; nothing is discarded.
    overrides = {"routing_enabled": True}
    if args.vlm:
        overrides["vlm"] = cfg.vlm.model_copy(update={"enabled": True})
    cfg = cfg.model_copy(update=overrides)
    ids = []  # the ids the pipeline will record, checked before anything is created or processed
    for entry in entries:
        try:
            ids.append(image_id(entry, b"" if entry.image_id else Path(entry.uri).read_bytes()))
        except OSError:
            ids.append(entry.uri)  # unreadable and unnamed: its uri stands in for the id
    if repeated := repeated_ids(ids):
        files = {i: sorted({e.uri for e, j in zip(entries, ids) if j == i}) for i in repeated}
        conflicts = {i: f for i, f in files.items() if len(f) > 1}
        parser.error(f"repeated image ids in --images {repeated[:10]}; a run holds each image once"
                     + (f"; different files share an id: {conflicts}" if conflicts else ""))
    splits = None
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        if manifest.get("version") not in ("split-v1", "split-v2"):
            parser.error("--split-manifest must be a split manifest from `vad-route sample split`")
        splits = {i: img["split"] for i, img in manifest["images"].items()}
        if missing := sorted(i for i in ids if i not in splits):
            parser.error(f"images not in --split-manifest: {missing[:5]}")
    secondary = (PrecomputedDetector(args.secondary_predictions, args.secondary_version)
                 if args.secondary_predictions else None)
    relevance = _build_relevance_scorer(parser, args, cfg)
    calibrator = None
    if args.relevance_calibrator:
        if relevance is None:
            parser.error("--relevance-calibrator needs a relevance scorer (--wemm or --relevance-scores)")
        try:
            calibrator = FittedRelevanceCalibrator.load(args.relevance_calibrator, cfg, relevance.model_version)
        except (CalibrationError, OSError) as e:
            hint = " (with --relevance-scores, pass the fit-time scorer as --relevance-version)" \
                if args.relevance_scores and "scorer" in str(e) else ""
            parser.error(f"{e}{hint}")
    vlm = None
    if cfg.vlm.enabled:
        for name in ("base_url", "model"):
            if not getattr(cfg.vlm, name):
                parser.error(f"VLM verification needs vlm.{name} in the config (there is no built-in endpoint or model)")
        try:
            vlm = OpenAICompatibleVerifier(cfg.vlm.base_url, cfg.vlm.model, cfg.vlm.timeout_s, cfg.vlm.max_tokens,
                                           cfg.vlm.crop_padding, cfg.vlm.api_key_env)
        except ValueError as e:
            parser.error(str(e))
    try:
        with job_lock(out):
            store = RouteStore(out / "routes.sqlite")
            pipe = RoutingPipeline(cfg, primary, store, secondary=secondary, relevance=relevance,
                                   calibrator=calibrator, vlm=vlm)
            report = _write_shadow_outputs(pipe.run(entries), cfg.task, out, splits)
    except RepeatedImageError as e:
        parser.error(str(e))
    except StoreLockedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    if (failed := report["relevance_scorer_failures"])["count"]:
        print(f"warning: relevance scorer failed for {failed['count']} of {report['images']} images "
              f"({failed['types']}); their relevance is 'unavailable' and they were routed to review", file=sys.stderr)
    return 0


def _write_shadow_outputs(results: list[RouteRecord | ErrorRecord], task: TaskSpec, out: Path,
                          splits: dict[str, str] | None = None) -> dict:
    """Write annotations, trace and report for THIS run's results only.

    A reused --out store may hold records from earlier runs (other manifests, tasks or
    configs); those must not leak into the exports. Each image (RouteRecord.image.id, or an
    ErrorRecord's record_id) must appear once; otherwise nothing is written.
    """
    keys = [r.image.id if isinstance(r, RouteRecord) else r.record_id for r in results]
    if repeated := repeated_ids(keys):
        raise RepeatedImageError(f"repeated image ids in the run's results {repeated[:10]}; nothing was written")
    records = sorted((result for result in results if isinstance(result, RouteRecord)),
                     key=lambda record: record.image.id)
    errors = [result for result in results if isinstance(result, ErrorRecord)]
    export_routed(records, task, out / "annotations.json", splits)
    with open(out / "route_trace.jsonl", "w", encoding="utf-8") as trace_file:
        for record in records:
            trace_file.write(record.model_dump_json() + "\n")
    report = route_report(records)
    report["errors"] = [error.model_dump(mode="json") for error in errors]
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
