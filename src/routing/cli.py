"""Command line entry point.

    vad-route export-schema --out schemas/
    vad-route shadow  --config configs/routing/example.yaml --images fixtures/manifest.jsonl \
                      --primary-predictions preds.jsonl --primary-version mymodel@abc --out runs/shadow1
    vad-route legacy  --config ... --images ... --primary-predictions ... --primary-version ... --out runs/legacy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from service_detector_gdino.client import RemoteDetector
from service_detector_vlm.openai_service_call import OpenAICompatibleVerifier
from services_embedding.client import WEMM_TEXT_SCALE, WeMMMCPRelevanceScorer

from .evaluation import route_report
from .export import export_legacy, export_routed
from .image_inputs import ManifestEntry, load_image, load_manifest
from .interfaces import Detector, RelevanceScorer
from .pipeline import RoutingPipeline
from .replay import PrecomputedDetector, PrecomputedRelevance
from .store import RouteStore, StoreLockedError, job_lock
from .schemas import ErrorRecord, RouteRecord, TaskSpec, export_json_schemas
from .task_config import RoutingConfig, load_config


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vad-route")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export-schema")
    s.add_argument("--out", default="schemas")
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
            p.add_argument("--vlm", action="store_true", help="enable VLM verification at vlm.base_url (overrides config)")
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.cmd == "export-schema":
        for path in export_json_schemas(args.out):
            print(path)
        return 0

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
    secondary = (PrecomputedDetector(args.secondary_predictions, args.secondary_version)
                 if args.secondary_predictions else None)
    relevance = _build_relevance_scorer(parser, args, cfg)
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
            pipe = RoutingPipeline(cfg, primary, store, secondary=secondary, relevance=relevance, vlm=vlm)
            report = _write_shadow_outputs(pipe.run(entries), cfg.task, out)
    except StoreLockedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    if (failed := report["relevance_scorer_failures"])["count"]:
        print(f"warning: relevance scorer failed for {failed['count']} of {report['images']} images "
              f"({failed['types']}); their relevance is 'unavailable' and they were routed to review", file=sys.stderr)
    return 0


def _write_shadow_outputs(results: list[RouteRecord | ErrorRecord], task: TaskSpec, out: Path) -> dict:
    """Write annotations, trace and report for THIS run's results only.

    A reused --out store may hold records from earlier runs (other manifests, tasks or
    configs); those must not leak into the exports.
    """
    records = sorted((result for result in results if isinstance(result, RouteRecord)),
                     key=lambda record: record.image.id)
    errors = [result for result in results if isinstance(result, ErrorRecord)]
    export_routed(records, task, out / "annotations.json")
    with open(out / "route_trace.jsonl", "w", encoding="utf-8") as trace_file:
        for record in records:
            trace_file.write(record.model_dump_json() + "\n")
    report = route_report(records)
    report["errors"] = [error.model_dump(mode="json") for error in errors]
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
