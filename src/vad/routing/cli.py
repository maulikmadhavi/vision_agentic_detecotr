"""Command line entry point.

    vad-route export-schema --out schemas/
    vad-route shadow  --config configs/routing/example.yaml --images fixtures/manifest.jsonl \
                      --primary-predictions preds.jsonl --primary-version mymodel@abc --out runs/shadow1
    vad-route legacy  --config ... --images ... --primary-predictions ... --primary-version ... --out runs/legacy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import LMStudioVLM, PrecomputedDetector, PrecomputedRelevance
from .evaluation import route_report
from .export import export_legacy, export_routed
from .pipeline import RouteStore, RoutingPipeline, StoreLockedError, job_lock, load_image, load_manifest
from .schemas import ErrorRecord, RouteRecord, export_json_schemas
from .task_config import load_config


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vad-route")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export-schema")
    s.add_argument("--out", default="schemas")
    for name in ("shadow", "legacy"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--images", required=True, help="image directory or JSONL manifest")
        p.add_argument("--primary-predictions", required=True)
        p.add_argument("--primary-version", required=True, help="model identifier only (not verified); the file's SHA-256 is bound into the run key separately")
        p.add_argument("--out", required=True)
        if name == "shadow":
            p.add_argument("--secondary-predictions")
            p.add_argument("--secondary-version", help="model identifier only (not verified); the file's SHA-256 is bound into the run key separately")
            rel = p.add_mutually_exclusive_group()
            rel.add_argument("--relevance-scores")
            rel.add_argument("--clip-weights", help="local open_clip weights file for live Stage A relevance "
                                                    "(needs the [clip] extra and relevance.raw_scale: clip_prompt_softmax_v1)")
            p.add_argument("--clip-arch", default="ViT-B-16-quickgelu", help="open_clip architecture for --clip-weights")
            p.add_argument("--relevance-version", default="precomputed", help="scorer identifier only (not verified); the file's SHA-256 is bound into the run key separately")
            p.add_argument("--vlm", action="store_true", help="enable LM Studio verification (overrides config)")
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if getattr(args, "secondary_predictions", None) and not args.secondary_version:
        parser.error("--secondary-version is required with --secondary-predictions (model versions are part of the record hash)")
    if args.cmd == "export-schema":
        for f in export_json_schemas(args.out):
            print(f)
        return 0

    cfg = load_config(args.config)
    out = Path(args.out)
    entries = load_manifest(args.images)
    primary = PrecomputedDetector(args.primary_predictions, args.primary_version)

    if args.cmd == "legacy":
        preds = []
        pipe = RoutingPipeline(cfg, primary, RouteStore(":memory:"))
        for e in entries:
            img, ref = load_image(e)
            preds.append(({"file_name": ref.uri, "width": ref.width, "height": ref.height, "vad_image_id": ref.id},
                          pipe.legacy_predict(img, ref)))
        export_legacy(preds, cfg.task, out / "annotations.json")
        print(f"legacy primary-only export: {out / 'annotations.json'}")
        return 0

    # Shadow mode: force-enable routing for this run only; nothing is discarded.
    upd = {"routing_enabled": True}
    if args.vlm:
        upd["vlm"] = cfg.vlm.model_copy(update={"enabled": True})
    cfg = cfg.model_copy(update=upd)
    secondary = (PrecomputedDetector(args.secondary_predictions, args.secondary_version)
                 if args.secondary_predictions else None)
    relevance = PrecomputedRelevance(args.relevance_scores, args.relevance_version) if args.relevance_scores else None
    if args.clip_weights:
        from .embedder import SCORE_SCALE, OpenClipRelevance
        if cfg.relevance.raw_scale != SCORE_SCALE:
            parser.error(f"--clip-weights needs relevance.raw_scale: {SCORE_SCALE} (with bands chosen for that scale)")
        relevance = OpenClipRelevance(args.clip_weights, args.clip_arch)
    vlm = LMStudioVLM(cfg.vlm.base_url, cfg.vlm.model, cfg.vlm.timeout_s, cfg.vlm.max_tokens,
                      cfg.vlm.crop_padding) if cfg.vlm.enabled else None
    try:
        with job_lock(out):
            store = RouteStore(out / "routes.sqlite")
            pipe = RoutingPipeline(cfg, primary, store, secondary=secondary, relevance=relevance, vlm=vlm)
            # Export only this run's results: a reused --out store may hold records from
            # earlier runs (other manifests, tasks or configs) that must not leak in.
            results = pipe.run(entries)
            records = sorted((r for r in results if isinstance(r, RouteRecord)), key=lambda r: r.image.id)
            errors = [r for r in results if isinstance(r, ErrorRecord)]
            export_routed(records, cfg.task, out / "annotations.json")
            with open(out / "route_trace.jsonl", "w", encoding="utf-8") as f:
                for r in records:
                    f.write(r.model_dump_json() + "\n")
            report = route_report(records)
            report["errors"] = [e.model_dump(mode="json") for e in errors]
            (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except StoreLockedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
