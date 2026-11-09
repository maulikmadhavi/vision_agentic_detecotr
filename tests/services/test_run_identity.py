"""Run identity must not depend on module or class names (moves/renames must not invalidate caches)."""

from __future__ import annotations

from routing.adapters import LMStudioVLM, PrecomputedDetector, PrecomputedRelevance
from routing.pipeline import RouteStore, RoutingPipeline
from routing.schemas import ImageRef

from ..routing.helpers import config


def test_replay_and_vlm_run_key_is_unchanged_by_moves_and_renames(tmp_path):
    # Golden key computed before the src/vad -> src/routing + src/services restructure.
    (tmp_path / "p.jsonl").write_bytes(b'{"image_id": "a", "detections": []}\n')
    (tmp_path / "r.jsonl").write_bytes(b'{"image_id": "a", "scores": {}}\n')
    pipe = RoutingPipeline(config(vlm={"enabled": True}), PrecomputedDetector(tmp_path / "p.jsonl", "prim@1"),
                           RouteStore(":memory:"), relevance=PrecomputedRelevance(tmp_path / "r.jsonl", "rel@1"),
                           vlm=LMStudioVLM())
    run = pipe.run_info
    assert (run.primary_model, run.relevance_scorer, run.vlm_model) == ("prim@1", "rel@1", "lmstudio:qwen3-vl-4b-instruct")
    ref = ImageRef(id="a", uri="a.png", width=320, height=240, sha256="0" * 64)
    assert pipe.record_key(ref) == "rec-bf54146413d69a17"


def test_services_resolves_to_src_not_the_tests_package():
    # tests/ and tests/services/ are packages (tests/__init__.py), so pytest's default "prepend"
    # import mode imports them as tests.services, never as a top-level `services` that would
    # shadow src/services.
    from pathlib import Path

    import routing
    import services
    import services.service_contracts as contracts

    assert Path(services.__file__).parent.name == "services" and Path(services.__file__).parent.parent.name == "src"
    assert Path(routing.__file__).parent.parent.name == "src"
    assert contracts.SERVICE_SCHEMA_VERSION == "svc-1.0"
    assert __name__ == "tests.services.test_run_identity"
