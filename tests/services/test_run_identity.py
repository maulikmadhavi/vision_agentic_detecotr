"""Run identity must not depend on module or class names (moves/renames must not invalidate caches)."""

from __future__ import annotations

import pytest

from service_detector_vlm.openai_service_call import OpenAICompatibleVerifier
from routing.replay import PrecomputedDetector, PrecomputedRelevance
from routing.pipeline import RoutingPipeline
from routing.store import RouteStore
from routing.schemas import ImageRef

from ..routing.helpers import config


def test_replay_and_vlm_run_key_is_unchanged_by_moves_and_renames(tmp_path):
    # Golden key. MIGRATION 2026-11-10 (user decision, board D-11): the VLM identity changed from
    # "lmstudio:{model}" to "openai-compatible:{model}@{endpoint}" and VLMConfig lost its LM Studio
    # defaults, so the key moved on purpose from rec-bf54146413d69a17 to rec-876f6df9a5eaad7d.
    # Re-supplying the old defaults and the old identity string reproduces the old key exactly.
    (tmp_path / "p.jsonl").write_bytes(b'{"image_id": "a", "detections": []}\n')
    (tmp_path / "r.jsonl").write_bytes(b'{"image_id": "a", "scores": {}}\n')
    pipe = RoutingPipeline(config(vlm={"enabled": True}), PrecomputedDetector(tmp_path / "p.jsonl", "prim@1"),
                           RouteStore(":memory:"), relevance=PrecomputedRelevance(tmp_path / "r.jsonl", "rel@1"),
                           vlm=OpenAICompatibleVerifier("http://localhost:1234/v1", "qwen3-vl-4b-instruct"))
    run = pipe.run_info
    assert (run.primary_model, run.relevance_scorer, run.vlm_model) == (
        "prim@1", "rel@1", "openai-compatible:qwen3-vl-4b-instruct@localhost:1234/v1")
    ref = ImageRef(id="a", uri="a.png", width=320, height=240, sha256="0" * 64)
    assert pipe.record_key(ref) == "rec-876f6df9a5eaad7d"


def test_integration_packages_resolve_to_src_and_the_old_services_package_is_gone():
    # tests/ and tests/services/ are packages (tests/__init__.py), so pytest's default "prepend"
    # import mode imports them as tests.services; nothing may shadow the src packages.
    import importlib
    from pathlib import Path

    import service_detector_gdino.contracts as contracts

    for name in ("routing", "services_embedding", "service_detector_vlm", "service_detector_gdino"):
        pkg = Path(importlib.import_module(name).__file__).parent
        assert (pkg.name, pkg.parent.name) == (name, "src")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("services")
    assert contracts.SERVICE_SCHEMA_VERSION == "svc-1.0"
    assert __name__ == "tests.services.test_run_identity"


def test_service_contracts_import_no_routing_and_keep_the_strict_config():
    # The server needs only service_detector_gdino.*; contracts carries its own copy of the Strict base.
    import subprocess
    import sys

    from routing.schemas import Strict as RoutingStrict
    from service_detector_gdino.contracts import Strict as ContractStrict

    assert ContractStrict is not RoutingStrict and ContractStrict.model_config == RoutingStrict.model_config
    code = "import sys, service_detector_gdino.contracts; sys.exit(any(m.split('.')[0] == 'routing' for m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
