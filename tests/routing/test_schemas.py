"""Contract validation and JSON Schema drift."""

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from vad.routing.schemas import (
    EXPORTED_MODELS, ClassSpec, Detection, Difficulty, ImageRef, RunInfo, TaskSpec, export_json_schemas,
)
from vad.routing.task_config import load_config

from .helpers import CLASS_A, config

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
REF = ImageRef(id="i", uri="u", width=100, height=50, sha256="0" * 64)


def det(**kw):
    base = dict(id="d1", class_id=CLASS_A, box_xyxy=(1, 1, 10, 10), raw_confidence=0.5,
                source="primary", model_version="m@1")
    return Detection(**{**base, **kw})


@pytest.mark.parametrize("kw", [
    {"box_xyxy": (10, 1, 5, 10)},
    {"box_xyxy": (-1, 1, 5, 10)},
    {"box_xyxy": (1, 1, 1, 10)},
    {"raw_confidence": math.nan},
    {"raw_confidence": math.inf},
    {"raw_confidence": 1.5},
    {"model_version": ""},
    {"coord_system": "cxcywh_normalized"},
    {"extra_field": 1},
])
def test_detection_rejects_invalid(kw):
    with pytest.raises(ValidationError):
        det(**kw)


def test_detection_context_checks():
    with pytest.raises(ValueError, match="unknown class_id"):
        det(class_id="not_registered").check_in(REF, (CLASS_A,))
    with pytest.raises(ValueError, match="exceeds image"):
        det(box_xyxy=(1, 1, 10, 60)).check_in(REF, (CLASS_A,))
    det().check_in(REF, (CLASS_A,))


def test_registry_and_config_validation():
    spec = ClassSpec(class_id="x", display_name="x", positive_prompts=("x",))
    with pytest.raises(ValidationError):
        TaskSpec(task_id="t", prompt_set_version="p", classes=(spec, spec))
    with pytest.raises(ValidationError):
        RunInfo(config_hash="abcdefgh", policy_version="v", primary_model="")
    with pytest.raises(ValidationError):
        config(matching={"class_mapping": {"foo": "unregistered"}})
    with pytest.raises(ValidationError):
        config(difficulty={"medium": 0.7, "hard": 0.6})
    with pytest.raises(ValidationError):
        Difficulty(value=0.4, status="unknown", bucket="unknown")


def test_example_config_loads_disabled():
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs/routing/example.yaml")
    assert cfg.routing_enabled is False and cfg.vlm.enabled is False


def test_config_hash_changes_with_thresholds():
    assert config().config_hash() != config(relevance={"raw_high": 0.6, "raw_low": 0.3}).config_hash()
    assert config().config_hash() == config().config_hash()


def test_committed_json_schemas_match_models(tmp_path):
    export_json_schemas(str(tmp_path))
    for m in EXPORTED_MODELS:
        name = f"{m.__name__}.schema.json"
        committed = SCHEMA_DIR / name
        assert committed.exists(), f"run `vad-route export-schema` to create {name}"
        assert json.loads(committed.read_text()) == json.loads((tmp_path / name).read_text()), \
            f"{name} drifted; re-export with `vad-route export-schema`"
