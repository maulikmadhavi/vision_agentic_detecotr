"""Live check of the open_clip relevance scorer. Run with: pytest -m clip

Uses VAD_CLIP_WEIGHTS if set, else the cached timm/vit_base_patch16_clip_224.openai
weights in the Hugging Face cache. Never downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.clip


def _weights() -> Path:
    env = os.environ.get("VAD_CLIP_WEIGHTS")
    if env:
        return Path(env)
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--timm--vit_base_patch16_clip_224.openai"
    found = sorted(hub.glob("snapshots/*/open_clip_model.safetensors"))
    if not found:
        pytest.skip("no local CLIP weights")
    return found[0]


def test_open_clip_scorer_separates_present_from_absent(tmp_path):
    pytest.importorskip("open_clip")
    from services.open_clip import OpenClipRelevanceScorer, SCORE_SCALE
    from routing.image_inputs import ManifestEntry
    from routing.pipeline import RoutingPipeline
    from routing.store import RouteStore

    from ..routing.helpers import ScriptedDetector, config

    scorer = OpenClipRelevanceScorer(_weights())
    im = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(im).rectangle([100, 80, 200, 160], fill=(0, 0, 255))
    prompts = {"blue_square": (("a blue square",), ("a red circle",)),
               "dog": (("a photo of a dog",), ())}
    s = scorer.score(im, "i", prompts)
    assert 0.0 <= s["dog"] <= 1.0 and 0.0 <= s["blue_square"] <= 1.0
    assert s["blue_square"] > 0.5 > s["dog"], s
    assert scorer.artifact_digest and SCORE_SCALE in scorer.model_version
    if scorer.device.startswith("cuda"):
        assert scorer.last_gpu_ms is not None and scorer.last_gpu_ms > 0

    # wired into the pipeline: only with a matching raw_scale
    path = tmp_path / "a.png"
    im.save(path)
    with pytest.raises(ValueError, match="raw_scale"):
        RoutingPipeline(config(), ScriptedDetector({"a": []}), RouteStore(":memory:"), relevance=scorer)
    cfg = config(relevance={"raw_scale": SCORE_SCALE, "raw_high": 0.6, "raw_low": 0.2})
    rec = RoutingPipeline(cfg, ScriptedDetector({"a": []}), RouteStore(":memory:"),
                          relevance=scorer).process(ManifestEntry(str(path), image_id="a"))
    assert rec.run.relevance_artifact_digest == scorer.artifact_digest
    assert set(rec.stage_a.per_class) == set(cfg.task.class_ids)
