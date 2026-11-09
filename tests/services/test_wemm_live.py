"""Live check against the WeMM embedding server (excluded by default; synthetic image only).

The endpoint comes only from the environment:  VAD_WEMM_URL=<url> pixi run test-wemm
"""

from __future__ import annotations

import os

import pytest
from PIL import Image, ImageDraw

from services.remote_adapters import WeMMMCPRelevanceScorer

pytestmark = pytest.mark.wemm


def test_wemm_scores_a_matching_prompt_above_an_unrelated_one():
    url = os.environ.get("VAD_WEMM_URL")
    if not url:
        pytest.skip("VAD_WEMM_URL is not set")
    img = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(img).rectangle([200, 160, 439, 319], fill=(0, 0, 255))
    scores = WeMMMCPRelevanceScorer(url=url).score(
        img, "live", {"blue_square": (("a blue square",), ()), "dog": (("a photo of a dog",), ())})
    assert set(scores) == {"blue_square", "dog"}
    assert all(-1.0 <= v <= 1.0 for v in scores.values())
    assert scores["blue_square"] > scores["dog"], scores
