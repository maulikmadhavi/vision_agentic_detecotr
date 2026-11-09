"""Live check against the configured OpenAI-compatible VLM endpoint (excluded by default).

Endpoint/model: VAD_VLM_URL / VAD_VLM_MODEL (default: local LM Studio). Run with: pixi run test-vlm
"""

from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from routing.schemas import VerificationRequest

from .live import require_vlm

pytestmark = pytest.mark.vlm


def test_vlm_confirms_and_rejects():
    im = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(im).rectangle([100, 80, 200, 160], fill=(0, 0, 255))
    vlm = replace(require_vlm(), timeout_s=180)
    present = vlm.verify(im, VerificationRequest(id="r1", image_id="i", class_id="blue_square",
                                                 region_xyxy=(100, 80, 201, 161), purpose="confirm_candidate"),
                         "a solid blue square")
    absent = vlm.verify(im, VerificationRequest(id="r2", image_id="i", class_id="red_circle",
                                                purpose="suspected_missing_class"), "a red circle")
    assert present.verdict == "present", present
    assert absent.verdict == "absent", absent
