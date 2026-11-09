"""Live check against a local LM Studio server (excluded by default). Run with: pixi run test-lmstudio"""

from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from routing.schemas import VerificationRequest

from .live import require_lmstudio

pytestmark = pytest.mark.lmstudio


def test_qwen_vl_confirms_and_rejects():
    im = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(im).rectangle([100, 80, 200, 160], fill=(0, 0, 255))
    base = require_lmstudio()
    vlm = replace(base, timeout_s=180)
    present = vlm.verify(im, VerificationRequest(id="r1", image_id="i", class_id="blue_square",
                                                 region_xyxy=(100, 80, 201, 161), purpose="confirm_candidate"),
                         "a solid blue square")
    absent = vlm.verify(im, VerificationRequest(id="r2", image_id="i", class_id="red_circle",
                                                purpose="suspected_missing_class"), "a red circle")
    assert present.verdict == "present", present
    assert absent.verdict == "absent", absent
