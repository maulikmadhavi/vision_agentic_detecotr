"""Live check against a local LM Studio server. Run with: pytest -m lmstudio"""

import pytest
from PIL import Image, ImageDraw

from vad.routing.adapters import LMStudioVLM
from vad.routing.schemas import VerificationRequest

pytestmark = pytest.mark.lmstudio


def test_qwen_vl_confirms_and_rejects():
    im = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(im).rectangle([100, 80, 200, 160], fill=(0, 0, 255))
    vlm = LMStudioVLM(timeout_s=180)
    present = vlm.verify(im, VerificationRequest(id="r1", image_id="i", class_id="blue_square",
                                                 region_xyxy=(100, 80, 201, 161), purpose="confirm_candidate"),
                         "a solid blue square")
    absent = vlm.verify(im, VerificationRequest(id="r2", image_id="i", class_id="red_circle",
                                                purpose="suspected_missing_class"), "a red circle")
    assert present.verdict == "present", present
    assert absent.verdict == "absent", absent
