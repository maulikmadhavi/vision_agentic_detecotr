"""Live check against the local Grounding DINO server (excluded by default).

Start the server first:  pixi run -m src/service_detector_gdino/pixi.toml serve
Then run:                pixi run test-gdino
Model identity: VAD_GDINO_URL / VAD_GDINO_MODEL_ID / VAD_GDINO_REVISION / VAD_GDINO_WEIGHTS_SHA256.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from routing.matching import iou
from routing.schemas import ClassSpec, TaskSpec

from .live import gdino_detector, require_gdino

pytestmark = pytest.mark.gdino


def test_remote_grounding_dino_finds_a_blue_square():
    require_gdino()
    task = TaskSpec(task_id="live", prompt_set_version="p1", classes=(
        ClassSpec(class_id="blue_square", display_name="Blue square", positive_prompts=("a blue square",)),))
    det = gdino_detector(task)
    img = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(img).rectangle([100, 80, 199, 159], fill=(0, 0, 255))
    res = det.detect(img, "live", ("blue_square",))
    assert any(d.class_id == "blue_square" and iou(d.box_xyxy, (100, 80, 200, 160)) > 0.5 for d in res.detections)
    assert res.gpu_ms is not None and res.gpu_ms > 0
