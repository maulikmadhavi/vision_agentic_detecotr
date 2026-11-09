"""Live check against the local Grounding DINO server (excluded by default).

Start the server first:  pixi run -m src/services/grounding_dino/pixi.toml serve
Then run:                pixi run test-gdino
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from routing.matching import iou
from services.http_detector import RemoteDetector
from routing.schemas import ClassSpec, TaskSpec

from .live import require_gdino

pytestmark = pytest.mark.gdino

REVISION = "a2bb814dd30d+bt0.35+tt0.25"
WEIGHTS = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"


def test_remote_grounding_dino_finds_a_blue_square():
    url = require_gdino()
    task = TaskSpec(task_id="live", prompt_set_version="p1", classes=(
        ClassSpec(class_id="blue_square", display_name="Blue square", positive_prompts=("a blue square",)),))
    det = RemoteDetector(base_url=url, task=task, model_id="grounding-dino-tiny", model_revision=REVISION,
                         weights_sha256=WEIGHTS, timeout_s=120)
    img = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(img).rectangle([100, 80, 199, 159], fill=(0, 0, 255))
    res = det.detect(img, "live", ("blue_square",))
    assert any(d.class_id == "blue_square" and iou(d.box_xyxy, (100, 80, 200, 160)) > 0.5 for d in res.detections)
    assert res.gpu_ms is not None and res.gpu_ms > 0
