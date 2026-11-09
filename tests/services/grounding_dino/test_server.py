"""Server tests with a fake model (no torch, no weights, no GPU).

Run in the service environment: pixi run -m src/services/grounding_dino/pixi.toml test
(skipped in the root environment, which has no fastapi).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import base64
import hashlib
import io

from fastapi.testclient import TestClient
from PIL import Image

from services.contracts import DetectClass, DetectRequest, DetectResponse

from services.grounding_dino import server


def _req(classes, image=None):
    image = image or Image.new("RGB", (64, 48), "white")
    buf = io.BytesIO()
    image.save(buf, "PNG")
    png = buf.getvalue()
    return DetectRequest(request_id="r1", image_png_b64=base64.b64encode(png).decode(),
                         image_sha256=hashlib.sha256(png).hexdigest(), width=image.width, height=image.height,
                         classes=tuple(DetectClass(class_id=c, prompts=p) for c, p in classes))


SQUARE = DetectClass(class_id="blue_square", prompts=("a blue square",))
CIRCLE = DetectClass(class_id="blue_circle", prompts=("a blue circle", "a round blue disc"))


def test_class_prompt_is_lowercase_and_period_separated():
    assert server.class_prompt(("A Blue Circle", "a round blue disc.")) == "a blue circle. a round blue disc."


def test_one_pass_per_class_attributes_boxes_by_pass_not_by_phrase():
    # Fake model output with the phrase problems that break phrase-to-class mapping:
    # overlapping words ("blue"), a merged span, an unmatched phrase. None of them lose or move a box.
    seen = []

    def predict(image, text):
        seen.append(text)
        if text.startswith("a blue square"):
            return [((1, 2, 10, 12), 0.8, "blue"), ((20, 2, 30, 12), 0.6, "a blue square a blue circle")], 3.0
        return [((5, 5, 9, 9), 0.5, "something unrelated")], 2.0

    boxes, gpu = server.detect_classes(Image.new("RGB", (64, 48)), (SQUARE, CIRCLE), predict)
    assert seen == ["a blue square.", "a blue circle. a round blue disc."]  # one pass each, only that class's prompts
    assert [(b.class_id, b.box_xyxy, b.score) for b in boxes] == [
        ("blue_square", (1, 2, 10, 12), 0.8), ("blue_square", (20, 2, 30, 12), 0.6),
        ("blue_circle", (5, 5, 9, 9), 0.5)]
    assert gpu == 5.0


def test_gpu_time_is_none_if_any_pass_did_not_measure():
    _, gpu = server.detect_classes(Image.new("RGB", (8, 8)), (SQUARE, CIRCLE),
                                   lambda i, t: ([], None if "circle" in t else 1.0))
    assert gpu is None


def _client(predict):
    return TestClient(server.create_app(predict, revision="rev+bt0.35+tt0.25", weights_sha256="a" * 64))


def test_detect_endpoint_returns_a_valid_svc_response():
    resp = _client(lambda i, t: ([((1, 1, 5, 5), 0.9, "x")], 1.5)).post(
        "/detect", json=_req([("blue_square", ("a blue square",))]).model_dump(mode="json"))
    out = DetectResponse.model_validate_json(resp.content)
    assert out.status == "ok" and out.request_id == "r1" and (out.width, out.height) == (64, 48)
    assert out.model_revision == "rev+bt0.35+tt0.25" and out.weights_sha256 == "a" * 64
    assert [(d.class_id, d.box_xyxy) for d in out.detections] == [("blue_square", (1, 1, 5, 5))] and out.gpu_ms == 1.5


def test_empty_result_is_ok_not_failed():
    out = DetectResponse.model_validate_json(_client(lambda i, t: ([], 1.0)).post(
        "/detect", json=_req([("blue_square", ("a blue square",))]).model_dump(mode="json")).content)
    assert out.status == "ok" and out.detections == ()


def test_model_error_and_bad_digest_return_failed_not_empty():
    def boom(image, text):
        raise RuntimeError("CUDA out of memory")

    body = _req([("blue_square", ("a blue square",))]).model_dump(mode="json")
    out = DetectResponse.model_validate_json(_client(boom).post("/detect", json=body).content)
    assert out.status == "failed" and "CUDA out of memory" in out.error and out.detections == ()
    body["image_sha256"] = "0" * 64
    out = DetectResponse.model_validate_json(_client(lambda i, t: ([], 1.0)).post("/detect", json=body).content)
    assert out.status == "failed" and "image_sha256" in out.error
