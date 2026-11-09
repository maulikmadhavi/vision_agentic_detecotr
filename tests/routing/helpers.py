"""Test doubles and a tiny deterministic fixture generator."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vad.routing.adapters import DetectorResult, RawDetection, uncertain
from vad.routing.schemas import ClassSpec, TaskSpec, VerificationResult
from vad.routing.task_config import RoutingConfig

# Deliberately non-COCO class ids.
CLASS_A = "blue_widget_x9"
CLASS_B = "zz-custom.class_7"
COLORS = {CLASS_A: (0, 0, 255), CLASS_B: (255, 0, 0)}


def task() -> TaskSpec:
    return TaskSpec(task_id="t-test", prompt_set_version="p1", classes=(
        ClassSpec(class_id=CLASS_A, display_name="Blue widget X9", positive_prompts=("a blue X9 widget",),
                  negative_prompts=("a blue bottle cap",)),
        ClassSpec(class_id=CLASS_B, display_name="Custom thing 7", positive_prompts=("a red custom thing",)),
    ))


def config(**overrides) -> RoutingConfig:
    base = {"routing_enabled": True, "task": task().model_dump()}
    for k, v in overrides.items():
        base[k] = v
    return RoutingConfig.model_validate(base)


def draw(path: Path, size: tuple[int, int], rects: Sequence[tuple[str, tuple[int, int, int, int]]]) -> Path:
    """Draw filled rectangles; rect boxes are exclusive xyxy pixel bounds."""
    im = Image.new("RGB", size, (200, 200, 200))
    d = ImageDraw.Draw(im)
    for cls, (x1, y1, x2, y2) in rects:
        d.rectangle([x1, y1, x2 - 1, y2 - 1], fill=COLORS[cls])
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[splits + 1]])
    ends = np.concatenate([idx[splits], [idx[-1]]]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


class BlobDetector:
    """Crop-capable fake detector: finds solid colour rectangles by reading pixels.

    ``min_side`` lets tests model a detector that misses small objects on the
    full image but finds them on SAHI tiles (objects look larger relative to the input).
    """

    supports_crops = True

    def __init__(self, model_version: str = "blob@1", min_frac: float = 0.0, confidence: float = 0.9,
                 gpu_ms: float | None = 5.0, label_map: dict | None = None):
        self.model_version, self.min_frac, self.confidence, self.gpu_ms = model_version, min_frac, confidence, gpu_ms
        self.label_map = label_map or {}
        self.calls = 0

    def detect(self, image, image_id, class_ids):
        self.calls += 1
        arr = np.asarray(image.convert("RGB"))
        out = []
        for cls, rgb in COLORS.items():
            mask = np.all(arr == rgb, axis=-1)
            for r0, r1 in _runs(mask.any(axis=1)):
                for c0, c1 in _runs(mask[r0:r1].any(axis=0)):
                    rows = _runs(mask[r0:r1, c0:c1].any(axis=1))
                    y0, y1 = r0 + rows[0][0], r0 + rows[-1][1]
                    if (c1 - c0) * (y1 - y0) < self.min_frac * image.width * image.height:
                        continue
                    out.append(RawDetection(self.label_map.get(cls, cls), (c0, y0, c1, y1), self.confidence))
        return DetectorResult(out, self.gpu_ms)


class ScriptedDetector:
    """Returns fixed raw detections per image id; counts calls."""

    supports_crops = False

    def __init__(self, preds: dict[str, list[RawDetection]], model_version: str = "scripted@1", gpu_ms=None):
        self.preds, self.model_version, self.gpu_ms, self.calls = preds, model_version, gpu_ms, 0

    def detect(self, image, image_id, class_ids):
        self.calls += 1
        return DetectorResult(list(self.preds.get(image_id, [])), self.gpu_ms)


class ScriptedRelevance:
    def __init__(self, scores: dict[str, dict[str, float]], model_version: str = "rel@1"):
        self.scores, self.model_version = scores, model_version

    def score(self, image, image_id, prompts):
        return dict(self.scores.get(image_id, {}))


class FakeVLM:
    """Answers from a {class_id: verdict} table; records every request."""

    def __init__(self, verdicts: dict[str, str] | None = None, default: str = "present", raise_exc=None):
        self.verdicts, self.default, self.raise_exc = verdicts or {}, default, raise_exc
        self.requests = []
        self.model_version = "fakevlm@1"

    def verify(self, image, request, class_desc):
        self.requests.append((request, class_desc))
        if self.raise_exc is not None:
            raise self.raise_exc
        v = self.verdicts.get(request.class_id, self.default)
        if v == "uncertain":
            return uncertain(request, "vlm_uncertain", self.model_version)
        return VerificationResult(request_id=request.id, class_id=request.class_id, verdict=v,
                                  candidate_ids=request.candidate_ids, region_xyxy=request.region_xyxy,
                                  reason_code=f"vlm_{v}", model_version=self.model_version)


def make_fixture(root: Path) -> dict:
    """Tiny reproducible fixture: 3 valid images, 1 corrupt file, manifest, predictions, relevance."""
    root.mkdir(parents=True, exist_ok=True)
    draw(root / "img_easy.png", (320, 240), [(CLASS_A, (60, 60, 160, 160))])
    draw(root / "img_zero.png", (320, 240), [])
    draw(root / "img_small.png", (640, 480), [(CLASS_B, (300, 200, 308, 208))])
    (root / "img_corrupt.png").write_bytes(b"not an image")
    manifest = [
        {"uri": "img_easy.png", "image_id": "easy", "group_id": "g1"},
        {"uri": "img_zero.png", "image_id": "zero", "group_id": "g2"},
        {"uri": "img_small.png", "image_id": "small", "group_id": "g3"},
        {"uri": "img_corrupt.png", "image_id": "corrupt", "group_id": "g4"},
    ]
    preds = [
        {"image_id": "easy", "detections": [{"class_id": CLASS_A, "box_xyxy": [60, 60, 160, 160], "confidence": 0.95}]},
        {"image_id": "zero", "detections": []},
        {"image_id": "small", "detections": [{"class_id": CLASS_B, "box_xyxy": [300, 200, 308, 208], "confidence": 0.4}]},
    ]
    rel = [
        {"image_id": "easy", "scores": {CLASS_A: 0.8, CLASS_B: 0.1}},
        {"image_id": "zero", "scores": {CLASS_A: 0.7, CLASS_B: 0.2}},
        {"image_id": "small", "scores": {CLASS_A: 0.1, CLASS_B: 0.6}},
    ]
    for name, rows in (("manifest.jsonl", manifest), ("primary_preds.jsonl", preds), ("relevance.jsonl", rel)):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg_path = root / "config.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(config(routing_enabled=False).model_dump(mode="json")), encoding="utf-8")
    return {"root": root, "manifest": root / "manifest.jsonl", "preds": root / "primary_preds.jsonl",
            "relevance": root / "relevance.jsonl", "config": cfg_path}
