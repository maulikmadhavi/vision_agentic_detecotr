"""Smoke test of the real model through the HTTP app; prints measurements as JSON.

Cases: a blue square on white (expect a blue_square box, no dog box) and a blank image
(expect status ok with no boxes). Reports per-request latency, server gpu_ms, peak
torch VRAM in this process, and total GPU memory in use (nvidia-smi, all processes).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import time

import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import server
from vad.routing.matching import iou
from vad.routing.service_contracts import DetectClass, DetectRequest, DetectResponse

SQUARE = (100, 80, 200, 160)
CLASSES = (DetectClass(class_id="blue_square", prompts=("a blue square",)),
           DetectClass(class_id="dog", prompts=("a dog",)))


def request(image: Image.Image, rid: str) -> dict:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    png = buf.getvalue()
    return DetectRequest(request_id=rid, image_png_b64=base64.b64encode(png).decode(),
                         image_sha256=hashlib.sha256(png).hexdigest(), width=image.width, height=image.height,
                         classes=CLASSES).model_dump(mode="json")


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def main() -> None:
    t0 = time.perf_counter()
    predict, digest = server.load_hf()
    load_s = time.perf_counter() - t0
    client = TestClient(server.create_app(predict, server.REVISION, digest))

    square = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(square).rectangle([SQUARE[0], SQUARE[1], SQUARE[2] - 1, SQUARE[3] - 1], fill=(0, 0, 255))
    blank = Image.new("RGB", (320, 240), "white")

    client.post("/detect", json=request(square, "warmup"))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = {}
    for name, img in (("square", square), ("blank", blank)):
        t = time.perf_counter()
        out = DetectResponse.model_validate_json(client.post("/detect", json=request(img, name)).content)
        results[name] = {"status": out.status, "latency_ms": round((time.perf_counter() - t) * 1000, 1),
                         "gpu_ms": out.gpu_ms, "boxes": [(d.class_id, [round(v, 1) for v in d.box_xyxy],
                                                          round(d.score, 3)) for d in out.detections]}
    sq = results["square"]["boxes"]
    # Plumbing checks decide the exit code. Model false positives are measured, not asserted:
    # on synthetic images Grounding DINO boxes something for every prompt (see README).
    checks = {
        "square_found": any(c == "blue_square" and iou(tuple(b), SQUARE) > 0.5 for c, b, _ in sq),
        "statuses_ok": all(r["status"] == "ok" for r in results.values()),
        "weights_digest_pinned": digest == server.WEIGHTS_SHA256,
    }
    observations = {
        "dog_false_positive_on_square": [s for c, _, s in sq if c == "dog"],
        "boxes_on_blank_image": results["blank"]["boxes"],
    }
    report = {"revision": server.REVISION, "device": "cuda" if torch.cuda.is_available() else "cpu",
              "load_s": round(load_s, 1), "results": results, "checks": checks, "observations": observations,
              "peak_torch_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
              "gpu_used_total_mib": gpu_used_mib()}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
