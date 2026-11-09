"""Grounding DINO detector server speaking svc-1.0 (see src/services/contracts.py).

One forward pass per requested class, prompted only with that class's prompts, so
every returned box belongs to that class. This avoids mapping Grounding DINO's
text spans back to classes, which can merge, overlap or match nothing and would
silently lose or misassign boxes. Cost: N passes for N classes.

    pixi run -m src/services/grounding_dino/pixi.toml fetch   # one-time, pinned download
    pixi run -m src/services/grounding_dino/pixi.toml serve   # http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import threading
from collections.abc import Callable, Sequence

from fastapi import FastAPI
from PIL import Image

from services.contracts import DetectClass, DetectRequest, DetectResponse, ServedBox

MODEL_ID = "grounding-dino-tiny"
HF_REPO = "IDEA-Research/grounding-dino-tiny"
HF_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
WEIGHTS_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
HF_FILES = ["model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json", "vocab.txt",
            "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]
BOX_THRESHOLD, TEXT_THRESHOLD = 0.35, 0.25
REVISION = f"{HF_REVISION[:12]}+bt{BOX_THRESHOLD}+tt{TEXT_THRESHOLD}"  # thresholds change the run key

# predict(image, text) -> ([(box_xyxy, score, phrase), ...], gpu_ms or None)
Predict = Callable[[Image.Image, str], tuple[list[tuple[Sequence[float], float, str]], float | None]]


def class_prompt(prompts: Sequence[str]) -> str:
    return " ".join(p.strip().lower().rstrip(".").strip() + "." for p in prompts)


def detect_classes(image: Image.Image, classes: Sequence[DetectClass], predict: Predict):
    """Run one pass per class; box -> class comes from the pass, never from the returned phrase."""
    boxes: list[ServedBox] = []
    gpu: float | None = 0.0
    for c in classes:
        found, ms = predict(image, class_prompt(c.prompts))
        gpu = None if (ms is None or gpu is None) else gpu + ms
        boxes += [ServedBox(class_id=c.class_id, box_xyxy=tuple(float(v) for v in b), score=float(s))
                  for b, s, _phrase in found]
    return boxes, gpu


def create_app(predict: Predict, revision: str, weights_sha256: str | None, model_id: str = MODEL_ID) -> FastAPI:
    app = FastAPI(title="grounding-dino svc-1.0")
    lock = threading.Lock()  # one request at a time on a 12 GB GPU

    @app.post("/detect")
    def detect(req: DetectRequest) -> DetectResponse:
        base = dict(schema_version="svc-1.0", request_id=req.request_id, model_id=model_id, model_revision=revision,
                    weights_sha256=weights_sha256, coord_system="xyxy_pixel_submitted",
                    width=req.width, height=req.height)
        try:
            png = base64.b64decode(req.image_png_b64, validate=True)
            if hashlib.sha256(png).hexdigest() != req.image_sha256:
                raise ValueError("image_sha256 does not match the image bytes")
            image = Image.open(io.BytesIO(png)).convert("RGB")
            if image.size != (req.width, req.height):
                raise ValueError(f"image is {image.size}, request says {(req.width, req.height)}")
            with lock:
                boxes, gpu = detect_classes(image, req.classes, predict)
            return DetectResponse(**base, status="ok", detections=tuple(boxes), gpu_ms=gpu)
        except Exception as e:  # any failure is reported, never returned as an empty result
            return DetectResponse(**base, status="failed", error=f"{type(e).__name__}: {e}"[:300])

    return app


def fetch() -> None:
    from huggingface_hub import hf_hub_download

    for f in HF_FILES:
        print(hf_hub_download(HF_REPO, f, revision=HF_REVISION))


def load_hf(device: str | None = None) -> tuple[Predict, str]:
    """Load the pinned checkpoint from the local cache only; refuse to start on a digest mismatch."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from services.runtime import cuda_timer, file_sha256

    weights = hf_hub_download(HF_REPO, "model.safetensors", revision=HF_REVISION, local_files_only=True)
    digest = file_sha256(weights)
    if digest != WEIGHTS_SHA256:
        raise SystemExit(f"weights digest {digest} != pinned {WEIGHTS_SHA256}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(revision=HF_REVISION, local_files_only=True)
    processor = AutoProcessor.from_pretrained(HF_REPO, **kw)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(HF_REPO, use_safetensors=True, **kw).to(device).eval()

    def predict(image: Image.Image, text: str):
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad(), cuda_timer(device) as t:
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]])[0]
        phrases = res.get("text_labels") or res.get("labels") or [""] * len(res["boxes"])
        return [(b, s, str(p)) for b, s, p in zip(res["boxes"].tolist(), res["scores"].tolist(), phrases)], t.ms

    return predict, digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the pinned checkpoint files, then exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    if args.fetch:
        fetch()
        return
    import uvicorn

    predict, digest = load_hf()
    uvicorn.run(create_app(predict, REVISION, digest), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
