"""CLIP-style text-image relevance scorer (Stage A) via open_clip.

Optional dependency: ``pip install -e ".[clip]"``. torch / open_clip are imported
lazily so the core package (and the default test run) never needs them.

Score definition (``score_scale = "clip_prompt_softmax_v1"``)
------------------------------------------------------------
For each class, the image is compared against that class's positive prompts,
its negative (confusable) prompts and a fixed set of generic background
prompts. ``score = sum(softmax(logit_scale * cosine)[positives])``: the share of
prompt mass on the positives, in [0, 1].

This is still a **raw, uncalibrated** score (stored as ``raw_similarity``). It is
*not* a cosine similarity (raw CLIP cosines sit around 0.2-0.35), and it is not a
probability. Relevance bands in config must be declared for this scale
(``relevance.raw_scale``); the pipeline refuses a mismatch when this scorer is
used live. The guard cannot see scores exported to a JSONL file and replayed via
``PrecomputedRelevance``: set ``raw_scale`` and bands by hand in that case.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from .runtime import cuda_timer, file_sha256


SCORE_SCALE = "clip_prompt_softmax_v1"


BACKGROUND_PROMPTS: tuple[str, ...] = (
    "a photo of an empty scene",
    "a photo of something else",
    "a blurry photo",
    "a photo of a plain background",
)


class OpenClipRelevanceScorer:
    """Per-class prompt-contrast relevance from a local open_clip checkpoint.

    ``weights`` must be a local file (no network). ``artifact_digest`` is the
    SHA-256 of the weights, so the run key is bound to the actual model, not a
    caller-asserted name. Text embeddings are cached per prompt set.
    """

    score_scale = SCORE_SCALE

    def __init__(self, weights: str | Path, arch: str = "ViT-B-16-quickgelu", device: str | None = None,
                 background_prompts: tuple[str, ...] = BACKGROUND_PROMPTS):
        import open_clip
        import torch

        self._torch = torch
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"CLIP weights not found: {self.weights}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch
        self.background_prompts = tuple(background_prompts)
        self.artifact_digest = file_sha256(self.weights)
        bg = hashlib.sha256("\n".join(self.background_prompts).encode()).hexdigest()[:8]
        self.model_version = f"open_clip:{arch}@{self.artifact_digest[:12]}:{SCORE_SCALE}:bg-{bg}"
        model, _, self._preprocess = open_clip.create_model_and_transforms(arch, pretrained=str(self.weights))
        self._model = model.eval().to(self.device)
        self._tokenizer = open_clip.get_tokenizer(arch)
        self._text_cache: dict[tuple[str, ...], object] = {}
        self.last_gpu_ms: float | None = None
        self.last_wall_ms: float | None = None

    def _encode_text(self, prompts: tuple[str, ...]):
        if prompts not in self._text_cache:
            t = self._model.encode_text(self._tokenizer(list(prompts)).to(self.device))
            self._text_cache[prompts] = t / t.norm(dim=-1, keepdim=True)
        return self._text_cache[prompts]

    def score(self, image: Image.Image, image_id: str,
              prompts: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]) -> dict[str, float]:
        torch = self._torch
        t0 = time.perf_counter()
        out: dict[str, float] = {}
        with torch.no_grad(), cuda_timer(self.device) as gpu:
            x = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            img = self._model.encode_image(x)
            img = img / img.norm(dim=-1, keepdim=True)
            scale = float(self._model.logit_scale.exp())
            for class_id, (pos, neg) in sorted(prompts.items()):
                texts = tuple(pos) + tuple(neg) + self.background_prompts
                probs = (scale * img @ self._encode_text(texts).T).softmax(dim=-1)[0]
                out[class_id] = round(float(probs[: len(pos)].sum()), 6)
        self.last_gpu_ms = gpu.ms
        self.last_wall_ms = round((time.perf_counter() - t0) * 1000, 3)
        return out
