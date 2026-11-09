"""WeMM-Embedding MCP client: text-image relevance via ``compute_similarity`` (cosine)."""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass, field
from typing import ClassVar

import httpx
from PIL import Image


WEMM_TEXT_SCALE = "wemm_text_cosine_v1"


class RemoteScorerError(RuntimeError):
    """The remote relevance scorer did not return a verified score; never treated as a score of 0."""


@dataclass
class WeMMMCPRelevanceScorer:
    """Text-image relevance from a WeMM-Embedding MCP server (``compute_similarity``, cosine).

    Implements the existing ``RelevanceScorer`` protocol. One JSON-RPC POST per image
    with all classes' positive prompts; per-class raw score = max cosine over that
    class's prompts (scale ``wemm_text_cosine_v1``, not a probability, not comparable
    with CLIP). Images are downscaled so the longest side is at most ``max_side``;
    that setting is part of ``model_version``. Retries once on a network error only.
    ``isError``, JSON-RPC errors, malformed results and ``meta`` mismatches raise
    :class:`RemoteScorerError`; they never become a score. Without a server-reported
    ``build`` the version is ``@unverified`` (board assumption 6).
    """

    url: str
    model: str = "tencent/WeMM-Embedding-4B"
    dim: int = 2560
    build: str | None = None
    max_side: int = 448
    timeout_s: float = 120
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    score_scale: ClassVar[str] = WEMM_TEXT_SCALE
    artifact_digest: ClassVar[None] = None

    @property
    def model_version(self) -> str:
        return f"wemm:{self.model}/d{self.dim}@{self.build or 'unverified'}+max{self.max_side}"

    def _data_uri(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        if max(img.size) > self.max_side:
            scale = self.max_side / max(img.size)
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _call(self, tool: str, arguments: dict) -> dict:
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
        headers = {"Accept": "application/json, text/event-stream"}
        resp = None
        for attempt in (1, 2):  # one retry, only for network errors
            try:
                with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                    resp = client.post(self.url, json=body, headers=headers)
                break
            except httpx.TransportError as e:
                if attempt == 2:
                    raise RemoteScorerError(f"network: {type(e).__name__}: {e}"[:300]) from e
        try:
            resp.raise_for_status()
            text = resp.text
            if "text/event-stream" in resp.headers.get("content-type", ""):
                text = [ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")][-1]
            msg = json.loads(text)
            if "error" in msg:
                raise RemoteScorerError(f"jsonrpc error: {msg['error']}"[:300])
            result = msg["result"]
            if result.get("isError"):
                raise RemoteScorerError(f"isError: {result.get('content')}"[:300])
            payload = json.loads(result["content"][0]["text"])
            meta = payload["meta"]
        except RemoteScorerError:
            raise
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            raise RemoteScorerError(f"{type(e).__name__}: {e}"[:300]) from e
        expected = {"backend": "wemm", "model": self.model, "dim": self.dim}
        if self.build is not None:
            expected["build"] = self.build
        if bad := {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}:
            raise RemoteScorerError(f"meta mismatch: {bad}")
        return payload

    def score(self, image: Image.Image, image_id: str,
              prompts: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]) -> dict[str, float]:
        texts, owners = [], []
        for class_id, (positives, _negatives) in prompts.items():
            texts += positives
            owners += [class_id] * len(positives)
        payload = self._call("compute_similarity", {"left": [self._data_uri(image)], "right": texts,
                                                    "mode": "matrix", "metric": "cosine", "top_k": len(texts)})
        try:
            (row,) = payload["results"]  # exactly one image was sent, so exactly one result row
            pairs = [(int(m["index"]), float(m["score"])) for m in row["matches"]]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise RemoteScorerError(f"malformed matches: {type(e).__name__}") from e
        by_index = dict(pairs)
        if (len(pairs) != len(by_index) or set(by_index) != set(range(len(texts)))
                or not all(math.isfinite(v) for v in by_index.values())):
            raise RemoteScorerError(f"expected one finite score per index 0..{len(texts) - 1}")
        out: dict[str, float] = {}
        for i, class_id in enumerate(owners):
            out[class_id] = max(out.get(class_id, -math.inf), by_index[i])
        return out
