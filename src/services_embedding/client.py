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
    class's prompts (scale ``wemm_text_cosine_v1``, not a probability).
    Images are downscaled so the longest side is at most ``max_side``;
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
        payload = self._decode(self._post(body))
        self._check_meta(payload["meta"])
        return payload

    def _post(self, body: dict) -> httpx.Response:
        """POST the JSON-RPC body; retry once, and only on a transport (network) error."""
        headers = {"Accept": "application/json, text/event-stream"}
        for attempt in (1, 2):
            try:
                with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                    return client.post(self.url, json=body, headers=headers)
            except httpx.TransportError as e:
                if attempt == 2:
                    raise RemoteScorerError(f"network: {type(e).__name__}: {e}"[:300]) from e
        raise AssertionError("unreachable")

    @staticmethod
    def _decode(resp: httpx.Response) -> dict:
        """The tool payload (with a ``meta`` key) from a JSON or event-stream reply; any failure raises."""
        try:
            resp.raise_for_status()
            text = resp.text
            if "text/event-stream" in resp.headers.get("content-type", ""):
                text = [ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")][-1]
            msg = json.loads(text)
            if "error" in msg:
                raise RemoteScorerError(f"jsonrpc error: {msg['error']}"[:300])
            result = msg["result"]
            if result.get("isError"):  # a tool error is never a score
                raise RemoteScorerError(f"isError: {result.get('content')}"[:300])
            payload = json.loads(result["content"][0]["text"])
            payload["meta"]  # a reply without metadata fails here, as a decoding error
        except RemoteScorerError:
            raise
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            raise RemoteScorerError(f"{type(e).__name__}: {e}"[:300]) from e
        return payload

    def _check_meta(self, meta: dict) -> None:
        """Every reply must name the expected backend, model and dim (and build, once one is pinned)."""
        expected = {"backend": "wemm", "model": self.model, "dim": self.dim}
        if self.build is not None:
            expected["build"] = self.build
        if bad := {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}:
            raise RemoteScorerError(f"meta mismatch: {bad}")

    def score(self, image: Image.Image, image_id: str,
              prompts: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]) -> dict[str, float]:
        texts, owners = [], []
        for class_id, (positives, _negatives) in prompts.items():
            texts += positives
            owners += [class_id] * len(positives)
        payload = self._call("compute_similarity", {"left": [self._data_uri(image)], "right": texts,
                                                    "mode": "matrix", "metric": "cosine", "top_k": len(texts)})
        # Exactly one finite score per prompt index: no duplicates, no extras, none missing.
        # A malformed match anywhere is reported as such, before any index/score problem.
        by_index: dict[int, float] = {}
        valid = True
        try:
            (row,) = payload["results"]  # exactly one image was sent, so exactly one result row
            for match in row["matches"]:
                index, value = int(match["index"]), float(match["score"])
                valid = valid and index not in by_index and 0 <= index < len(texts) and math.isfinite(value)
                by_index[index] = value
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise RemoteScorerError(f"malformed matches: {type(e).__name__}") from e
        if not valid or len(by_index) != len(texts):
            raise RemoteScorerError(f"expected one finite score per index 0..{len(texts) - 1}")
        out: dict[str, float] = {}
        for index, class_id in enumerate(owners):
            out[class_id] = max(out.get(class_id, -math.inf), by_index[index])
        return out
