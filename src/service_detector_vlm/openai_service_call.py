"""VLM verifier client for any OpenAI-compatible Chat Completions endpoint (verifier only; never authors boxes).

The endpoint must accept image data URIs (``image_url`` content parts) and
``response_format: {"type": "json_schema"}``. LM Studio is one such server.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field

import httpx
from PIL import Image

from routing.schemas import VerificationRequest, VerificationResult
from routing.verification_results import parse_vlm_answer, uncertain


_VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "class_id": {"type": "string"},
        "verdict": {"type": "string", "enum": ["present", "absent", "uncertain"]},
        "reason": {"type": "string"},
    },
    "required": ["class_id", "verdict", "reason"],
    "additionalProperties": False,
}


@dataclass
class OpenAICompatibleVerifier:
    """VLM verifier over an OpenAI-compatible endpoint; ``base_url`` and ``model`` come from config.

    Exactly one HTTP attempt per request, hard timeout, no retries (a rejected
    ``json_schema`` request is not retried in another format). Any transport error,
    malformed or contradictory answer becomes ``uncertain``. ``api_key_env`` names an
    environment variable holding a bearer token; the token never enters repr, traces or
    the recorded identity. The identity stays ``lmstudio:{model}`` until a deliberate migration.
    """

    base_url: str
    model: str
    timeout_s: float = 60
    max_tokens: int = 200
    crop_padding: float = 0.25
    api_key_env: str | None = None
    transport: httpx.BaseTransport | None = field(default=None, repr=False)
    _headers: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api_key_env:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise ValueError(f"vlm.api_key_env names {self.api_key_env!r}, but that environment variable is not set")
            self._headers = {"Authorization": f"Bearer {token}"}

    @property
    def model_version(self) -> str:
        return f"lmstudio:{self.model}"

    def _image_payload(self, image: Image.Image, region) -> str:
        if region is not None:
            x1, y1, x2, y2 = region
            px, py = (x2 - x1) * self.crop_padding, (y2 - y1) * self.crop_padding
            image = image.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                                min(image.width, int(x2 + px + 1)), min(image.height, int(y2 + py + 1))))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _prompt(self, request: VerificationRequest, class_desc: str) -> str:
        if request.purpose == "confirm_candidate":
            q = f"Does this image crop contain a '{class_desc}'?"
        else:
            q = f"Is there any '{class_desc}' visible anywhere in this image?"
        return (f"{q} Answer only with JSON: class_id must be exactly \"{request.class_id}\", "
                "verdict is one of present|absent|uncertain, reason is a short phrase. "
                "Say uncertain if you cannot tell.")

    def verify(self, image: Image.Image, request: VerificationRequest, class_desc: str) -> VerificationResult:
        body = {
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self._prompt(request, class_desc)},
                {"type": "image_url", "image_url": {"url": self._image_payload(image, request.region_xyxy)}},
            ]}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "verification", "strict": True, "schema": _VLM_SCHEMA}},
        }
        try:
            with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                resp = client.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException as e:
            return uncertain(request, "vlm_timeout", self.model_version, str(e))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            return uncertain(request, "vlm_unavailable", self.model_version, f"{type(e).__name__}: {e}")
        return parse_vlm_answer(content, request, self.model_version)
