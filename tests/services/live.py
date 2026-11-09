"""Live-test configuration and readiness probes.

Endpoints and models come from the environment (defaults: this machine's local setup):

  VAD_GDINO_URL, VAD_GDINO_MODEL_ID, VAD_GDINO_REVISION, VAD_GDINO_WEIGHTS_SHA256
  VAD_WEMM_URL                      required for WeMM tests; never printed
  VAD_VLM_URL, VAD_VLM_MODEL        any OpenAI-compatible endpoint (default: local LM Studio)
  VAD_VLM_API_KEY_ENV               optional: NAME of the variable holding a bearer token

An unreachable service is skipped, or fails when VAD_LIVE_REQUIRED=1 (release checks must not
silently skip). A reachable service must answer one real synthetic inference, else the test fails.
A skip is not a pass: run the live tasks with ``-rs`` and record the skip reasons.
"""

from __future__ import annotations

import functools
import os
import socket
from urllib.parse import urlsplit

import pytest
from PIL import Image, ImageDraw

from routing.schemas import ClassSpec, TaskSpec, VerificationRequest
from routing.task_config import VLMConfig
from service_detector_gdino.client import RemoteDetector, RemoteDetectorError
from service_detector_vlm.openai_service_call import OpenAICompatibleVerifier
from services_embedding.client import RemoteScorerError, WeMMMCPRelevanceScorer

GDINO_URL = os.environ.get("VAD_GDINO_URL", "http://127.0.0.1:8001")
GDINO_MODEL_ID = os.environ.get("VAD_GDINO_MODEL_ID", "grounding-dino-tiny")
GDINO_REVISION = os.environ.get("VAD_GDINO_REVISION", "a2bb814dd30d+bt0.35+tt0.25")
GDINO_WEIGHTS = os.environ.get("VAD_GDINO_WEIGHTS_SHA256",
                               "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3")
VLM_URL = os.environ.get("VAD_VLM_URL", VLMConfig().base_url)
VLM_MODEL = os.environ.get("VAD_VLM_MODEL", VLMConfig().model)
VLM_API_KEY_ENV = os.environ.get("VAD_VLM_API_KEY_ENV") or None
_BROKEN_VLM = {"vlm_unavailable", "vlm_timeout", "vlm_malformed", "vlm_contradictory"}


def _unavailable(reason: str) -> None:
    """Service absent: skip, or fail in required mode."""
    if os.environ.get("VAD_LIVE_REQUIRED") == "1":
        pytest.fail(f"{reason} (VAD_LIVE_REQUIRED=1)")
    pytest.skip(reason)


def _port_open(url: str) -> bool:
    u = urlsplit(url)
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        socket.create_connection((u.hostname, port), timeout=3).close()
        return True
    except OSError:
        return False


def _blue_square() -> Image.Image:
    im = Image.new("RGB", (96, 96), "white")
    ImageDraw.Draw(im).rectangle([24, 24, 71, 71], fill=(0, 0, 255))
    return im


def gdino_detector(task: TaskSpec) -> RemoteDetector:
    return RemoteDetector(base_url=GDINO_URL, task=task, model_id=GDINO_MODEL_ID, model_revision=GDINO_REVISION,
                          weights_sha256=GDINO_WEIGHTS, timeout_s=120)


def vlm_verifier(**kw) -> OpenAICompatibleVerifier:
    return OpenAICompatibleVerifier(VLM_URL, VLM_MODEL, api_key_env=VLM_API_KEY_ENV, **kw)


@functools.cache
def _gdino_ready() -> str | None:
    task = TaskSpec(task_id="probe", prompt_set_version="p", classes=(
        ClassSpec(class_id="blue_square", display_name="Blue square", positive_prompts=("a blue square",)),))
    try:
        gdino_detector(task).detect(_blue_square(), "probe", ("blue_square",))
        return None
    except RemoteDetectorError as e:
        return str(e)[:200]


@functools.cache
def _wemm_ready(url: str) -> str | None:
    try:
        WeMMMCPRelevanceScorer(url=url).score(_blue_square(), "probe", {"blue_square": (("a blue square",), ())})
        return None
    except RemoteScorerError as e:
        return str(e).split(":")[0]  # error kind only: messages can contain the endpoint


@functools.cache
def _vlm_ready() -> str | None:
    req = VerificationRequest(id="probe", image_id="probe", class_id="blue_square", purpose="suspected_missing_class")
    res = vlm_verifier(timeout_s=180).verify(_blue_square(), req, "a blue square")
    return res.reason_code if res.reason_code in _BROKEN_VLM else None


def require_gdino() -> str:
    if not _port_open(GDINO_URL):
        _unavailable(f"Grounding DINO server not reachable at {GDINO_URL} "
                     "(start it: pixi run -m src/service_detector_gdino/pixi.toml serve)")
    if (err := _gdino_ready()) is not None:
        pytest.fail(f"Grounding DINO is reachable but a synthetic inference failed: {err}")
    return GDINO_URL


def require_wemm() -> str:
    url = os.environ.get("VAD_WEMM_URL")
    if not url:
        _unavailable("VAD_WEMM_URL is not set")
    if not _port_open(url):
        _unavailable("WeMM server at $VAD_WEMM_URL not reachable (VPN?)")  # never print the endpoint
    if (err := _wemm_ready(url)) is not None:
        pytest.fail(f"WeMM server is reachable but a synthetic inference failed ({err})")
    return url


def require_vlm() -> OpenAICompatibleVerifier:
    if not _port_open(VLM_URL):
        _unavailable(f"VLM endpoint not reachable at {VLM_URL}")
    if (err := _vlm_ready()) is not None:
        pytest.fail(f"VLM endpoint {VLM_URL} ({VLM_MODEL}) is reachable but a synthetic verification failed: {err}")
    return vlm_verifier()
