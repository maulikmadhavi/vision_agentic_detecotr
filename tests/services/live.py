"""Reachability probes for the live tests: skip with a clear reason when a service is not reachable.

A skip is not a pass: run the live tasks with ``-rs`` and record the skip reasons.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlsplit

import httpx
import pytest

from routing.task_config import VLMConfig
from service_detector_vlm.openai_service_call import OpenAICompatibleVerifier

GDINO_URL = os.environ.get("VAD_GDINO_URL", "http://127.0.0.1:8001")


def _port_open(url: str) -> bool:
    u = urlsplit(url)
    try:
        socket.create_connection((u.hostname, u.port or 80), timeout=3).close()
        return True
    except OSError:
        return False


def require_gdino() -> str:
    if not _port_open(GDINO_URL):
        pytest.skip(f"Grounding DINO server not reachable at {GDINO_URL} "
                    "(start it: pixi run -m src/service_detector_gdino/pixi.toml serve)")
    return GDINO_URL


def require_wemm() -> str:
    url = os.environ.get("VAD_WEMM_URL")
    if not url:
        pytest.skip("VAD_WEMM_URL is not set")
    if not _port_open(url):
        pytest.skip("WeMM server at $VAD_WEMM_URL not reachable (VPN?)")  # never print the endpoint
    return url


def require_lmstudio() -> OpenAICompatibleVerifier:
    cfg = VLMConfig()  # the configured local endpoint/model (C4 makes this explicit)
    vlm = OpenAICompatibleVerifier(cfg.base_url, cfg.model)
    try:
        ids = {m["id"] for m in httpx.get(f"{vlm.base_url}/models", timeout=5).json()["data"]}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        pytest.skip(f"LM Studio not reachable at {vlm.base_url}: {type(e).__name__}")
    if vlm.model not in ids:
        pytest.skip(f"LM Studio does not list model {vlm.model!r}")
    return vlm
