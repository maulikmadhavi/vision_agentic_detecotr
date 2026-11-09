"""The live-test probes themselves (default suite; no service is contacted)."""

from __future__ import annotations

import pytest

from . import live


@pytest.mark.parametrize("url,port", [("https://vlm.example/v1", 443), ("http://vlm.example/v1", 80),
                                      ("https://vlm.example:8443/v1", 8443)])
def test_port_probe_uses_the_scheme_default_port(monkeypatch, url, port):
    seen = []

    def connect(address, timeout):
        seen.append(address)
        raise OSError("not really connecting")

    monkeypatch.setattr(live.socket, "create_connection", connect)
    assert live._port_open(url) is False and seen == [("vlm.example", port)]


def test_unreachable_service_skips_by_default_and_fails_in_required_mode(monkeypatch):
    monkeypatch.setattr(live, "_port_open", lambda url: False)
    monkeypatch.delenv("VAD_LIVE_REQUIRED", raising=False)
    with pytest.raises(pytest.skip.Exception, match="not reachable"):
        live.require_gdino()
    monkeypatch.setenv("VAD_LIVE_REQUIRED", "1")
    with pytest.raises(pytest.fail.Exception, match="not reachable"):
        live.require_gdino()


def test_reachable_service_that_fails_a_synthetic_inference_fails_not_skips(monkeypatch):
    monkeypatch.delenv("VAD_LIVE_REQUIRED", raising=False)
    monkeypatch.setattr(live, "_port_open", lambda url: True)
    monkeypatch.setattr(live, "_vlm_ready", lambda: "vlm_unavailable")
    with pytest.raises(pytest.fail.Exception, match="synthetic"):
        live.require_vlm()


def test_unset_wemm_url_never_prints_an_endpoint(monkeypatch):
    monkeypatch.delenv("VAD_WEMM_URL", raising=False)
    monkeypatch.setenv("VAD_LIVE_REQUIRED", "1")
    with pytest.raises(pytest.fail.Exception, match="VAD_WEMM_URL is not set"):
        live.require_wemm()
