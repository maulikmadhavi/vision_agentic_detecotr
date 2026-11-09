"""OpenAI-compatible VLM verifier: explicit endpoint/model, optional bearer token from a named env var."""

from __future__ import annotations

import json

import httpx
import pytest
import yaml
from PIL import Image

from routing import cli
from routing.schemas import VerificationRequest
from services.lmstudio import OpenAICompatibleVerifier

from ..routing.helpers import CLASS_A, config, make_fixture

REQ = VerificationRequest(id="r1", image_id="a", class_id=CLASS_A, purpose="suspected_missing_class")
ANSWER = {"choices": [{"message": {"content": json.dumps({"class_id": CLASS_A, "verdict": "present", "reason": "x"})}}]}


def _verifier(calls, **kw):
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=ANSWER)
    return OpenAICompatibleVerifier("http://vlm.test/v1", "some-vl-model", transport=httpx.MockTransport(handler), **kw)


def test_endpoint_and_model_must_be_given_explicitly():
    with pytest.raises(TypeError):
        OpenAICompatibleVerifier()


def test_recorded_identity_is_unchanged_by_the_rename():
    assert OpenAICompatibleVerifier("http://vlm.test/v1", "m").model_version == "lmstudio:m"


def test_no_authorization_header_without_api_key_env():
    calls = []
    assert _verifier(calls).verify(Image.new("RGB", (8, 8)), REQ, "x").verdict == "present"
    assert "authorization" not in calls[0].headers


def test_bearer_token_comes_from_the_named_env_var_and_stays_out_of_identity(monkeypatch):
    monkeypatch.setenv("VAD_TEST_VLM_KEY", "s3cret-token")
    calls = []
    v = _verifier(calls, api_key_env="VAD_TEST_VLM_KEY")
    assert v.verify(Image.new("RGB", (8, 8)), REQ, "x").verdict == "present"
    assert calls[0].headers["authorization"] == "Bearer s3cret-token"
    assert "s3cret-token" not in repr(v) and v.model_version == "lmstudio:some-vl-model"


def test_named_env_var_that_is_not_set_fails_at_construction(monkeypatch):
    monkeypatch.delenv("VAD_TEST_VLM_KEY", raising=False)
    with pytest.raises(ValueError, match="VAD_TEST_VLM_KEY"):
        OpenAICompatibleVerifier("http://vlm.test/v1", "m", api_key_env="VAD_TEST_VLM_KEY")


def test_api_key_env_is_not_part_of_the_config_hash():
    assert config(vlm={"api_key_env": "VAD_TEST_VLM_KEY"}).config_hash() == config().config_hash()


def test_cli_vlm_with_an_unset_key_variable_fails_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VAD_TEST_VLM_KEY", raising=False)
    fx = make_fixture(tmp_path / "fx")
    cfg = yaml.safe_load(fx["config"].read_text(encoding="utf-8"))
    cfg["vlm"]["api_key_env"] = "VAD_TEST_VLM_KEY"
    fx["config"].write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args = ["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
            "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1", "--vlm",
            "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2 and "VAD_TEST_VLM_KEY" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
