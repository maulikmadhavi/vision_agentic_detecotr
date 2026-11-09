"""OpenAI-compatible VLM verifier: explicit endpoint/model, optional bearer token from a named env var."""

from __future__ import annotations

import json

import httpx
import pytest
import yaml
from PIL import Image

from routing import cli
from routing.schemas import VerificationRequest
from service_detector_vlm.openai_service_call import OpenAICompatibleVerifier

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


def test_recorded_identity_names_the_model_and_the_endpoint():
    assert OpenAICompatibleVerifier("http://vlm.test:1234/v1", "m").model_version == "openai-compatible:m@vlm.test:1234/v1"


def test_same_model_on_two_endpoints_has_two_identities():
    a = OpenAICompatibleVerifier("http://gpu-a.local:1234/v1", "m").model_version
    b = OpenAICompatibleVerifier("http://gpu-b.local:1234/v1", "m").model_version
    assert a != b


def test_identity_never_contains_url_credentials():
    v = OpenAICompatibleVerifier("https://user:pa55word@vlm.example/v1/", "m")
    assert v.model_version == "openai-compatible:m@vlm.example/v1" and "pa55word" not in v.model_version


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
    assert "s3cret-token" not in repr(v) and v.model_version == "openai-compatible:some-vl-model@vlm.test/v1"


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
    cfg["vlm"].update(base_url="http://vlm.test/v1", model="some-vl-model", api_key_env="VAD_TEST_VLM_KEY")
    fx["config"].write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args = ["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
            "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1", "--vlm",
            "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2 and "VAD_TEST_VLM_KEY" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("missing", ["base_url", "model"])
def test_cli_vlm_needs_an_explicit_endpoint_and_model(tmp_path, capsys, missing):
    fx = make_fixture(tmp_path / "fx")
    cfg = yaml.safe_load(fx["config"].read_text(encoding="utf-8"))
    cfg["vlm"].update(base_url="http://vlm.test/v1", model="some-vl-model")
    cfg["vlm"].pop(missing)
    fx["config"].write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args = ["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
            "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1", "--vlm",
            "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2 and f"vlm.{missing}" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_vlm_config_has_no_built_in_endpoint_or_model():
    assert (config().vlm.base_url, config().vlm.model) == (None, None)
