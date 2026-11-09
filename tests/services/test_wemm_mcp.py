"""Contract tests for WeMMMCPRelevanceScorer (WeMM over MCP JSON-RPC) with fake HTTP (no server, no VPN)."""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from routing.image_inputs import ManifestEntry
from routing.pipeline import RoutingPipeline
from routing.store import RouteStore
from services_embedding.client import RemoteScorerError, WeMMMCPRelevanceScorer
from routing.schemas import RouteRecord

from ..routing.helpers import CLASS_A, CLASS_B, ScriptedDetector, config, draw

URL = "http://wemm.test/mcp"  # fake; the real endpoint only ever comes from $VAD_WEMM_URL
META = {"backend": "wemm", "model": "tencent/WeMM-Embedding-4B", "dim": 2560}
WEMM_CFG = {"raw_scale": "wemm_text_cosine_v1", "raw_high": 0.3, "raw_low": 0.1}  # test-only bands
PROMPTS = {CLASS_A: (("a blue X9 widget", "a small blue plastic widget"), ("a blue bottle cap",)),
           CLASS_B: (("a red custom thing",), ())}


def rpc_result(payload, is_error=False):
    return {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": json.dumps(payload)}],
                                                   "isError": is_error}}


def scores_payload(scores, meta=META):
    return {"results": [{"matches": [{"index": i, "label": f"t{i}", "score": s} for i, s in enumerate(scores)]}],
            "meta": meta}


def _rows(rows, meta=META):
    """Raw reply with explicit result rows of (index, score) matches, for malformed-response tests."""
    return {"results": [{"matches": [{"index": i, "label": f"t{i}", "score": s} for i, s in row]} for row in rows],
            "meta": meta}


def server(reply, calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        out = reply(json.loads(request.content), len(calls))
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, httpx.Response) else httpx.Response(200, json=out)
    return httpx.MockTransport(handler)


def scorer(reply, calls=None, **kw):
    calls = [] if calls is None else calls
    return WeMMMCPRelevanceScorer(url=URL, transport=server(reply, calls), **kw), calls


def _img(w=320, h=240):
    return Image.new("RGB", (w, h), "white")


def test_scores_are_per_class_max_over_positive_prompts():
    s, calls = scorer(lambda b, n: rpc_result(scores_payload([0.21, 0.35, 0.12])))
    out = s.score(_img(), "i", PROMPTS)
    assert out == {CLASS_A: 0.35, CLASS_B: 0.12}  # A = max(0.21, 0.35); negatives are not sent
    (req,) = calls
    body = json.loads(req.content)
    assert body["method"] == "tools/call" and body["params"]["name"] == "compute_similarity"
    args = body["params"]["arguments"]
    assert args["right"] == ["a blue X9 widget", "a small blue plastic widget", "a red custom thing"]
    assert (args["mode"], args["metric"], args["top_k"]) == ("matrix", "cosine", 3)
    assert len(args["left"]) == 1 and args["left"][0].startswith("data:image/png;base64,")
    assert "text/event-stream" in req.headers["accept"]
    assert s.score_scale == "wemm_text_cosine_v1"
    assert s.model_version == "wemm:tencent/WeMM-Embedding-4B/d2560@unverified+max448" and s.artifact_digest is None


def test_large_images_are_downscaled_to_max_side_before_sending():
    s, calls = scorer(lambda b, n: rpc_result(scores_payload([0.1, 0.1, 0.1])))
    s.score(_img(1000, 500), "i", PROMPTS)
    uri = json.loads(calls[0].content)["params"]["arguments"]["left"][0]
    sent = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert sent.size == (448, 224)


def test_event_stream_response_is_parsed():
    sse = "event: message\ndata: " + json.dumps(rpc_result(scores_payload([0.3, 0.2, 0.1]))) + "\n\n"
    s, _ = scorer(lambda b, n: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))
    assert s.score(_img(), "i", PROMPTS) == {CLASS_A: 0.3, CLASS_B: 0.1}


def test_build_is_verified_once_the_server_reports_it():
    meta = {**META, "build": "abc123"}
    ok, _ = scorer(lambda b, n: rpc_result(scores_payload([0.1, 0.1, 0.1], meta)), build="abc123")
    assert ok.score(_img(), "i", PROMPTS) and ok.model_version.endswith("@abc123+max448")
    bad, _ = scorer(lambda b, n: rpc_result(scores_payload([0.1, 0.1, 0.1])), build="abc123")  # build missing
    with pytest.raises(RemoteScorerError):
        bad.score(_img(), "i", PROMPTS)


FAILURES = {
    "isError": lambda b, n: rpc_result({"error": "backend down"}, is_error=True),
    "jsonrpc-error": lambda b, n: {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "boom"}},
    "malformed-text": lambda b, n: {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "{"}]}},
    "missing-index": lambda b, n: rpc_result({"results": [{"matches": [{"index": 0, "label": "t", "score": 0.2}]}],
                                              "meta": META}),
    "duplicate-index": lambda b, n: rpc_result(_rows([[(0, 0.1), (1, 0.9), (1, 0.2), (2, 0.1)]])),
    "extra-row": lambda b, n: rpc_result(_rows([[(0, 0.1), (1, 0.2), (2, 0.3)], [(0, 0.9), (1, 0.9), (2, 0.9)]])),
    "zero-rows": lambda b, n: rpc_result(_rows([])),
    "index-out-of-range": lambda b, n: rpc_result(_rows([[(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4)]])),
    "mock-backend": lambda b, n: rpc_result(scores_payload([0.1, 0.1, 0.1], {**META, "backend": "mock"})),
    "wrong-model": lambda b, n: rpc_result(scores_payload([0.1, 0.1, 0.1], {**META, "model": "other"})),
    "wrong-dim": lambda b, n: rpc_result(scores_payload([0.1, 0.1, 0.1], {**META, "dim": 1024})),
    "http-500": lambda b, n: httpx.Response(500, text="oops"),
    "network-twice": lambda b, n: httpx.ConnectError("vpn down"),
    "timeout-twice": lambda b, n: httpx.ReadTimeout("slow"),
    "nan-score": lambda b, n: rpc_result(_rows([[(0, 0.1), (1, float("nan")), (2, 0.3)]])),
    "inf-score": lambda b, n: rpc_result(_rows([[(0, 0.1), (1, float("inf")), (2, 0.3)]])),
    "non-numeric-score": lambda b, n: rpc_result(_rows([[(0, 0.1), (1, "high"), (2, 0.3)]])),
    "missing-meta": lambda b, n: rpc_result({"results": [{"matches": []}]}),
    "missing-results": lambda b, n: rpc_result({"meta": META}),
    "empty-content": lambda b, n: {"jsonrpc": "2.0", "id": 1, "result": {"content": []}},
    "sse-without-data": lambda b, n: httpx.Response(200, text="event: message\n\n",
                                                    headers={"content-type": "text/event-stream"}),
}
RETRIED = {"network-twice", "timeout-twice"}  # transport failures only: one retry, then fail


@pytest.mark.parametrize("name", list(FAILURES))
def test_failures_raise(name):
    s, calls = scorer(FAILURES[name])
    with pytest.raises(RemoteScorerError):
        s.score(_img(), "i", PROMPTS)
    assert len(calls) == (2 if name in RETRIED else 1)  # retry once, only on a network error


def test_positive_control_a_valid_reply_with_unordered_matches_and_a_real_zero_is_accepted():
    s, _ = scorer(lambda b, n: rpc_result(_rows([[(2, 0.0), (0, 0.0), (1, 0.25)]])))
    assert s.score(_img(), "i", PROMPTS) == {CLASS_A: 0.25, CLASS_B: 0.0}


def test_one_network_error_is_retried_once():
    s, calls = scorer(lambda b, n: httpx.ReadTimeout("slow") if n == 1 else rpc_result(scores_payload([0.4, 0.1, 0.2])))
    assert s.score(_img(), "i", PROMPTS) == {CLASS_A: 0.4, CLASS_B: 0.2} and len(calls) == 2


def _route(tmp_path, s, name="a"):
    entry = ManifestEntry(str(draw(tmp_path / f"{name}.png", (320, 240), [])), image_id=name)
    return RoutingPipeline(config(relevance=WEMM_CFG), ScriptedDetector({name: []}),
                           RouteStore(tmp_path / f"{name}.sqlite"), relevance=s).process(entry)


@pytest.mark.parametrize("name", list(FAILURES))
def test_zero_score_never_comes_from_an_error(tmp_path, name):
    # Behavior change: a scorer failure no longer sends the image to the error queue.
    # Stage A records relevance as unavailable, with no per-class score at all.
    rec = _route(tmp_path, scorer(FAILURES[name])[0], name.replace("-", "_"))
    assert isinstance(rec, RouteRecord)
    assert rec.stage_a.relevance_status == "unavailable" and rec.stage_a.per_class == {}
    assert "relevance_scorer_failed:RemoteScorerError" in rec.stage_a.warnings
    assert rec.completeness.status != "complete"


def test_positive_control_a_real_zero_cosine_is_recorded_as_zero(tmp_path):
    rec = _route(tmp_path, scorer(lambda b, n: rpc_result(scores_payload([0.0] * len(b["params"]["arguments"]["right"]))))[0])
    assert {k: v.raw_similarity for k, v in rec.stage_a.per_class.items()} == {CLASS_A: 0.0, CLASS_B: 0.0}
    assert rec.stage_a.relevance_status == "uncalibrated"


def test_any_scorer_exception_leaves_the_image_in_review_not_the_error_queue(tmp_path):
    class Broken:
        model_version = "broken@1"

        def score(self, image, image_id, prompts):
            raise RuntimeError("anything")

    rec = RoutingPipeline(config(), ScriptedDetector({"a": []}), RouteStore(tmp_path / "r.sqlite"),
                          relevance=Broken()).process(
        ManifestEntry(str(draw(tmp_path / "a.png", (320, 240), [])), image_id="a"))
    assert isinstance(rec, RouteRecord) and "relevance_scorer_failed:RuntimeError" in rec.stage_a.warnings
    assert rec.stage_a.relevance_status == "unavailable"


# --- CLI: --wemm reads the endpoint only from $VAD_WEMM_URL ---------------------------------------


def _fx(tmp_path, raw_scale="wemm_text_cosine_v1"):
    import yaml
    from ..routing.helpers import make_fixture
    fx = make_fixture(tmp_path / "fx")
    cfg = yaml.safe_load(fx["config"].read_text(encoding="utf-8"))
    cfg["relevance"].update(raw_scale=raw_scale, raw_high=0.3, raw_low=0.1)
    fx["config"].write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return fx


def _args(fx, out):
    return ["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
            "--primary-predictions", str(fx["preds"]), "--primary-version", "p@1", "--out", str(out), "--wemm"]


def test_cli_wemm_requires_the_env_var(tmp_path, monkeypatch, capsys):
    from routing import cli
    monkeypatch.delenv("VAD_WEMM_URL", raising=False)
    with pytest.raises(SystemExit):
        cli.main(_args(_fx(tmp_path), tmp_path / "out"))
    assert "VAD_WEMM_URL" in capsys.readouterr().err


def test_cli_wemm_requires_matching_raw_scale(tmp_path, monkeypatch, capsys):
    from routing import cli
    monkeypatch.setenv("VAD_WEMM_URL", URL)
    with pytest.raises(SystemExit):
        cli.main(_args(_fx(tmp_path, raw_scale="unspecified"), tmp_path / "out"))
    assert "wemm_text_cosine_v1" in capsys.readouterr().err


def test_cli_wemm_scores_stage_a(tmp_path, monkeypatch):
    from routing import cli
    calls = []
    monkeypatch.setenv("VAD_WEMM_URL", URL)
    monkeypatch.setattr(cli, "WeMMMCPRelevanceScorer", lambda **kw: WeMMMCPRelevanceScorer(
        transport=server(lambda b, n: rpc_result(scores_payload([0.4, 0.05][:len(b["params"]["arguments"]["right"])])),
                         calls), **kw))
    out = tmp_path / "out"
    assert cli.main(_args(_fx(tmp_path), out)) == 0
    trace = [json.loads(line) for line in (out / "route_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {t["run"]["relevance_scorer"] for t in trace} == {"wemm:tencent/WeMM-Embedding-4B/d2560@unverified+max448"}
    assert all(t["stage_a"]["per_class"][CLASS_A]["raw_similarity"] == 0.4 for t in trace) and calls


def test_rename_keeps_run_keys():
    # Golden key computed with the class under its first name (RemoteRelevanceScorer in
    # vad.routing.remote_adapters): renames and module moves must not change model_version,
    # RunInfo or the record key. The old-name alias was dropped in the restructure.
    from routing.schemas import ImageRef
    from services_embedding.client import WeMMMCPRelevanceScorer

    ref = ImageRef(id="a", uri="a.png", width=320, height=240, sha256="0" * 64)
    pipe = RoutingPipeline(config(relevance=WEMM_CFG), ScriptedDetector({}), RouteStore(":memory:"),
                           relevance=WeMMMCPRelevanceScorer(url=URL))
    assert pipe.record_key(ref) == "rec-e025bfad645c479b"


# --- Stage F: a run where the scorer failed must say so ------------------------------------------


def test_route_report_counts_relevance_scorer_failures(tmp_path):
    from routing.evaluation import route_report

    class Broken:
        model_version = "broken@1"

        def score(self, image, image_id, prompts):
            raise RuntimeError("down")

    pipe = RoutingPipeline(config(), ScriptedDetector({"a": [], "b": []}), RouteStore(tmp_path / "r.sqlite"),
                           relevance=Broken())
    recs = [pipe.process(ManifestEntry(str(draw(tmp_path / f"{n}.png", (320, 240), [])), image_id=n)) for n in "ab"]
    assert route_report(recs)["relevance_scorer_failures"] == {"count": 2, "types": {"RuntimeError": 2}}
    ok = _route(tmp_path, scorer(lambda b, n: rpc_result(scores_payload([0.2] * len(b["params"]["arguments"]["right"]))))[0])
    assert route_report([ok])["relevance_scorer_failures"] == {"count": 0, "types": {}}


def test_cli_warns_on_stderr_when_the_scorer_failed(tmp_path, monkeypatch, capsys):
    from routing import cli
    monkeypatch.setenv("VAD_WEMM_URL", URL)
    monkeypatch.setattr(cli, "WeMMMCPRelevanceScorer", lambda **kw: WeMMMCPRelevanceScorer(
        transport=server(FAILURES["isError"], []), **kw))
    out = tmp_path / "out"
    assert cli.main(_args(_fx(tmp_path), out)) == 0  # the run still completes; images go to review
    err = capsys.readouterr().err
    assert "relevance scorer failed for 3 of 3 images" in err and "RemoteScorerError" in err
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["relevance_scorer_failures"] == {"count": 3, "types": {"RemoteScorerError": 3}}
