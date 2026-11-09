"""Contract tests for RemoteDetector against fake HTTP servers (no network, no torch)."""

from __future__ import annotations

import base64
import hashlib
import io
import json

import httpx
import pytest
from PIL import Image

from routing.interfaces import RawDetection
from routing.image_inputs import ManifestEntry
from routing.pipeline import RoutingPipeline
from routing.store import RouteStore
from services.http_detector import RemoteDetector, RemoteDetectorError
from routing.schemas import ErrorRecord, RouteRecord
from routing.slicing import run_sliced
from routing.task_config import SlicingConfig

from ..routing.helpers import CLASS_A, CLASS_B, BlobDetector, config, draw, task

REV = "rev-1"
WEIGHTS = "a" * 64


def server(reply, calls=None):
    """Fake detector server: `reply(request_json, image)` returns the response dict (or raises)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        img = Image.open(io.BytesIO(base64.b64decode(body["image_png_b64"])))
        out = reply(body, img)
        return out if isinstance(out, httpx.Response) else httpx.Response(200, json=out)

    return httpx.MockTransport(handler)


def ok(body, img, detections=(), **over):
    resp = {"schema_version": "svc-1.0", "request_id": body["request_id"], "status": "ok",
            "model_id": "gdino-tiny", "model_revision": REV, "weights_sha256": WEIGHTS,
            "coord_system": "xyxy_pixel_submitted", "width": img.width, "height": img.height,
            "detections": list(detections), "gpu_ms": 12.5}
    resp.update(over)
    return resp


def detector(transport, **kw):
    args = {"base_url": "http://detector.test", "task": task(), "model_id": "gdino-tiny",
            "model_revision": REV, "weights_sha256": WEIGHTS, "transport": transport}
    args.update(kw)
    return RemoteDetector(**args)


def _img(w=320, h=240):
    return Image.new("RGB", (w, h), (200, 200, 200))


def test_boxes_come_back_as_raw_detections_with_server_gpu_time():
    calls = []
    det = detector(server(lambda b, i: ok(b, i, [{"class_id": CLASS_A, "box_xyxy": [10, 20, 50, 60], "score": 0.7}]),
                          calls))
    res = det.detect(_img(), "img1", task().class_ids)
    assert res.detections == [RawDetection(CLASS_A, (10, 20, 50, 60), 0.7)] and res.gpu_ms == 12.5
    (req,) = calls
    assert req["schema_version"] == "svc-1.0" and (req["width"], req["height"]) == (320, 240)
    assert req["image_sha256"] == hashlib.sha256(base64.b64decode(req["image_png_b64"])).hexdigest()
    assert {c["class_id"]: c["prompts"] for c in req["classes"]} == {
        CLASS_A: ["a blue X9 widget"], CLASS_B: ["a red custom thing"]}  # project prompts, no taxonomy
    assert det.model_version == f"remote:gdino-tiny@{REV}" and det.artifact_digest == WEIGHTS


def test_missing_gpu_time_stays_none():
    res = detector(server(lambda b, i: ok(b, i, gpu_ms=None))).detect(_img(), "img1", task().class_ids)
    assert res.gpu_ms is None


def test_valid_empty_result_is_zero_boxes_not_an_error(tmp_path):
    entry = ManifestEntry(str(draw(tmp_path / "a.png", (320, 240), [])), image_id="a")
    rec = RoutingPipeline(config(), detector(server(lambda b, i: ok(b, i))),
                          RouteStore(tmp_path / "r.sqlite")).process(entry)
    assert isinstance(rec, RouteRecord) and rec.primary_candidates == ()


def _timeout(body, img):
    raise httpx.ReadTimeout("slow")


@pytest.mark.parametrize("reply", [
    lambda b, i: ok(b, i, status="failed", error="cuda oom"),
    lambda b, i: httpx.Response(200, content=b"not json"),
    lambda b, i: {**ok(b, i), "unexpected": 1},
    lambda b, i: httpx.Response(500, json={"error": "boom"}),
    _timeout,
    lambda b, i: ok(b, i, request_id="someone-else"),
    lambda b, i: ok(b, i, width=i.width + 1),
    lambda b, i: ok(b, i, model_revision="rev-2"),
    lambda b, i: ok(b, i, model_id="other-model"),
    lambda b, i: ok(b, i, weights_sha256="b" * 64),
    lambda b, i: ok(b, i, weights_sha256=None),  # expected digest cannot be verified
    lambda b, i: ok(b, i, coord_system="xyxy_normalized"),
], ids=["failed", "not-json", "extra-field", "http-500", "timeout", "request-id", "size-echo",
        "revision", "model-id", "weights", "weights-unreported", "coords"])
def test_failures_raise_and_never_become_zero_boxes(reply):
    with pytest.raises(RemoteDetectorError):
        detector(server(reply)).detect(_img(), "img1", task().class_ids)


def test_primary_failure_goes_to_the_error_queue(tmp_path):
    entry = ManifestEntry(str(draw(tmp_path / "a.png", (320, 240), [])), image_id="a")
    rec = RoutingPipeline(config(), detector(server(lambda b, i: ok(b, i, status="failed"))),
                          RouteStore(tmp_path / "r.sqlite")).process(entry)
    assert isinstance(rec, ErrorRecord) and rec.error.startswith("routing_failed_at:primary")


def test_unverified_weights_are_recorded_as_no_digest():
    det = detector(server(lambda b, i: ok(b, i, weights_sha256=None)), weights_sha256=None)
    assert det.detect(_img(), "img1", task().class_ids).detections == []
    assert det.artifact_digest is None  # board assumption 6: identifier only, not a verified artifact


def test_oversized_image_raises_before_any_call():
    calls = []
    det = detector(server(lambda b, i: ok(b, i), calls), max_image_pixels=320 * 240 - 1)
    with pytest.raises(RemoteDetectorError, match="max_image_pixels"):
        det.detect(_img(), "img1", task().class_ids)
    assert calls == []


def test_sahi_tiles_are_offset_once_into_original_pixels(tmp_path):
    # The fake server finds the blob in whatever crop it receives and answers in crop pixels.
    blob = BlobDetector()

    def reply(body, img):
        found = blob.detect(img, "x", task().class_ids).detections
        return ok(body, img, [{"class_id": d.class_id, "box_xyxy": list(d.box_xyxy), "score": 0.9} for d in found])

    path = draw(tmp_path / "big.png", (1200, 900), [(CLASS_A, (700, 500, 740, 540))])
    from routing.image_inputs import load_image
    image, ref = load_image(ManifestEntry(str(path), image_id="big"))
    calls = []
    res = run_sliced(detector(server(reply, calls)), image, ref, task().class_ids,
                     SlicingConfig(tile=640, overlap=0.2, include_full_image=False), {}, max_slices=16)
    assert len(calls) == res.slices_run > 1  # one API call per tile
    assert [d.box_xyxy for d in res.detections] == [(700.0, 500.0, 740.0, 540.0)]


def test_model_revision_is_part_of_the_record_key(tmp_path):
    entry = ManifestEntry(str(draw(tmp_path / "a.png", (320, 240), [])), image_id="a")
    from routing.image_inputs import load_image
    _, ref = load_image(entry)
    keys = {RoutingPipeline(config(), detector(None, model_revision=r), RouteStore(":memory:")).record_key(ref)
            for r in ("rev-1", "rev-2")}
    assert len(keys) == 2


def test_class_outside_the_task_raises_before_any_call():
    calls = []
    with pytest.raises(RemoteDetectorError, match="not in task"):
        detector(server(lambda b, i: ok(b, i), calls)).detect(_img(), "img1", (CLASS_A, "unknown_q"))
    assert calls == []


# --- stage 1: `vad-route shadow --primary-url` -------------------------------------------------


def _cli_args(fx, out, *extra):
    return ["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]), "--out", str(out),
            "--relevance-scores", str(fx["relevance"]), *extra]


def test_cli_shadow_with_primary_url_uses_remote_detector(tmp_path, monkeypatch):
    from routing import cli
    from ..routing.helpers import make_fixture

    blob = BlobDetector()

    def reply(body, img):
        found = blob.detect(img, "x", task().class_ids).detections
        return ok(body, img, [{"class_id": d.class_id, "box_xyxy": list(d.box_xyxy), "score": 0.9} for d in found])

    calls = []
    monkeypatch.setattr(cli, "RemoteDetector", lambda **kw: RemoteDetector(transport=server(reply, calls), **kw))
    fx = make_fixture(tmp_path / "fx")
    out = tmp_path / "shadow"
    assert cli.main(_cli_args(fx, out, "--primary-url", "http://detector.test", "--primary-model-id", "gdino-tiny",
                              "--primary-revision", REV, "--primary-weights-sha256", WEIGHTS)) == 0
    assert calls  # the remote server was actually called
    trace = [json.loads(line) for line in (out / "route_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {t["run"]["primary_model"] for t in trace} == {f"remote:gdino-tiny@{REV}"}  # single version source
    assert {t["run"]["primary_artifact_digest"] for t in trace} == {WEIGHTS}
    ann = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
    easy = next(i["id"] for i in ann["images"] if i["vad_image_id"] == "easy")
    assert [a["bbox"] for a in ann["annotations"] if a["image_id"] == easy] == [[60.0, 60.0, 100.0, 100.0]]


@pytest.mark.parametrize("extra,message", [
    (["--primary-url", "http://d", "--primary-model-id", "m", "--primary-revision", "r", "--primary-version", "v@1"],
     "not used with --primary-url"),
    (["--primary-url", "http://d", "--primary-model-id", "m"], "needs --primary-model-id and --primary-revision"),
    (["--primary-predictions", "p.jsonl"], "--primary-version is required"),
    (["--primary-predictions", "p.jsonl", "--primary-version", "v@1", "--primary-revision", "r"],
     "only with --primary-url"),
    (["--primary-predictions", "p.jsonl", "--primary-version", "v@1", "--primary-url", "http://d"], "not allowed with"),
], ids=["version-with-url", "missing-revision", "missing-version", "remote-flag-without-url", "both-sources"])
def test_cli_primary_source_flags_are_validated(tmp_path, capsys, extra, message):
    from routing import cli
    from ..routing.helpers import make_fixture

    fx = make_fixture(tmp_path / "fx")
    with pytest.raises(SystemExit) as e:
        cli.main(_cli_args(fx, tmp_path / "out", *extra))
    assert e.value.code == 2 and message in capsys.readouterr().err
