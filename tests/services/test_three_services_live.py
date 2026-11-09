"""Live shadow run with all three services at once (excluded by default; synthetic images only).

Needs the Grounding DINO server (pixi run -m src/service_detector_gdino/pixi.toml serve), LM Studio
with qwen3-vl-4b-instruct, and the WeMM server at $VAD_WEMM_URL. Skips if any one is unreachable.
Run:  VAD_WEMM_URL=<url> pixi run test-e2e
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

from routing import cli
from routing.schemas import VerificationResult

from .live import require_gdino, require_lmstudio, require_wemm

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
REVISION = "a2bb814dd30d+bt0.35+tt0.25"
WEIGHTS = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
BLUE, RED = (0, 0, 255), (255, 0, 0)
IMAGES = {  # name -> (size, [(shape, box, color)])
    "square": ((640, 480), [("rect", (200, 160, 359, 299), BLUE)]),
    "circle": ((640, 480), [("ellipse", (260, 180, 379, 299), RED)]),
    "both": ((640, 480), [("rect", (60, 60, 179, 179), BLUE), ("ellipse", (420, 280, 539, 399), RED)]),
    "blank": ((640, 480), []),
    "tiny": ((1280, 960), [("rect", (900, 700, 915, 715), BLUE)]),  # small object: exercises slicing
}


@pytest.fixture(autouse=True)
def services():
    return {"gdino": require_gdino(), "wemm": require_wemm(), "vlm": require_lmstudio()}


def _closed_port_url(path: str = "") -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now: connections are refused
    return f"http://127.0.0.1:{port}{path}"


def _inputs(root: Path, names, vlm_base_url: str | None = None) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        size, shapes = IMAGES[name]
        im = Image.new("RGB", size, "white")
        d = ImageDraw.Draw(im)
        for shape, box, color in shapes:
            (d.rectangle if shape == "rect" else d.ellipse)(box, fill=color)
        im.save(root / f"{name}.png")
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"uri": f"{n}.png", "image_id": n, "group_id": n}) + "\n" for n in names))
    cfg = yaml.safe_load((REPO / "configs/routing/example.yaml").read_text(encoding="utf-8"))
    cfg["task"] = {"task_id": "e2e-synthetic", "prompt_set_version": "p1", "classes": [
        {"class_id": "blue_square", "display_name": "Blue square", "positive_prompts": ["a blue square"]},
        {"class_id": "red_circle", "display_name": "Red circle", "positive_prompts": ["a red circle"]}]}
    cfg["relevance"].update(raw_scale="wemm_text_cosine_v1", raw_high=0.3, raw_low=0.1)  # test-only bands
    if vlm_base_url:
        cfg["vlm"]["base_url"] = vlm_base_url
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return manifest, config


def _shadow(root: Path, manifest: Path, config: Path, primary_url: str) -> tuple[int, dict, list[dict], dict]:
    out = root / "out"
    rc = cli.main(["shadow", "--config", str(config), "--images", str(manifest), "--primary-url", primary_url,
                   "--primary-model-id", "grounding-dino-tiny", "--primary-revision", REVISION,
                   "--primary-weights-sha256", WEIGHTS, "--primary-timeout", "120", "--wemm", "--vlm",
                   "--out", str(out)])
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    records = [json.loads(ln) for ln in (out / "route_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    return rc, report, records, json.loads((out / "annotations.json").read_text(encoding="utf-8"))


def test_three_services_route_verify_and_export_together(tmp_path, services):
    rc, report, records, ann = _shadow(tmp_path, *_inputs(tmp_path, list(IMAGES)), services["gdino"])
    assert rc == 0 and report["errors"] == [] and len(records) == len(IMAGES)
    assert report["relevance_scorer_failures"] == {"count": 0, "types": {}}

    # Run identity records all three servers, identically for every record.
    (run,) = {json.dumps(r["run"], sort_keys=True) for r in records}
    run = json.loads(run)
    assert run["primary_model"] == f"remote:grounding-dino-tiny@{REVISION}" and run["primary_artifact_digest"] == WEIGHTS
    assert run["relevance_scorer"].startswith("wemm:tencent/WeMM-Embedding-4B/d2560@")
    assert run["vlm_model"] == services["vlm"].model_version

    for r in records:  # WeMM scored every image
        assert r["stage_a"]["relevance_status"] != "unavailable" and set(r["stage_a"]["per_class"]) == {
            "blue_square", "red_circle"}
    verifications = [v for r in records for v in r["verifications"]]
    assert verifications, "the VLM was never called; the run did not exercise LM Studio"
    assert {v["model_version"] for v in verifications} == {run["vlm_model"]}
    assert not {v["reason_code"] for v in verifications} & {"vlm_unavailable", "vlm_timeout"}
    assert any(v["verdict"] != "uncertain" for v in verifications)

    # The VLM never authors boxes: every final box is a detector candidate's box.
    assert "box_xyxy" not in VerificationResult.model_fields
    accepted = set()
    for r in records:
        dets = {d["id"]: d for d in r["primary_candidates"] + r["secondary_candidates"] + r["sahi_candidates"]}
        for c in r["final_candidates"]:
            assert set(c["source_ids"]) <= set(dets) and c["box_xyxy"] == dets[c["canonical_source_id"]]["box_xyxy"]
            if c["status"] == "accepted":
                accepted.add(c["id"])
    # Only accepted candidates are exported.
    assert sorted(a["vad_candidate_id"] for a in ann["annotations"]) == sorted(accepted)


def test_detector_down_queues_the_image_as_an_error(tmp_path):
    rc, report, records, ann = _shadow(tmp_path, *_inputs(tmp_path, ["square"]), _closed_port_url())
    assert rc == 0 and records == [] and ann["annotations"] == []
    (err,) = report["errors"]
    assert err["error"].startswith("routing_failed_at:primary: RemoteDetectorError")


def test_wemm_down_leaves_relevance_unavailable_and_warns(tmp_path, services, monkeypatch, capsys):
    monkeypatch.setenv("VAD_WEMM_URL", _closed_port_url("/mcp"))
    rc, report, (rec,), _ = _shadow(tmp_path, *_inputs(tmp_path, ["square"]), services["gdino"])
    assert rc == 0 and report["errors"] == []
    assert report["relevance_scorer_failures"] == {"count": 1, "types": {"RemoteScorerError": 1}}
    assert rec["stage_a"]["relevance_status"] == "unavailable" and rec["stage_a"]["per_class"] == {}
    assert "relevance_scorer_failed:RemoteScorerError" in rec["stage_a"]["warnings"]
    assert rec["completeness"]["status"] != "complete"
    assert "relevance scorer failed for 1 of 1 images" in capsys.readouterr().err


def test_lmstudio_down_gives_uncertain_verifications_and_review(tmp_path, services):
    manifest, config = _inputs(tmp_path, ["square"], vlm_base_url=_closed_port_url("/v1"))
    rc, report, (rec,), _ = _shadow(tmp_path, manifest, config, services["gdino"])
    assert rc == 0 and report["errors"] == []
    assert rec["verifications"] and {(v["verdict"], v["reason_code"]) for v in rec["verifications"]} == {
        ("uncertain", "vlm_unavailable")}
    assert rec["completeness"]["status"] != "complete"
    assert not any(c["status"] == "accepted" and c["verification"] for c in rec["final_candidates"])
