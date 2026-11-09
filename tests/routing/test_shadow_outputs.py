"""S4b (remaining R5-5): a shadow run writes each image once, or refuses before writing anything."""

from __future__ import annotations

import json

import pytest

from routing import cli
from routing.image_inputs import ManifestEntry
from routing.pipeline import RoutingPipeline
from routing.store import RouteStore

from .helpers import ScriptedDetector, config, make_fixture, task


def _shadow(fx, out, rows):
    manifest = fx["root"] / "dup_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return cli.main(["shadow", "--config", str(fx["config"]), "--images", str(manifest),
                     "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1", "--out", str(out)])


@pytest.mark.parametrize("rows, match", [
    ([{"uri": "img_easy.png", "image_id": "easy"}, {"uri": "img_easy.png", "image_id": "easy"}], "'easy'"),
    ([{"uri": "img_easy.png"}, {"uri": "img_easy.png"}], "sha256-"),  # no explicit id: the content-hash id repeats
    ([{"uri": "img_corrupt.png", "image_id": "corrupt"}, {"uri": "img_corrupt.png", "image_id": "corrupt"}],
     "'corrupt'"),  # unreadable entries follow the same rule
    ([{"uri": "missing.png"}, {"uri": "missing.png"}], "missing.png"),  # unreadable and unnamed: the uri is the id
], ids=["same entry twice", "same file without ids", "unreadable entry twice", "missing file without ids"])
def test_shadow_refuses_a_repeated_manifest_image_before_creating_anything(tmp_path, capsys, rows, match):
    fx = make_fixture(tmp_path / "fx")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        _shadow(fx, out, rows)
    err = capsys.readouterr().err
    assert exc.value.code == 2 and "repeated image ids" in err and match in err and not out.exists()


def test_two_files_sharing_an_explicit_id_are_refused_as_a_conflict(tmp_path, capsys):
    fx = make_fixture(tmp_path / "fx")
    with pytest.raises(SystemExit) as exc:
        _shadow(fx, tmp_path / "out", [{"uri": "img_easy.png", "image_id": "easy"},
                                       {"uri": "img_zero.png", "image_id": "easy"}])
    err = capsys.readouterr().err
    assert exc.value.code == 2 and "'easy'" in err and "different files" in err and not (tmp_path / "out").exists()


def test_the_same_images_in_separate_runs_are_fine(tmp_path):
    fx = make_fixture(tmp_path / "fx")
    rows = [{"uri": "img_easy.png", "image_id": "easy"}, {"uri": "img_zero.png", "image_id": "zero"}]
    assert _shadow(fx, tmp_path / "a", rows) == 0 and _shadow(fx, tmp_path / "b", rows) == 0


def _record(tmp_path, name="one"):
    fx_img = tmp_path / f"{name}.png"
    from PIL import Image
    Image.new("RGB", (32, 32), (255, 255, 255)).save(fx_img)
    return RoutingPipeline(config(), ScriptedDetector({}), RouteStore(":memory:")).process(
        ManifestEntry(str(fx_img), image_id=name))


def test_the_output_boundary_refuses_repeated_records_and_writes_nothing(tmp_path):
    """The reviewer's case: [record, record] produced two annotation images, two trace rows, images = 2."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "annotations.json").write_text("earlier run", encoding="utf-8")
    record = _record(tmp_path)
    with pytest.raises(ValueError, match=r"repeated image ids.*\['one'\]"):
        cli._write_shadow_outputs([record, _record(tmp_path, "two"), record], task(), out)
    assert (out / "annotations.json").read_text(encoding="utf-8") == "earlier run"
    assert not (out / "route_trace.jsonl").exists() and not (out / "report.json").exists()


def test_the_output_boundary_refuses_a_repeated_error_record(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    err = RoutingPipeline(config(), ScriptedDetector({}), RouteStore(":memory:")).process(
        ManifestEntry(str(bad), image_id="bad"))
    with pytest.raises(ValueError, match="repeated"):
        cli._write_shadow_outputs([err, err], task(), tmp_path)
    assert not (tmp_path / "annotations.json").exists()
