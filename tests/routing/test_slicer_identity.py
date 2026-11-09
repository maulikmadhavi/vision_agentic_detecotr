"""A dedicated slicing detector is part of the run identity; runs without one keep their keys."""

from __future__ import annotations

import pytest

from routing.image_inputs import ManifestEntry
from routing.pipeline import RoutingPipeline
from routing.schemas import ImageRef, RouteRecord
from routing.store import RouteStore

from .helpers import CLASS_A, CLASS_B, BlobDetector, ScriptedDetector, ScriptedRelevance, config, draw

REF = ImageRef(id="a", uri="a.png", width=320, height=240, sha256="0" * 64)


class Kill(BaseException):
    """A process death: escapes ``except Exception`` like a real crash would."""


class _Slicer(BlobDetector):
    """Crop-capable fake whose artifact digest can be set, like a file-backed model."""

    def __init__(self, model_version="crop-A", artifact_digest=None):
        super().__init__(model_version)
        self.artifact_digest = artifact_digest


def _small_object(tmp_path):
    entry = ManifestEntry(str(draw(tmp_path / "a.png", (320, 240), [(CLASS_A, (150, 110, 158, 118))])), image_id="a")
    return entry, ScriptedRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.05}})  # tiny box -> high_miss_risk -> sahi


def _run(tmp_path, slicer, store=None):
    entry, rel = _small_object(tmp_path)
    store = store or RouteStore(tmp_path / "r.sqlite")
    return RoutingPipeline(config(), BlobDetector("prim@1"), store, relevance=rel, sahi_detector=slicer).process(entry)


# --- runs without a dedicated slicer keep their keys (values computed before the change) -------


def test_no_slicer_keeps_its_record_key():
    pipe = RoutingPipeline(config(), ScriptedDetector({}, "prim@1"), RouteStore(":memory:"))
    assert pipe.record_key(REF) == "rec-df065f380d6cc7b1"
    assert (pipe.run_info.sahi_model, pipe.run_info.sahi_artifact_digest) == (None, None)


def test_primary_as_slicer_keeps_its_record_key():
    pipe = RoutingPipeline(config(), BlobDetector("blob-prim@1"), RouteStore(":memory:"))
    assert pipe.record_key(REF) == "rec-ffb63ad9c73ab3f3"  # the primary's identity already covers slicing
    assert pipe.run_info.sahi_model is None


def test_the_primary_passed_again_as_slicer_is_not_recorded_twice():
    primary = BlobDetector("blob-prim@1")
    pipe = RoutingPipeline(config(), primary, RouteStore(":memory:"), sahi_detector=primary)
    assert pipe.record_key(REF) == "rec-ffb63ad9c73ab3f3" and pipe.run_info.sahi_model is None


# --- a dedicated slicer is recorded ------------------------------------------------------------


def test_dedicated_slicer_identity_is_recorded():
    pipe = RoutingPipeline(config(), ScriptedDetector({}, "prim@1"), RouteStore(":memory:"),
                           sahi_detector=_Slicer("crop-A", "digest-1"))
    assert (pipe.run_info.sahi_model, pipe.run_info.sahi_artifact_digest) == ("crop-A", "digest-1")
    assert pipe.record_key(REF) != "rec-df065f380d6cc7b1"


@pytest.mark.parametrize("second", [_Slicer("crop-B"), _Slicer("crop-A", "digest-2")], ids=["model", "digest"])
def test_a_changed_slicer_does_not_reuse_the_old_final_record(tmp_path, second):
    first = _run(tmp_path, _Slicer("crop-A", "digest-1"))
    rec = _run(tmp_path, second)
    assert second.calls > 0 and rec.record_id != first.record_id
    assert (rec.run.sahi_model, rec.run.sahi_artifact_digest) == (second.model_version, second.artifact_digest)


@pytest.mark.parametrize("second", [_Slicer("crop-B"), _Slicer("crop-A", "digest-2")], ids=["model", "digest"])
def test_a_changed_slicer_does_not_reuse_old_slices_after_an_interruption(tmp_path, second):
    store = RouteStore(tmp_path / "r.sqlite")

    def put_record(rec):
        raise Kill()
    store.put_record = put_record
    with pytest.raises(Kill):
        _run(tmp_path, _Slicer("crop-A", "digest-1"), store)  # the sahi step is persisted, the record is not
    rec = _run(tmp_path, second)
    assert second.calls > 0 and rec.run.sahi_model == second.model_version


def test_an_unchanged_slicer_reuses_the_final_record(tmp_path):
    first = _run(tmp_path, _Slicer("crop-A", "digest-1"))
    again = _Slicer("crop-A", "digest-1")
    rec = _run(tmp_path, again)
    assert again.calls == 0 and rec == first


def test_an_unchanged_slicer_reuses_slices_after_an_interruption(tmp_path):
    store = RouteStore(tmp_path / "r.sqlite")

    def put_record(rec):
        raise Kill()
    store.put_record = put_record
    with pytest.raises(Kill):
        _run(tmp_path, _Slicer("crop-A", "digest-1"), store)
    again = _Slicer("crop-A", "digest-1")
    _run(tmp_path, again)
    assert again.calls == 0


def test_records_stored_before_the_slicer_fields_existed_still_load(tmp_path):
    data = _run(tmp_path, None).model_dump(mode="json")
    for name in ("sahi_model", "sahi_artifact_digest"):
        data["run"].pop(name, None)
    assert RouteRecord.model_validate(data).run.sahi_model is None
