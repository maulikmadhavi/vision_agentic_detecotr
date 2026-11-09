"""Tests for the 2026-11-09 reviewer follow-ups (docs/project_board.md)."""

from __future__ import annotations

import pytest

from routing import cli
from routing.adapters import RawDetection
from routing.evaluation import HumanLabel, verification_items, vlm_agreement
from routing.pipeline import ManifestEntry, RouteStore, RoutingPipeline, StoreLockedError, job_lock
from routing.policy import Capabilities, route
from routing.schemas import BoxFeatures, ClassRelevance, Difficulty, Quality, StageA, VerificationResult

from .helpers import CLASS_A, CLASS_B, FakeVLM, ScriptedDetector, ScriptedRelevance, config, draw, make_fixture


def _sa(band="high", raw=0.9, blur=0.1, status="uncalibrated"):
    per_class = {} if status == "unavailable" else {CLASS_A: ClassRelevance(raw_similarity=raw)}
    return StageA(per_class=per_class, relevance_status=status, relevance_band=band,
                  quality=Quality(blur=blur, exposure=0.0, min_side_px=480, low_resolution=False))


def _features(n=1):
    if n == 0:
        return BoxFeatures(num_boxes=0, per_class_counts={}, crowding=0, class_ambiguity=0,
                           high_relevance_zero_boxes=False)
    return BoxFeatures(num_boxes=1, per_class_counts={CLASS_A: 1}, min_box_area_fraction=0.1, crowding=0,
                       class_ambiguity=0, mean_confidence=0.95, min_confidence=0.95, high_relevance_zero_boxes=False)


EASY = Difficulty(value=0.1, status="uncalibrated", bucket="easy")
UNKNOWN = Difficulty(status="unknown", bucket="unknown", reasons=("zero_primary_boxes",))


def _entry(tmp_path, rects=()):
    return ManifestEntry(str(draw(tmp_path / "a.png", (320, 240), list(rects))), image_id="a")


# --- §5.4: heuristic quality cut-offs live in config -------------------------------------------


def test_poor_quality_cutoffs_come_from_config():
    sa = _sa(blur=0.6)
    always = {"poor_quality_sample_rate": 1.0}
    default = route("x", 640, 480, sa, _features(), EASY, Capabilities(), config(audit=always))
    lowered = route("x", 640, 480, sa, _features(), EASY, Capabilities(),
                    config(audit=always, quality={"poor_blur": 0.5}))
    assert "audit_sampled_poor_quality" not in default.reason_codes
    assert "audit_sampled_poor_quality" in lowered.reason_codes and "audit_sample" in lowered.actions


def test_uncalibrated_difficulty_is_labelled_in_reason_codes():
    d = route("x", 640, 480, _sa(), _features(), EASY, Capabilities(), config())
    assert "relevance_uncalibrated" in d.reason_codes and "difficulty_uncalibrated" in d.reason_codes


# --- §5.4: open policy choices are explicit config knobs; defaults keep route-v1 ----------------


def test_unknown_relevance_knob():
    sa = _sa(band="unknown", status="unavailable")
    default = route("x", 640, 480, sa, _features(0), UNKNOWN, Capabilities(), config())
    assert "moderate_or_unknown_relevance_zero_boxes" in default.reason_codes and default.priority == 1
    as_high = route("x", 640, 480, sa, _features(0), UNKNOWN, Capabilities(secondary=True),
                    config(policy={"unknown_relevance_as": "high"}))
    assert {"unknown_relevance_treated_as_high", "unknown_relevance_zero_boxes"} <= set(as_high.reason_codes)
    assert "high_relevance_zero_boxes" not in as_high.reason_codes  # never claims relevance it did not measure
    assert as_high.priority == 2 and "secondary" in as_high.actions


def test_unknown_relevance_cannot_be_configured_as_low():
    with pytest.raises(ValueError):
        config(policy={"unknown_relevance_as": "low"})


def test_moderate_without_secondary_knob():
    sa = _sa(band="medium", raw=0.4)
    caps = Capabilities(vlm=True)
    default = route("x", 640, 480, sa, _features(), EASY, caps, config())
    human = route("x", 640, 480, sa, _features(), EASY, caps, config(policy={"moderate_without_secondary": "human"}))
    assert "vlm_verify" in default.actions and "human_review" not in default.actions
    assert "human_review" in human.actions and "vlm_verify" not in human.actions


def test_cross_class_knob(tmp_path):
    box = (40, 40, 200, 200)
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, box, 0.95)]})
    secondary = ScriptedDetector({"a": [RawDetection(CLASS_B, box, 0.9)]}, "sec@1")
    rel = ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.4}})  # medium -> moderate route with secondary

    def run(sub, **policy):
        cfg = config(vlm={"enabled": True}, policy=policy)
        return RoutingPipeline(cfg, primary, RouteStore(tmp_path / sub / "r.sqlite"), secondary=secondary,
                               relevance=rel, vlm=FakeVLM(default="present")).process(_entry(tmp_path))

    def primary_status(rec):
        (c,) = [c for c in rec.final_candidates if c.class_id == CLASS_A]
        assert "cross_class_conflict" in c.flags
        return c.status

    assert primary_status(run("d")) == "needs_review"
    assert primary_status(run("off", cross_class_always_review=False)) == "accepted"


# --- same-model review finding: moderate route without a second detector ----------------------


def test_moderate_route_without_secondary_does_not_auto_accept_primary(tmp_path):
    box = (40, 40, 200, 200)
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, box, 0.97)]})
    rel = ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.1}})  # medium relevance, easy box

    no_vlm = RoutingPipeline(config(), primary, RouteStore(tmp_path / "n" / "r.sqlite"),
                             relevance=rel).process(_entry(tmp_path))
    assert "moderate_relevance" in no_vlm.route.reason_codes and "human_review" in no_vlm.route.actions
    assert [(c.status, "secondary_unconfirmed" in c.flags) for c in no_vlm.final_candidates] == [("needs_review", True)]

    vlm = FakeVLM(default="present")
    with_vlm = RoutingPipeline(config(vlm={"enabled": True}), primary, RouteStore(tmp_path / "v" / "r.sqlite"),
                               relevance=rel, vlm=vlm).process(_entry(tmp_path))
    assert [r.purpose for r, _ in vlm.requests] == ["confirm_candidate"]  # the box is actually checked
    assert [c.status for c in with_vlm.final_candidates] == ["accepted"]

    secondary = ScriptedDetector({"a": [RawDetection(CLASS_A, box, 0.9)]}, "sec@1")
    agreed = RoutingPipeline(config(), primary, RouteStore(tmp_path / "s" / "r.sqlite"), secondary=secondary,
                             relevance=rel).process(_entry(tmp_path))
    assert [(c.status, c.flags) for c in agreed.final_candidates] == [("accepted", ())]


def test_moderate_route_with_failed_secondary_does_not_auto_accept(tmp_path):
    class Boom:
        model_version, supports_crops = "sec@boom", False

        def detect(self, image, image_id, class_ids):
            raise RuntimeError("secondary down")

    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, (40, 40, 200, 200), 0.97)]})
    rel = ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.1}})
    rec = RoutingPipeline(config(), primary, RouteStore(tmp_path / "r.sqlite"), secondary=Boom(),
                          relevance=rel).process(_entry(tmp_path))
    assert "secondary_failed:RuntimeError" in rec.trace.warnings
    assert [c.status for c in rec.final_candidates] == ["needs_review"]


# --- §5.4: one writing job per output directory -------------------------------------------------


def test_job_lock_is_exclusive_and_released(tmp_path):
    with job_lock(tmp_path / "out"):
        with pytest.raises(StoreLockedError):
            with job_lock(tmp_path / "out"):
                pass
        with job_lock(tmp_path / "other"):  # different directory is independent
            pass
    with job_lock(tmp_path / "out"):  # released on exit
        pass


def test_cli_shadow_refuses_a_locked_output_dir(tmp_path, capsys):
    fx = make_fixture(tmp_path / "fx")
    out = tmp_path / "shadow"
    args = ["shadow", "--config", str(fx["config"]), "--images", str(fx["manifest"]),
            "--primary-predictions", str(fx["preds"]), "--primary-version", "prim@1", "--out", str(out)]
    with job_lock(out):
        assert cli.main(args) == 2
    assert "in use by another routing job" in capsys.readouterr().err
    assert not (out / "routes.sqlite").exists()
    assert cli.main(args) == 0


# --- CLIP scorer scale guard (no torch needed) ---------------------------------------------------


def test_scorer_with_declared_scale_requires_matching_config(tmp_path):
    class ScaledRelevance(ScriptedRelevance):
        score_scale = "clip_prompt_softmax_v1"

    rel = ScaledRelevance({"a": {CLASS_A: 0.9, CLASS_B: 0.1}})
    with pytest.raises(ValueError, match="raw_scale"):
        RoutingPipeline(config(), ScriptedDetector({"a": []}), RouteStore(":memory:"), relevance=rel)
    cfg = config(relevance={"raw_scale": "clip_prompt_softmax_v1", "raw_high": 0.6, "raw_low": 0.2})
    rec = RoutingPipeline(cfg, ScriptedDetector({"a": []}), RouteStore(":memory:"),
                          relevance=rel).process(_entry(tmp_path))
    assert rec.stage_a.relevance_band == "high"
    # scorers without a declared scale (precomputed files) are unaffected
    RoutingPipeline(config(), ScriptedDetector({}), RouteStore(":memory:"), relevance=ScriptedRelevance({}))


def test_core_package_imports_without_torch():
    import subprocess
    import sys
    code = ("import sys; sys.modules['torch'] = None; sys.modules['open_clip'] = None; "
            "import routing.cli, routing.pipeline, services.embedder")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --- VLM agreement against human labels ---------------------------------------------------------


def _res(img, verdict, model="vlm@4b", cand=("c1",)):
    # request ids differ per run/model on purpose: labels must not depend on them
    return img, VerificationResult(request_id=f"vreq-{model}-{img}", class_id=CLASS_A, verdict=verdict,
                                   candidate_ids=cand, reason_code=f"vlm_{verdict}", model_version=model)


def _lab(img, verdict, who, cand=("c1",)):
    return HumanLabel(img, CLASS_A, cand, verdict, who)


def test_vlm_agreement_scores_against_humans_only():
    results = [_res("i1", "absent"), _res("i2", "absent"), _res("i3", "present"), _res("i4", "uncertain"),
               _res("i5", "absent", cand=()), _res("i6", "present")]
    labels = [_lab("i1", "absent", "alice"), _lab("i2", "present", "alice"), _lab("i3", "present", "bob"),
              _lab("i4", "absent", "bob"), _lab("i5", "absent", "bob", cand=())]
    rep = vlm_agreement(results, labels)
    m = rep["by_model"]["vlm@4b"]
    assert rep["labeled"] == 5 and rep["unlabeled"] == 1
    assert m["confusion_vlm|human"] == {"absent|absent": 2, "absent|present": 1, "present|present": 1,
                                        "uncertain|absent": 1}
    assert m["absent_precision"] == round(2 / 3, 4) and m["absent_n"] == 3
    assert 0 < m["absent_precision_wilson_lower95"] < m["absent_precision"]
    assert m["decided_rate"] == 0.8 and m["accuracy_on_decided"] == 0.75
    assert m["by_purpose"]["suspected_missing_class"]["n"] == 1

    with pytest.raises(ValueError, match="not human-authored"):
        vlm_agreement(results, [_lab("i1", "absent", "vlm@4b")])
    with pytest.raises(ValueError, match="conflicting"):
        vlm_agreement(results, [_lab("i1", "absent", "a"), _lab("i1", "present", "b")])


def test_one_label_set_scores_two_vlm_models_from_real_runs(tmp_path):
    """Labels written once must match verifications from runs with different VLMs (request ids differ)."""
    primary = ScriptedDetector({"a": [RawDetection(CLASS_A, (40, 40, 200, 200), 0.97)]})
    rel = ScriptedRelevance({"a": {CLASS_A: 0.4, CLASS_B: 0.1}})
    recs = []
    for name, verdict in (("4b", "absent"), ("8b", "present")):
        vlm = FakeVLM(default=verdict)
        vlm.model_version = f"vlm@{name}"
        recs.append(RoutingPipeline(config(vlm={"enabled": True}), primary, RouteStore(tmp_path / name / "r.sqlite"),
                                    relevance=rel, vlm=vlm).process(_entry(tmp_path)))
    items = verification_items(recs)
    assert len({r.request_id for _, r in items}) == 2  # run-specific ids differ ...
    (cand,) = {tuple(r.candidate_ids) for _, r in items}  # ... but the question is the same
    rep = vlm_agreement(items, [HumanLabel("a", CLASS_A, cand, "present", "alice")])
    assert rep["unlabeled"] == 0
    assert rep["by_model"]["vlm@4b"]["accuracy_on_decided"] == 0.0
    assert rep["by_model"]["vlm@8b"]["accuracy_on_decided"] == 1.0
