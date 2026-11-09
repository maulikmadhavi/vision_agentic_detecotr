"""A2 stage 1: near-duplicate grouping and the exclusions file (synthetic images only)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from PIL import Image

from routing import cli
from routing.sample import (
    EXCLUSION_REASONS, Fingerprint, SampleError, exclusions_digest, fingerprint, group_images, hamming, load_exclusions,
)


def _pattern(seed: int, size=(256, 192)) -> Image.Image:
    """Smooth random structure (an 8x6 grid upscaled), so distinct images differ at the hash scale."""
    grid = np.random.default_rng(seed).integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(grid).resize(size, Image.Resampling.BILINEAR)


@pytest.fixture
def images(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    _pattern(1).save(root / "a.png")
    _pattern(1).resize((128, 96)).save(root / "a_half.png")                 # resized copy
    _pattern(1).resize((512, 384)).save(root / "a_double.png")
    _pattern(1).convert("RGB").save(root / "a_q40.jpg", quality=40)         # recompressed copy
    shutil.copy(root / "a.png", root / "a_bytes.png")                        # byte-identical copy
    _pattern(1).resize((128, 96)).convert("RGB").save(root / "a_small_q5.jpg", quality=5)  # resized + recompressed
    _pattern(1).crop((16, 12, 240, 180)).save(root / "a_crop6.png")       # cropped: a near miss, not merged
    _pattern(2).save(root / "b.png")
    _pattern(3).save(root / "c.png")
    _pattern(4).save(root / "d.png")
    (root / "broken.png").write_bytes(b"not an image")
    rows = [{"uri": name, "image_id": name.split(".")[0]} for name in
            ("a.png", "a_half.png", "a_double.png", "a_q40.jpg", "a_bytes.png", "a_small_q5.jpg", "a_crop6.png",
             "b.png", "c.png", "d.png", "broken.png")]
    rows[8]["group_id"] = "source-7"  # c and d come from the same source page
    rows[9]["group_id"] = "source-7"
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root, manifest


# --- fingerprints -------------------------------------------------------------------------------


def test_resized_and_recompressed_copies_stay_within_the_threshold(images):
    root, _ = images
    base = fingerprint(root / "a.png")
    for copy in ("a_half.png", "a_double.png", "a_q40.jpg", "a_bytes.png", "a_small_q5.jpg"):
        assert hamming(base.dhash, fingerprint(root / copy).dhash) <= 6, copy
    assert hamming(base.dhash, fingerprint(root / "a_small_q5.jpg").dhash) > 0  # a real, non-zero tolerance


def test_distinct_images_are_far_apart(images):
    root, _ = images
    hashes = [fingerprint(root / f"{n}.png").dhash for n in "abcd"]
    assert min(hamming(x, y) for i, x in enumerate(hashes) for y in hashes[i + 1:]) > 10


def test_hamming_counts_differing_bits():
    assert (hamming(0b1011, 0b0001), hamming(0, 2 ** 64 - 1)) == (2, 64)


# --- grouping (pure) ----------------------------------------------------------------------------


def _fp(sha, dhash):
    return Fingerprint(sha256=sha, dhash=dhash)


def test_threshold_merges_at_six_and_lists_near_misses_without_merging():
    fps = {"x": _fp("1", 0), "six": _fp("2", 0b111111), "seven": _fp("3", 0b1111111 << 20), "far": _fp("4", 2 ** 64 - 1)}
    out = group_images(fps, {}, max_hamming=6)
    assert out.groups["x"] == out.groups["six"] and len({out.groups[k] for k in fps}) == 3
    pairs = {(p["a"], p["b"]): p for p in out.pairs}
    assert pairs[("six", "x")] == {"a": "six", "b": "x", "reason": "near_duplicate", "distance": 6, "merged": True}
    assert pairs[("seven", "x")]["merged"] is False and pairs[("seven", "x")]["distance"] == 7
    assert not any("far" in key for key in pairs)  # beyond the review margin: not listed


def test_exact_bytes_and_same_source_merge_transitively():
    fps = {"p": _fp("same", 0), "q": _fp("same", 2 ** 64 - 1), "r": _fp("r", 0x0F0F0F0F),
           "s": _fp("s", 0xFFFFFFFF00000000)}  # s is at least 32 bits from every other hash
    out = group_images(fps, {"q": "page-1", "r": "page-1"}, max_hamming=6)
    assert out.groups["p"] == out.groups["q"] == out.groups["r"] != out.groups["s"]
    assert {"a": "p", "b": "q", "reason": "exact", "distance": 64, "merged": True} in out.pairs


def test_group_ids_do_not_depend_on_input_order():
    fps = {"a": _fp("1", 0), "b": _fp("2", 1), "c": _fp("3", 2 ** 64 - 1)}
    reordered = dict(reversed(list(fps.items())))
    assert group_images(fps, {}).groups == group_images(reordered, {}).groups


# --- grouping through the CLI -------------------------------------------------------------------


def test_cli_sample_group_writes_groups_pairs_and_unreadable(images, tmp_path):
    root, manifest = images
    out = tmp_path / "groups.json"
    assert cli.main(["sample", "group", "--images", str(manifest), "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    group = {i: v["group_id"] for i, v in doc["images"].items()}
    assert group["a"] == group["a_half"] == group["a_double"] == group["a_q40"] == group["a_bytes"] == group["a_small_q5"]
    near_miss = next(p for p in doc["pairs"] if p["a"] == "a" and p["b"] == "a_crop6")
    assert group["a_crop6"] != group["a"] and near_miss["merged"] is False and 6 < near_miss["distance"] <= 10
    assert group["c"] == group["d"] and len({group["a"], group["b"], group["c"]}) == 3
    assert doc["max_hamming"] == 6 and doc["unreadable"] == ["broken"]
    reasons = {(p["a"], p["b"]): p["reason"] for p in doc["pairs"]}
    assert reasons[("a", "a_bytes")] == "exact" and reasons[("a", "a_q40")] == "near_duplicate"


# --- exclusions ---------------------------------------------------------------------------------


IDS = ("a", "b", "c")


def _excl(*rows, recorded_by="rev-1"):
    doc = {"vad_exclusions": 1, "exclusions": [dict(r) for r in rows]}
    if recorded_by is not None:
        doc["recorded_by"] = recorded_by
    return doc


def test_exclusions_load_with_reasons_and_out_of_domain_flag():
    out = load_exclusions(_excl({"image_id": "a", "reason": "screenshot"},
                                {"image_id": "b", "reason": "illustration", "out_of_domain_set": True}), IDS)
    assert [(e.image_id, e.reason, e.out_of_domain_set) for e in out] == [("a", "screenshot", False),
                                                                          ("b", "illustration", True)]
    assert set(EXCLUSION_REASONS) >= {"illustration", "collage", "screenshot", "duplicate", "license_unclear", "other"}


@pytest.mark.parametrize("doc,match", [
    (_excl({"image_id": "zz", "reason": "screenshot"}), "zz"),
    (_excl({"image_id": "a", "reason": "screenshot"}, {"image_id": "a", "reason": "collage"}), "twice"),
    (_excl({"image_id": "a", "reason": "blurry"}), "blurry"),
    (_excl({"image_id": "a", "reason": "other"}), "note"),
    (_excl({"image_id": "a", "reason": "collage"}, recorded_by=None), "recorded_by"),
    ({"exclusions": []}, "vad_exclusions"),
])
def test_invalid_exclusions_are_rejected(doc, match):
    with pytest.raises(SampleError, match=match):
        load_exclusions(doc, IDS)


def test_exclusions_digest_ignores_formatting_but_not_content():
    doc = _excl({"image_id": "a", "reason": "collage"})
    same = json.loads(json.dumps(doc, indent=4, sort_keys=True))
    changed = _excl({"image_id": "a", "reason": "screenshot"})
    assert exclusions_digest(doc) == exclusions_digest(same) != exclusions_digest(changed)
