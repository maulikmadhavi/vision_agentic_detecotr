"""Compare run-metrics-v1 files of one series (`vad-route compare`).

The last file is the candidate. Its baseline is the file before it (``previous``) or, with a
guardrails file, the earlier run with the highest coco mAP50-95 among those that pass the
guardrails against the first file, the series reference (``best``). Deltas are in percentage
points for both evaluators. Only the custom-vad-1 delta has an interval: a seeded paired bootstrap
over source groups, recomputed from the stored per-image outcomes (whole groups resampled, dataset AP
recomputed; per-image AP is never averaged). A point estimate whose interval spans zero is "no
detectable change".

The series key (cohort, task class ids, working IoU, coco settings, image/label/cohort digests,
evaluator versions) must match; development runs therefore never compare with frozen_eval runs, and
parent linkage never authorizes a comparison. Only a content-verified series (series.images_sha256
set) gets an interval, a best run and a verdict; otherwise the output is a descriptive table marked
NOT A FINDING. The selected source is reported with primary beside it, plus unscored_routed and
accepted-annotation precision/recall. With min_meaningful_gain_pp and patience in the guardrails file
a plateau is reported as an operational stop rule only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .metrics import map_from_outcomes

SOURCES = {"primary": ("detector",), "routed_all": ("routed", "all_unrejected"),
           "routed_accepted": ("routed", "accepted_only")}
GUARDRAIL_KEYS = {"version", "max_recall_drop_pp", "max_fp_per_image_increase", "max_review_rate_increase_pp",
                  "max_vlm_calls_per_image_increase", "max_latency_p90_increase_ms"}
PLATEAU_KEYS = {"min_meaningful_gain_pp", "patience"}  # optional, together: the operational stop rule
PLATEAU_RULE = ("plateau: stop tuning when `patience` consecutive runs gain less than min_meaningful_gain_pp coco "
                "mAP50-95 over the best run so far; an operational rule, not evidence of convergence")
DESCRIPTIVE = ("NOT A FINDING: descriptive comparison; the series is not content-verified (no verified split binds "
               "the images), so no interval and no ranking")
METRICS = ("map50_95", "map50")


class CompareError(ValueError):
    """The files cannot be compared as asked."""


def _flatten(value, prefix: str = "") -> dict:
    if isinstance(value, Mapping):
        return {k: v for key, item in value.items() for k, v in _flatten(item, f"{prefix}{key}.").items()}
    return {prefix.rstrip("."): value}


def _differences(a: Mapping, b: Mapping) -> list[str]:
    fa, fb = _flatten(a), _flatten(b)
    return sorted(k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k))


def _block(doc: Mapping, source: str) -> Mapping:
    block = doc
    for key in SOURCES[source]:
        block = block[key]
    return block


def _pp(new: float | None, old: float | None) -> float | None:
    return None if new is None or old is None else 100.0 * (new - old)


def _classes(docs: Sequence[Mapping], source: str) -> list[str]:
    return sorted({c for d in docs for o in d["per_image"].values() for c in o[source]["classes"]})


def _recall_and_fp(doc: Mapping, source: str) -> tuple[dict[str, float | None], float | None]:
    """Recall per class and false positives per image at IoU 0.50, from the stored outcomes of ``source``."""
    found: dict[str, list[int]] = {}
    fp = 0
    for outcome in doc["per_image"].values():
        for c, entry in outcome[source]["classes"].items():
            counts = found.setdefault(c, [0, 0])
            counts[0] += sum(1 for _, codes in entry["ranked"] if codes[0] == 1)
            counts[1] += entry["gt"]
            fp += sum(1 for _, codes in entry["ranked"] if codes[0] == 0)
    images = len(doc["per_image"])
    return {c: (f / n if n else None) for c, (f, n) in found.items()}, (fp / images if images else None)


def guardrail_failures(run: Mapping, baseline: Mapping, rules: Mapping, source: str) -> list[str]:
    """The guardrails ``run`` breaks against ``baseline``; a value that cannot be computed fails its rule."""
    failures: list[str] = []

    def check(name: str, value: float | None, limit: float) -> None:
        if value is None:
            failures.append(f"{name}: value unavailable")
        elif value > limit:
            failures.append(f"{name}: {value:.4g} > {limit}")

    recall_run, fp_run = _recall_and_fp(run, source)
    recall_base, fp_base = _recall_and_fp(baseline, source)
    for c in sorted(recall_base):
        if recall_base[c] is not None:
            check(f"max_recall_drop_pp.per_class[{c}]", _pp(recall_base[c], recall_run.get(c)),
                  rules["max_recall_drop_pp"]["per_class"])
    for size, base in baseline["detector"]["recall_by_size"].items():  # sizes are stored for the primary detector only
        if base["recall"] is not None:
            check(f"max_recall_drop_pp.per_size[{size}] (primary detector)",
                  _pp(base["recall"], run["detector"]["recall_by_size"][size]["recall"]),
                  rules["max_recall_drop_pp"]["per_size"])
    check("max_fp_per_image_increase", None if fp_run is None or fp_base is None else fp_run - fp_base,
          rules["max_fp_per_image_increase"])
    cost, base_cost = run["cost"], baseline["cost"]
    check("max_review_rate_increase_pp", _pp(cost["review_rate"], base_cost["review_rate"]),
          rules["max_review_rate_increase_pp"])
    for name, key in (("max_vlm_calls_per_image_increase", "vlm_calls_per_image"),):
        check(name, None if cost[key] is None or base_cost[key] is None else cost[key] - base_cost[key], rules[name])
    p90, base_p90 = cost["latency_ms"]["p90"], base_cost["latency_ms"]["p90"]
    check("max_latency_p90_increase_ms", None if p90 is None or base_p90 is None else p90 - base_p90,
          rules["max_latency_p90_increase_ms"])
    return failures


def _interval(baseline: Mapping, candidate: Mapping, source: str, seed: int, resamples: int,
              verified: bool) -> dict | str:
    docs = (baseline, candidate)
    if not verified:
        return "unavailable: the series is not content-verified (no verified split binds the images)"
    if any(o["group_id"] is None for d in docs for o in d["per_image"].values()):
        return "unavailable: no group ids in a file"
    if any(_block(d, source)["custom_map50_95"] is None for d in docs):
        return "unavailable: the custom AP of this source is unavailable in a file"
    groups: dict[str, list[str]] = {}
    for image_id, outcome in sorted(baseline["per_image"].items()):
        groups.setdefault(outcome["group_id"], []).append(image_id)
    names, classes = sorted(groups), _classes(docs, source)
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(resamples):
        images = [i for g in rng.integers(0, len(names), len(names)) for i in groups[names[g]]]
        old = map_from_outcomes([baseline["per_image"][i][source] for i in images], classes)
        new = map_from_outcomes([candidate["per_image"][i][source] for i in images], classes)
        for m in METRICS:
            if (d := _pp(new[m], old[m])) is not None:
                deltas[m].append(d)
    return {**{f"{m}_pp": [float(np.percentile(deltas[m], 2.5)), float(np.percentile(deltas[m], 97.5))]
               for m in METRICS},
            "method": "paired group bootstrap, percentile 2.5-97.5, on custom-vad-1 matching",
            "resamples": resamples, "seed": seed, "groups": len(names)}


def compare_runs(docs: Sequence[Mapping], *, source: str, against: str, guardrails: Mapping | None = None,
                 seed: int = 0, resamples: int = 1000) -> dict:
    """Compare the last run with its baseline; ``docs`` are verified run-metrics-v1 documents in order."""
    if len(docs) < 2:
        raise CompareError("compare needs at least two run-metrics files (the last one is the candidate)")
    for d in docs:
        if d["mixed_runs"]:
            fields = sorted(k for k in set().union(*d["runs"]) if len({str(r.get(k)) for r in d["runs"]}) > 1)
            raise CompareError(f"run-metrics {d['metrics_id']} mixes runs differing in {fields}; "
                               "evaluate each run's trace separately")
    for d in docs[1:]:
        if diff := _differences(docs[0]["series"], d["series"]):
            raise CompareError(f"not the same series: {d['metrics_id']} differs from {docs[0]['metrics_id']} in "
                               f"series fields {diff}")
    if guardrails is not None:
        extra = set(guardrails) - GUARDRAIL_KEYS
        if (guardrails.get("version") != "guardrails-v1" or not GUARDRAIL_KEYS <= set(guardrails)
                or extra not in (set(), PLATEAU_KEYS)):
            raise CompareError(f"--guardrails must be a guardrails-v1 file with exactly {sorted(GUARDRAIL_KEYS)}, "
                               f"optionally with both of {sorted(PLATEAU_KEYS)}")
    # Only a content-verified series (images bound by a verified split; mixed runs are refused above)
    # gets intervals, a best run and a verdict; anything else is a descriptive table, NOT A FINDING.
    verified = docs[0]["series"]["images_sha256"] is not None
    candidate, earlier = docs[-1], docs[:-1]
    report = None
    if against == "best":
        if not verified:
            raise CompareError("--against best needs a content-verified series (series.images_sha256 set by a verified "
                               "split); this series can only be compared descriptively")
        if guardrails is None:
            raise CompareError("--against best needs --guardrails: eligibility is judged by the guardrails file")
        runs = [{"metrics_id": d["metrics_id"],
                 "failed": [] if d is docs[0] else guardrail_failures(d, docs[0], guardrails, source)} for d in earlier]
        report = [{**r, "eligible": not r["failed"]} for r in runs]
        eligible = [d for d, r in zip(earlier, report) if r["eligible"]]
        baseline = max(eligible, key=lambda d: _block(d, source)["coco"]["map50_95"] or float("-inf"))
    else:
        baseline = earlier[-1]
    deltas = _deltas(baseline, candidate, source)
    coco, custom = deltas["coco"], deltas["custom"]
    custom["interval"] = _interval(baseline, candidate, source, seed, resamples, verified)
    if source != "primary":
        deltas["unscored_routed"] = {"baseline": _block(baseline, source)["unscored_routed"],
                                     "candidate": _block(candidate, source)["unscored_routed"]}
    if not verified:
        verdict = DESCRIPTIVE
    elif isinstance(custom["interval"], str):
        verdict = "interval unavailable: not ranked"
    else:
        low, high = custom["interval"]["map50_95_pp"]
        verdict = "no detectable change" if low <= 0 <= high else ("improved" if low > 0 else "worse")
    warnings = [f"coco and custom-vad-1 deltas have opposite signs on {m} ({coco[f'{m}_pp']:+.2f} vs "
                f"{custom[f'{m}_pp']:+.2f} pp); check the 100-detection cap and crowd/ignore matching"
                for m in METRICS
                if coco[f"{m}_pp"] is not None and custom[f"{m}_pp"] is not None and coco[f"{m}_pp"] * custom[f"{m}_pp"] < 0]
    guard = None
    if guardrails is not None:
        failed = guardrail_failures(candidate, baseline, guardrails, source)
        guard = {"version": guardrails["version"], "candidate": {"passed": not failed, "failed": failed},
                 **({"runs": report} if report is not None else {})}
    accepted = {"baseline": {k: baseline["accepted_annotations"][k] for k in ("precision", "recall")},
                "candidate": {k: candidate["accepted_annotations"][k] for k in ("precision", "recall")}}
    accepted.update({f"{k}_pp": _pp(accepted["candidate"][k], accepted["baseline"][k]) for k in ("precision", "recall")})
    return {"version": "compare-v1", "status": "content-verified" if verified else DESCRIPTIVE, "source": source,
            "against": against, "candidate": candidate["metrics_id"], "baseline": baseline["metrics_id"],
            "series": docs[0]["series"], "deltas": deltas,
            **({"primary": _deltas(baseline, candidate, "primary")} if source != "primary" else {}),
            "accepted_annotations": accepted, "verdict": verdict, "warnings": warnings, "guardrails": guard,
            "plateau": _plateau(docs, source, guardrails)}


def _deltas(baseline: Mapping, candidate: Mapping, source: str) -> dict:
    """coco and custom-vad-1 deltas (percentage points) of one prediction source; no interval here."""
    old, new = _block(baseline, source), _block(candidate, source)
    return {"coco": {f"{m}_pp": _pp(new["coco"][m], old["coco"][m]) for m in METRICS},
            "custom": {f"{m}_pp": _pp(new[f"custom_{m}"], old[f"custom_{m}"]) for m in METRICS}}


def _plateau(docs: Sequence[Mapping], source: str, guardrails: Mapping | None) -> dict | None:
    if guardrails is None or not PLATEAU_KEYS <= set(guardrails):
        return None
    need, patience = guardrails["min_meaningful_gain_pp"], guardrails["patience"]
    values = [_block(d, source)["coco"]["map50_95"] for d in docs]
    best, since = values[0], 0
    for value in values[1:]:
        if value is not None and (best is None or 100.0 * (value - best) >= need):
            best, since = value, 0
        else:
            since += 1
            if value is not None and best is not None:
                best = max(best, value)
    return {"stop_tuning": since >= patience, "runs_without_gain": since, "min_meaningful_gain_pp": need,
            "patience": patience, "rule": PLATEAU_RULE}
