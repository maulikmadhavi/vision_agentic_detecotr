"""Build and parse constrained VLM verification results; anything unexpected becomes ``uncertain``."""

from __future__ import annotations

import json

from pydantic import ValidationError

from .schemas import VerificationRequest, VerificationResult


def uncertain(request: VerificationRequest, reason_code: str, model_version: str, note: str = "") -> VerificationResult:
    return VerificationResult(
        request_id=request.id, class_id=request.class_id, verdict="uncertain",
        candidate_ids=request.candidate_ids, region_xyxy=request.region_xyxy,
        reason_code=reason_code, model_note=note[:200], model_version=model_version,
    )


def parse_vlm_answer(content: object, request: VerificationRequest, model_version: str) -> VerificationResult:
    try:
        ans = json.loads(content) if isinstance(content, str) else None
        if not isinstance(ans, dict) or set(ans) - {"class_id", "verdict", "reason"}:
            return uncertain(request, "vlm_malformed", model_version, str(content)[:200])
        verdict, cls = ans.get("verdict"), ans.get("class_id")
        if verdict not in ("present", "absent", "uncertain"):
            return uncertain(request, "vlm_malformed", model_version, str(content)[:200])
        if cls != request.class_id:
            return uncertain(request, "vlm_contradictory", model_version, f"answered for class {cls!r}")
        return VerificationResult(
            request_id=request.id, class_id=request.class_id, verdict=verdict,
            candidate_ids=request.candidate_ids, region_xyxy=request.region_xyxy,
            reason_code=f"vlm_{verdict}", model_note=str(ans.get("reason", ""))[:200],
            model_version=model_version,
        )
    except (json.JSONDecodeError, ValidationError) as e:
        return uncertain(request, "vlm_malformed", model_version, f"{type(e).__name__}")
