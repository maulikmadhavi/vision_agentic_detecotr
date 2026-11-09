"""Human review loop (reviewer work order A7; board T-7).

Stage 1: the append-only adjudications log. Every human decision is an
:class:`~routing.schemas.Adjudication`, written as one JSON line. The file is only ever
appended to (never rewritten), so the history of decisions is replayable; reading verifies
every line (including its content-derived id) and drops exact repeats.
Model suggestions stay in the routed records; this log holds human decisions only.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from .schemas import Adjudication


class AdjudicationLogError(ValueError):
    """A line of the adjudications log is not a valid Adjudication."""


def append_adjudications(path: str | Path, adjudications: Iterable[Adjudication]) -> None:
    """Append decisions to the log, one JSON line each; earlier lines are never touched."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for adjudication in adjudications:
            f.write(adjudication.model_dump_json() + "\n")


def read_adjudications(path: str | Path) -> list[Adjudication]:
    """All decisions in log order; a missing log is empty; exact repeats (same id) are read once."""
    p = Path(path)
    if not p.exists():
        return []
    seen: set[str] = set()
    out: list[Adjudication] = []
    for number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            adjudication = Adjudication.model_validate_json(line)
        except ValidationError as e:
            raise AdjudicationLogError(f"{p}: line {number}: {e}") from None
        if adjudication.id not in seen:
            seen.add(adjudication.id)
            out.append(adjudication)
    return out
