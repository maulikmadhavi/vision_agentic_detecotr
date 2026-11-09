"""Persistence: SQLite step/record/error store and the per-output-directory job lock."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .schemas import ErrorRecord, RouteRecord


class StoreLockedError(RuntimeError):
    """Another routing job holds the output directory."""


@contextmanager
def job_lock(out_dir: str | Path):
    """Exclusive per-output-directory lock for a writing job (shadow run).

    Uses an OS byte-range lock (``msvcrt`` on Windows, ``flock`` elsewhere), so the
    lock disappears with the process: no stale-lock cleanup and no PID probing.
    Readers (reports, tests reopening a store) do not need it.
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    f = open(p / ".vad-route.lock", "a+", encoding="utf-8")
    try:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise StoreLockedError(f"{p} is in use by another routing job; run one job per output directory") from e
        try:
            yield p
        finally:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


class RouteStore:
    """SQLite store of step results, final records and the error queue."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS steps (record_key TEXT, step TEXT, payload TEXT, PRIMARY KEY (record_key, step));
            CREATE TABLE IF NOT EXISTS records (record_key TEXT PRIMARY KEY, image_id TEXT, status TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS errors (record_key TEXT PRIMARY KEY, payload TEXT);
        """)

    def get_step(self, key: str, step: str):
        row = self.db.execute("SELECT payload FROM steps WHERE record_key=? AND step=?", (key, step)).fetchone()
        return json.loads(row[0]) if row else None

    def put_step(self, key: str, step: str, payload) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO steps VALUES (?,?,?)", (key, step, json.dumps(payload, sort_keys=True)))

    def get_record(self, key: str) -> RouteRecord | None:
        row = self.db.execute("SELECT payload FROM records WHERE record_key=?", (key,)).fetchone()
        return RouteRecord.model_validate_json(row[0]) if row else None

    def put_record(self, rec: RouteRecord) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
                            (rec.record_id, rec.image.id, rec.completeness.status, rec.model_dump_json()))

    def put_error(self, err: ErrorRecord) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO errors VALUES (?,?)", (err.record_id, err.model_dump_json()))

    def records(self) -> list[RouteRecord]:
        return [RouteRecord.model_validate_json(r[0]) for r in self.db.execute("SELECT payload FROM records ORDER BY image_id")]

    def errors(self) -> list[ErrorRecord]:
        return [ErrorRecord.model_validate_json(r[0]) for r in self.db.execute("SELECT payload FROM errors ORDER BY record_key")]
