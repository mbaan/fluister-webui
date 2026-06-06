"""SQLite persistence for transcription jobs.

Jobs are stored as rows and returned as plain dicts (easy to JSON-serialise for
the API and SSE). Connections are opened per operation with WAL enabled, which
is safe across the request handlers and the background worker thread.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Job lifecycle states.
STATUS_QUEUED = "queued"
STATUS_CONVERTING = "converting"
STATUS_TRANSCRIBING = "transcribing"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_INTERRUPTED = "interrupted"

# Statuses considered "in flight" (used to recover after a restart).
ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_CONVERTING, STATUS_TRANSCRIBING)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                   TEXT PRIMARY KEY,
    original_filename    TEXT NOT NULL,
    stored_path          TEXT NOT NULL,
    wav_path             TEXT,
    msg_timestamp        TEXT,
    msg_timestamp_source TEXT,
    msg_has_time         INTEGER,
    language             TEXT NOT NULL DEFAULT 'auto',
    detected_language    TEXT,
    duration             REAL,
    status               TEXT NOT NULL,
    error                TEXT,
    progress             REAL NOT NULL DEFAULT 0,
    transcript_text      TEXT,
    model_name           TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    started_at           TEXT,
    finished_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at DESC);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def create_job(db_path: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Insert a new job. ``job`` must contain at least id, original_filename,
    stored_path, language, status, model_name. Missing optional columns default
    to NULL/0. Returns the stored row."""
    columns = [
        "id", "original_filename", "stored_path", "wav_path", "msg_timestamp",
        "msg_timestamp_source", "msg_has_time", "language", "detected_language",
        "duration", "status", "error", "progress", "transcript_text",
        "model_name", "created_at", "started_at", "finished_at",
    ]
    row = {c: job.get(c) for c in columns}
    row.setdefault("created_at", now_iso())
    if row["created_at"] is None:
        row["created_at"] = now_iso()
    if row["progress"] is None:
        row["progress"] = 0
    placeholders = ", ".join(f":{c}" for c in columns)
    with _connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO jobs ({', '.join(columns)}) VALUES ({placeholders})", row
        )
    return get_job(db_path, job["id"])  # type: ignore[return-value]


def update_job(db_path: Path, job_id: str, **fields: Any) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{k} = :{k}" for k in fields)
    params = dict(fields)
    params["id"] = job_id
    with _connect(db_path) as conn:
        conn.execute(f"UPDATE jobs SET {assignments} WHERE id = :id", params)


def get_job(db_path: Path, job_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def list_jobs(db_path: Path, limit: int = 200) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]


def delete_job(db_path: Path, job_id: str) -> dict[str, Any] | None:
    job = get_job(db_path, job_id)
    if job is None:
        return None
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return job


def mark_interrupted(db_path: Path) -> list[str]:
    """Flag any jobs still 'in flight' from a previous run as interrupted.
    Returns the affected job ids."""
    placeholders = ", ".join("?" for _ in ACTIVE_STATUSES)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"SELECT id FROM jobs WHERE status IN ({placeholders})", ACTIVE_STATUSES
        )
        ids = [r["id"] for r in cur.fetchall()]
        if ids:
            conn.execute(
                f"UPDATE jobs SET status = ?, error = ? WHERE status IN ({placeholders})",
                (STATUS_INTERRUPTED, "Interrupted by server restart", *ACTIVE_STATUSES),
            )
    return ids
