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
    finished_at          TEXT,
    diarized             INTEGER NOT NULL DEFAULT 0,
    speakers             TEXT,
    size                 INTEGER,
    segments_json        TEXT,
    tidied_json          TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at DESC);

CREATE TABLE IF NOT EXISTS persons (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    centroid   BLOB,
    n_samples  INTEGER NOT NULL DEFAULT 0,
    dim        INTEGER,
    keywords   TEXT
);

CREATE TABLE IF NOT EXISTS person_embeddings (
    id         TEXT PRIMARY KEY,
    person_id  TEXT NOT NULL,
    job_id     TEXT,
    embedding  BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pe_person ON person_embeddings (person_id);
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
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    pcols = {r["name"] for r in conn.execute("PRAGMA table_info(persons)").fetchall()}
    if "keywords" not in pcols:
        conn.execute("ALTER TABLE persons ADD COLUMN keywords TEXT")

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "diarized" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN diarized INTEGER NOT NULL DEFAULT 0")
    if "speakers" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN speakers TEXT")
    if "segments_json" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN segments_json TEXT")
    if "tidied_json" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN tidied_json TEXT")
    if "size" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN size INTEGER")
        # Backfill size from existing upload files so older jobs also dedupe.
        for r in conn.execute("SELECT id, stored_path FROM jobs").fetchall():
            sp = r["stored_path"]
            try:
                if sp and Path(sp).is_file():
                    conn.execute(
                        "UPDATE jobs SET size = ? WHERE id = ?",
                        (Path(sp).stat().st_size, r["id"]),
                    )
            except OSError:
                pass


def create_job(db_path: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Insert a new job. ``job`` must contain at least id, original_filename,
    stored_path, language, status, model_name. Missing optional columns default
    to NULL/0. Returns the stored row."""
    columns = [
        "id", "original_filename", "stored_path", "wav_path", "msg_timestamp",
        "msg_timestamp_source", "msg_has_time", "language", "detected_language",
        "duration", "status", "error", "progress", "transcript_text",
        "model_name", "created_at", "started_at", "finished_at", "size",
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


def find_duplicate(
    db_path: Path, original_filename: str, size: int
) -> dict[str, Any] | None:
    """Return an existing non-failed job with the same filename + size, if any
    (used to skip re-transcribing a file that was already uploaded)."""
    statuses = (STATUS_QUEUED, STATUS_CONVERTING, STATUS_TRANSCRIBING, STATUS_DONE)
    placeholders = ", ".join("?" for _ in statuses)
    with _connect(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM jobs WHERE original_filename = ? AND size = ? "
            f"AND status IN ({placeholders}) ORDER BY created_at LIMIT 1",
            (original_filename, size, *statuses),
        ).fetchone()
    return dict(row) if row else None


def clear_jobs(db_path: Path) -> int:
    """Delete every job row (persons / voice gallery are left intact).
    Returns the number of jobs removed."""
    with _connect(db_path) as conn:
        return conn.execute("DELETE FROM jobs").rowcount


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


# ── persons / global voice gallery ──────────────────────────────────────────
_PERSON_COLS = ["id", "name", "created_at", "centroid", "n_samples", "dim", "keywords"]


def create_person(db_path: Path, person: dict[str, Any]) -> dict[str, Any]:
    row = {c: person.get(c) for c in _PERSON_COLS}
    if not row.get("created_at"):
        row["created_at"] = now_iso()
    if row.get("n_samples") is None:
        row["n_samples"] = 0
    placeholders = ", ".join(f":{c}" for c in _PERSON_COLS)
    with _connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO persons ({', '.join(_PERSON_COLS)}) VALUES ({placeholders})",
            row,
        )
    return get_person(db_path, person["id"])  # type: ignore[return-value]


def get_person(db_path: Path, person_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    return dict(row) if row else None


def list_persons(db_path: Path) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        cur = conn.execute("SELECT * FROM persons ORDER BY created_at ASC")
        return [dict(r) for r in cur.fetchall()]


def update_person(db_path: Path, person_id: str, **fields: Any) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{k} = :{k}" for k in fields)
    params = dict(fields)
    params["id"] = person_id
    with _connect(db_path) as conn:
        conn.execute(f"UPDATE persons SET {assignments} WHERE id = :id", params)


def delete_person(db_path: Path, person_id: str) -> dict[str, Any] | None:
    person = get_person(db_path, person_id)
    if person is None:
        return None
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM person_embeddings WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    return person


def add_person_embedding(db_path: Path, row: dict[str, Any]) -> None:
    cols = ["id", "person_id", "job_id", "embedding", "created_at"]
    r = {c: row.get(c) for c in cols}
    if not r.get("created_at"):
        r["created_at"] = now_iso()
    placeholders = ", ".join(f":{c}" for c in cols)
    with _connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO person_embeddings ({', '.join(cols)}) VALUES ({placeholders})",
            r,
        )


def list_person_embeddings(db_path: Path, person_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM person_embeddings WHERE person_id = ? ORDER BY created_at ASC",
            (person_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def reassign_embeddings(db_path: Path, src_person_id: str, dst_person_id: str) -> None:
    """Move all of src's voice samples to dst (used when merging persons)."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE person_embeddings SET person_id = ? WHERE person_id = ?",
            (dst_person_id, src_person_id),
        )


def delete_job_embeddings(db_path: Path, person_id: str, job_id: str) -> None:
    """Drop a (person, job) voice sample so re-diarizing a job replaces rather
    than appends its contribution."""
    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM person_embeddings WHERE person_id = ? AND job_id = ?",
            (person_id, job_id),
        )


def persons_with_embeddings_for_job(db_path: Path, job_id: str) -> list[str]:
    """Person ids that have at least one voice sample from ``job_id``."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT DISTINCT person_id FROM person_embeddings WHERE job_id = ?",
            (job_id,),
        )
        return [r["person_id"] for r in cur.fetchall()]


def delete_embeddings_for_job(db_path: Path, job_id: str) -> None:
    """Drop every voice sample contributed by ``job_id`` (across all persons)."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM person_embeddings WHERE job_id = ?", (job_id,))
