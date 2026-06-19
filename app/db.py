"""SQLite persistence for transcription jobs.

Jobs are stored as rows and returned as plain dicts (easy to JSON-serialise for
the API and SSE). Connections are opened per operation with WAL enabled, which
is safe across the request handlers and the background worker thread.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Whether this SQLite build has FTS5 (set by init_db). When false, full-text
# search falls back to a LIKE scan. Snippet matches are wrapped in these private
# -use markers so the web UI can render them as <mark> regardless of path.
_FTS_OK = False
_SNIPPET_OPEN = ""
_SNIPPET_CLOSE = ""

# Job lifecycle states.
STATUS_QUEUED = "queued"
STATUS_CONVERTING = "converting"
STATUS_TRANSCRIBING = "transcribing"
STATUS_TIDYING = "tidying"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_INTERRUPTED = "interrupted"

# Statuses considered "in flight" (used to recover after a restart). TIDYING is
# deliberately absent: a job interrupted mid-tidy has its full transcript
# persisted already, so recovery completes it as DONE instead (see
# mark_interrupted) rather than flagging it as a failure.
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
    global _FTS_OK
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        _FTS_OK = _init_fts(conn)


def _init_fts(conn: sqlite3.Connection) -> bool:
    """Create the full-text index (if this SQLite build has FTS5) and rebuild it
    from the jobs table, so the index always matches on startup. Returns whether
    FTS5 is available; when it isn't, search falls back to a LIKE scan."""
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts "
            "USING fts5(job_id UNINDEXED, filename, body)"
        )
    except sqlite3.OperationalError:
        return False
    # Rebuild from jobs. Guard transcript_text: a hand-migrated old DB may not
    # have it yet (it's added elsewhere), in which case index filenames only.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    body = "COALESCE(transcript_text, '')" if "transcript_text" in cols else "''"
    conn.execute("DELETE FROM jobs_fts")
    conn.execute(
        f"INSERT INTO jobs_fts(job_id, filename, body) "
        f"SELECT id, original_filename, {body} FROM jobs"
    )
    return True


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
        if _FTS_OK:
            conn.execute(
                "INSERT INTO jobs_fts(job_id, filename, body) VALUES (?, ?, ?)",
                (row["id"], row["original_filename"], ""),
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
        # Keep the search index current only when the indexed text actually
        # changes (not on every progress tick).
        if _FTS_OK and ("transcript_text" in fields or "original_filename" in fields):
            _reindex_job(conn, job_id)


def _reindex_job(conn: sqlite3.Connection, job_id: str) -> None:
    """Refresh one job's FTS row from its current filename + transcript."""
    r = conn.execute(
        "SELECT original_filename, transcript_text FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    conn.execute("DELETE FROM jobs_fts WHERE job_id = ?", (job_id,))
    if r:
        conn.execute(
            "INSERT INTO jobs_fts(job_id, filename, body) VALUES (?, ?, ?)",
            (job_id, r["original_filename"], r["transcript_text"] or ""),
        )


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
        if _FTS_OK:
            conn.execute("DELETE FROM jobs_fts WHERE job_id = ?", (job_id,))
    return job


def find_duplicate(
    db_path: Path, original_filename: str, size: int
) -> dict[str, Any] | None:
    """Return an existing non-failed job with the same filename + size, if any
    (used to skip re-transcribing a file that was already uploaded)."""
    statuses = (
        STATUS_QUEUED, STATUS_CONVERTING, STATUS_TRANSCRIBING,
        STATUS_TIDYING, STATUS_DONE,
    )
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
        n = conn.execute("DELETE FROM jobs").rowcount
        if _FTS_OK:
            conn.execute("DELETE FROM jobs_fts")
        return n


def mark_interrupted(db_path: Path) -> list[str]:
    """Flag any jobs still 'in flight' from a previous run as interrupted.
    Returns the affected job ids.

    Jobs caught mid-tidy are completed as DONE instead: their transcript was
    fully persisted before the tidy pass started, only the optional readable
    view is missing."""
    placeholders = ", ".join("?" for _ in ACTIVE_STATUSES)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, progress = 1.0 WHERE status = ?",
            (STATUS_DONE, STATUS_TIDYING),
        )
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


# ── full-text search ─────────────────────────────────────────────────────────
def _fts_match_expr(query: str) -> str:
    """Turn a free-text query into a safe FTS5 MATCH expression: each word
    becomes a prefix term (implicit AND). Strips operators that could break the
    parser. Returns "" when there's nothing searchable."""
    tokens = re.findall(r"\w+", query, re.UNICODE)
    return " ".join(t + "*" for t in tokens)


def _like_snippet(text: str, q: str, radius: int = 64) -> str:
    """A small context window around the first match, with the match wrapped in
    the shared marker chars (used by the LIKE fallback so the UI renders the
    same way as the FTS path)."""
    if not text:
        return ""
    i = text.lower().find(q.lower())
    if i < 0:
        s = text.strip()
        return (s[:140] + "…") if len(s) > 140 else s
    start, end = max(0, i - radius), min(len(text), i + len(q) + radius)
    return (
        ("…" if start > 0 else "") + text[start:i]
        + _SNIPPET_OPEN + text[i:i + len(q)] + _SNIPPET_CLOSE
        + text[i + len(q):end] + ("…" if end < len(text) else "")
    )


def search_jobs(db_path: Path, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Full-text search across job filenames + transcripts. Returns
    [{job_id, filename, snippet}] ranked by relevance (bm25 under FTS5, recency
    under the LIKE fallback). Snippet matches are wrapped in marker chars."""
    q = (query or "").strip()
    if not q:
        return []
    with _connect(db_path) as conn:
        if _FTS_OK:
            expr = _fts_match_expr(q)
            if expr:
                try:
                    rows = conn.execute(
                        "SELECT j.id AS id, j.original_filename AS filename, "
                        "snippet(jobs_fts, 2, char(57344), char(57345), '…', 14) AS snip "
                        "FROM jobs_fts JOIN jobs j ON j.id = jobs_fts.job_id "
                        "WHERE jobs_fts MATCH ? ORDER BY bm25(jobs_fts) LIMIT ?",
                        (expr, limit),
                    ).fetchall()
                    return [
                        {"job_id": r["id"], "filename": r["filename"], "snippet": r["snip"] or ""}
                        for r in rows
                    ]
                except sqlite3.OperationalError:
                    pass  # malformed MATCH or no FTS — fall through to LIKE
        like = f"%{q}%"
        rows = conn.execute(
            "SELECT id, original_filename AS filename, transcript_text FROM jobs "
            "WHERE transcript_text LIKE ? OR original_filename LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (like, like, limit),
        ).fetchall()
        return [
            {"job_id": r["id"], "filename": r["filename"],
             "snippet": _like_snippet(r["transcript_text"] or "", q)}
            for r in rows
        ]


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
