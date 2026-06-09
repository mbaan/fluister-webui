"""tidied_json column exists, migrates onto an old DB, and round-trips."""
import sqlite3

from app import db


def test_fresh_db_has_tidied_json(tmp_path):
    p = tmp_path / "t.db"
    db.init_db(p)
    job = db.create_job(p, {
        "id": "j1", "original_filename": "a.m4a", "stored_path": "/x/a.m4a",
        "language": "auto", "status": db.STATUS_QUEUED, "model_name": "large-v3",
    })
    assert "tidied_json" in job
    assert job["tidied_json"] is None
    db.update_job(p, "j1", tidied_json='[{"speaker": null, "text": "Hi."}]')
    assert db.get_job(p, "j1")["tidied_json"] == '[{"speaker": null, "text": "Hi."}]'


def test_migration_adds_column_to_old_db(tmp_path):
    p = tmp_path / "old.db"
    # Minimal pre-tidied jobs table (no tidied_json).
    conn = sqlite3.connect(p)
    conn.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, original_filename TEXT NOT NULL, "
        "stored_path TEXT NOT NULL, language TEXT NOT NULL DEFAULT 'auto', "
        "status TEXT NOT NULL, model_name TEXT NOT NULL, created_at TEXT NOT NULL, "
        "progress REAL NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO jobs (id, original_filename, stored_path, status, model_name, created_at) "
        "VALUES ('old1', 'o.m4a', '/x/o.m4a', 'done', 'large-v3', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    db.init_db(p)  # runs _migrate
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    assert "tidied_json" in cols
    db.update_job(p, "old1", tidied_json="[]")
    assert db.get_job(p, "old1")["tidied_json"] == "[]"
