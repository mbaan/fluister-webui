"""Full-text search: index stays in sync with the jobs table, queries match
filenames + transcripts, snippets carry markers, and the LIKE fallback works."""
from app import db


def _job(p, jid, name, text=None):
    db.create_job(p, {
        "id": jid, "original_filename": name, "stored_path": f"/x/{name}",
        "language": "auto", "status": db.STATUS_QUEUED, "model_name": "large-v3",
    })
    if text is not None:
        db.update_job(p, jid, transcript_text=text, status=db.STATUS_DONE)


def test_search_matches_transcript_body(tmp_path):
    p = tmp_path / "t.db"; db.init_db(p)
    _job(p, "a", "interview.m4a", "We discussed the quarterly budget and hiring plans.")
    _job(p, "b", "other.m4a", "Nothing relevant here.")
    hits = db.search_jobs(p, "budget")
    assert [h["job_id"] for h in hits] == ["a"]


def test_search_is_prefix(tmp_path):
    p = tmp_path / "t.db"; db.init_db(p)
    _job(p, "a", "x.m4a", "a strategic planning session")
    assert any(h["job_id"] == "a" for h in db.search_jobs(p, "plan"))


def test_search_matches_filename(tmp_path):
    p = tmp_path / "t.db"; db.init_db(p)
    _job(p, "a", "budget-call.m4a", "hello world")
    assert any(h["job_id"] == "a" for h in db.search_jobs(p, "budget"))


def test_snippet_wraps_match_in_markers(tmp_path):
    p = tmp_path / "t.db"; db.init_db(p)
    _job(p, "a", "x.m4a", "the budget meeting ran long today")
    hits = db.search_jobs(p, "budget")
    assert hits and db._SNIPPET_OPEN in hits[0]["snippet"]


def test_delete_removes_from_index(tmp_path):
    p = tmp_path / "t.db"; db.init_db(p)
    _job(p, "a", "x.m4a", "a rare unicorn appeared")
    db.delete_job(p, "a")
    assert db.search_jobs(p, "unicorn") == []


def test_clear_empties_index(tmp_path):
    p = tmp_path / "t.db"; db.init_db(p)
    _job(p, "a", "x.m4a", "ephemeral content")
    db.clear_jobs(p)
    assert db.search_jobs(p, "ephemeral") == []


def test_rediarize_reset_drops_old_body(tmp_path):
    p = tmp_path / "t.db"; db.init_db(p)
    _job(p, "a", "x.m4a", "original words about penguins")
    db.update_job(p, "a", transcript_text=None)  # rediarize clears the transcript
    assert db.search_jobs(p, "penguins") == []


def test_empty_query_returns_nothing(tmp_path):
    p = tmp_path / "t.db"; db.init_db(p)
    _job(p, "a", "x.m4a", "anything")
    assert db.search_jobs(p, "   ") == []


def test_like_fallback_when_fts_unavailable(tmp_path, monkeypatch):
    p = tmp_path / "t.db"; db.init_db(p)
    _job(p, "a", "x.m4a", "the fallback path still finds this")
    monkeypatch.setattr(db, "_FTS_OK", False)
    hits = db.search_jobs(p, "fallback")
    assert any(h["job_id"] == "a" for h in hits)
    assert db._SNIPPET_OPEN in hits[0]["snippet"]
