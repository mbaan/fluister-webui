"""Tests for per-person hotwords: the union builder, DB column persistence,
engine threading, and queue integration."""

from __future__ import annotations

import asyncio
import sqlite3
import wave

import httpx
import pytest

from app import audio, db
from app.models import Segment, TranscribeInfo
from app.queue import JobQueue
from app.speakers import build_hotwords


# ── build_hotwords (pure) ────────────────────────────────────────────────────

def test_build_hotwords_includes_names_and_keywords():
    persons = [
        {"name": "Jolis", "keywords": "Xenos, Praxis"},
        {"name": "Tijn", "keywords": "Energiehaven"},
    ]
    hot = build_hotwords(persons)
    for term in ("Jolis", "Tijn", "Xenos", "Praxis", "Energiehaven"):
        assert term in hot


def test_build_hotwords_excludes_placeholder_names():
    persons = [{"name": "Person 3", "keywords": "tegellegger"}]
    hot = build_hotwords(persons)
    assert "Person 3" not in hot
    assert "Person" not in hot
    assert "tegellegger" in hot


def test_build_hotwords_splits_on_commas_and_newlines():
    persons = [{"name": "Person 1", "keywords": "Gouda, Brabant\nWaalwijk"}]
    hot = build_hotwords(persons)
    terms = {t.strip() for t in hot.split(",")}
    assert {"Gouda", "Brabant", "Waalwijk"} <= terms


def test_build_hotwords_dedupes_case_insensitively_keeping_first_seen():
    persons = [
        {"name": "Jolis", "keywords": "Gouda"},
        {"name": "jolis", "keywords": "gouda"},
    ]
    hot = build_hotwords(persons)
    terms = [t.strip() for t in hot.split(",")]
    # "Jolis"/"jolis" collapse to one entry; same for Gouda
    assert sum(1 for t in terms if t.lower() == "jolis") == 1
    assert sum(1 for t in terms if t.lower() == "gouda") == 1
    # first-seen casing preserved
    assert "Jolis" in terms
    assert "Gouda" in terms


def test_build_hotwords_empty_returns_none():
    assert build_hotwords([]) is None
    assert build_hotwords([{"name": "Person 1", "keywords": None}]) is None
    assert build_hotwords([{"name": "Person 2", "keywords": "   "}]) is None


# ── DB: keywords column ──────────────────────────────────────────────────────

def test_create_person_round_trips_keywords(tmp_path):
    dbp = tmp_path / "t.db"
    db.init_db(dbp)
    db.create_person(dbp, {"id": "p1", "name": "Jolis", "keywords": "Xenos, Praxis"})
    assert db.get_person(dbp, "p1")["keywords"] == "Xenos, Praxis"


def test_update_person_sets_keywords(tmp_path):
    dbp = tmp_path / "t.db"
    db.init_db(dbp)
    db.create_person(dbp, {"id": "p1", "name": "Jolis"})
    db.update_person(dbp, "p1", keywords="Gouda")
    assert db.get_person(dbp, "p1")["keywords"] == "Gouda"


def test_migrate_adds_keywords_to_legacy_persons_table(tmp_path):
    """A persons table created before this feature gains the column, keeping rows."""
    dbp = tmp_path / "legacy.db"
    conn = sqlite3.connect(dbp)
    conn.executescript(
        """
        CREATE TABLE persons (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
            centroid BLOB, n_samples INTEGER NOT NULL DEFAULT 0, dim INTEGER
        );
        INSERT INTO persons (id, name, created_at, n_samples)
        VALUES ('old', 'Tijn', '2026-01-01T00:00:00+00:00', 1);
        """
    )
    conn.commit()
    conn.close()

    db.init_db(dbp)  # runs _migrate

    cols = {c["name"] for c in _table_cols(dbp, "persons")}
    assert "keywords" in cols
    row = db.get_person(dbp, "old")
    assert row["name"] == "Tijn"  # existing row survived
    assert row["keywords"] is None


def _table_cols(dbp, table):
    conn = sqlite3.connect(dbp)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()


# ── engine threading (app/transcriber.py) ────────────────────────────────────

class _FakeInfo:
    language = "nl"
    duration = 3.0


class _RecordingPipeline:
    """Stands in for BatchedInferencePipeline; records transcribe kwargs."""

    def __init__(self, model=None):
        self.calls: list[dict] = []

    def transcribe(self, wav_path, **kwargs):
        self.calls.append(kwargs)
        return iter([]), _FakeInfo()


class _OOMPipeline(_RecordingPipeline):
    def transcribe(self, wav_path, **kwargs):
        raise RuntimeError("CUDA failed with error out of memory")


class _RecordingModel:
    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []

    def transcribe(self, wav_path, **kwargs):
        self.calls.append(kwargs)
        return iter([]), _FakeInfo()


def _make_transcriber(monkeypatch, pipeline_cls):
    import app.transcriber as tr

    monkeypatch.setattr(tr, "WhisperModel", _RecordingModel)
    monkeypatch.setattr(tr, "BatchedInferencePipeline", pipeline_cls)
    return tr.Transcriber(model_name="tiny", device="cpu", compute_type="int8")


def test_hotwords_passed_to_batched_pipeline(monkeypatch):
    t = _make_transcriber(monkeypatch, _RecordingPipeline)
    t.transcribe("x.wav", duration=3.0, hotwords="Jolis, Tijn")
    assert t.pipeline.calls[0]["hotwords"] == "Jolis, Tijn"


def test_hotwords_passed_to_nonbatched_fallback(monkeypatch):
    # Batched path OOMs all the way down to the non-batched model.transcribe.
    t = _make_transcriber(monkeypatch, _OOMPipeline)
    t.transcribe("x.wav", duration=3.0, hotwords="Jolis, Tijn")
    assert t.model.calls[-1]["hotwords"] == "Jolis, Tijn"


def test_hotwords_default_is_none(monkeypatch):
    t = _make_transcriber(monkeypatch, _RecordingPipeline)
    t.transcribe("x.wav", duration=3.0)
    assert t.pipeline.calls[0]["hotwords"] is None


# ── queue integration: _process feeds the gallery union ──────────────────────

class _CapturingTranscriber:
    device = "cpu"
    compute_type = "int8"

    def __init__(self, sink):
        self.sink = sink

    def transcribe(self, wav_path, duration, language=None, on_segment=None, hotwords=None):
        self.sink.append(hotwords)
        seg = Segment(0.0, 1.0, "hi")
        if on_segment:
            on_segment(seg, 1.0)
        return [seg], [], TranscribeInfo(language="nl", duration=duration)


def _write_silent_wav(path: str) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)


async def _client():
    transport = httpx.ASGITransport(app=_app())
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _app():
    from app.main import app
    return app


async def _wait_done(client, job_id, timeout=5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        job = (await client.get(f"/api/jobs/{job_id}")).json()
        if job["status"] in (db.STATUS_DONE, db.STATUS_ERROR):
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.mark.anyio
async def test_process_feeds_gallery_hotwords(monkeypatch):
    import numpy as np

    from app.main import app, settings
    from app.speakers import Gallery

    sink: list = []
    monkeypatch.setattr(
        JobQueue, "_default_factory", lambda self: _CapturingTranscriber(sink)
    )
    monkeypatch.setattr(audio, "convert_to_wav", lambda src, dst: _write_silent_wav(str(dst)))
    monkeypatch.setattr(audio, "probe_duration", lambda path: 3.0)

    async with app.router.lifespan_context(app):
        g = Gallery(settings.db_path)
        pid, _ = g.assign_or_create(np.array([1, 0, 0], dtype="float32"))
        g.rename(pid, "Jolis")
        db.update_person(settings.db_path, pid, keywords="Xenos, Praxis")
        try:
            async with await _client() as client:
                files = [("files", ("hw.m4a", b"x", "audio/mp4"))]
                jid = (await client.post("/api/jobs", files=files)).json()["created"][0]["id"]
                await _wait_done(client, jid)
        finally:
            db.delete_person(settings.db_path, pid)

    assert sink, "transcribe was never called"
    hot = sink[-1]
    assert hot and "Jolis" in hot and "Xenos" in hot and "Praxis" in hot
