"""Best-effort tidy step: populated when LLM is up, skipped/safe when not."""
import asyncio
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TRANSCRIBE_DATA_DIR", tempfile.mkdtemp(prefix="fluister-tidy-"))

import pytest

import app.db as qdb
import app.queue as qmod
from app import audio
from app.config import load_settings
from app.models import Segment, TranscribeInfo
from app.queue import JobQueue


class _FakeLLM:
    def __init__(self, available):
        self.available = available
        self.base_url = "http://x:8080"
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def _queue(settings, llm):
    q = JobQueue(settings)
    q.llm_server = llm
    return q


def test_maybe_tidy_populates_when_available(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    monkeypatch.setattr(
        qmod, "tidy_turns",
        lambda turns, base_url, timeout, language=None, on_progress=None: [
            {"speaker": t.speaker, "text": t.text.upper()} for t in turns
        ],
    )
    q = _queue(settings, _FakeLLM(available=True))
    segs = [Segment(0, 1, "hi", "Ann"), Segment(1, 2, "yo", "Ann")]
    out = q._maybe_tidy("job1", segs)
    assert out == [{"speaker": "Ann", "text": "HI YO"}]


def test_maybe_tidy_skips_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    called = []
    monkeypatch.setattr(qmod, "tidy_turns", lambda *a, **k: called.append(1) or [])
    q = _queue(settings, _FakeLLM(available=False))
    assert q._maybe_tidy("job1", [Segment(0, 1, "hi")]) is None
    assert called == []


def test_maybe_tidy_best_effort_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(qmod, "tidy_turns", boom)
    q = _queue(settings, _FakeLLM(available=True))
    assert q._maybe_tidy("job1", [Segment(0, 1, "hi")]) is None  # swallowed


def test_maybe_tidy_reports_progress(tmp_path, monkeypatch):
    """Per-turn progress is published over SSE and persisted to the job row."""
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    qdb.init_db(settings.db_path)
    qdb.create_job(settings.db_path, {
        "id": "job1", "original_filename": "a.ogg", "stored_path": "a.ogg",
        "language": "auto", "status": qdb.STATUS_TIDYING, "model_name": "m",
    })

    def fake_tidy(turns, base_url, timeout, language=None, on_progress=None):
        out = []
        for i, t in enumerate(turns, start=1):
            out.append({"speaker": t.speaker, "text": t.text})
            on_progress(i, len(turns))
        return out

    monkeypatch.setattr(qmod, "tidy_turns", fake_tidy)
    q = _queue(settings, _FakeLLM(available=True))
    published = []
    monkeypatch.setattr(
        q, "publish_threadsafe", lambda jid, ev, data: published.append((jid, ev, data))
    )
    segs = [Segment(0, 1, "hi", "Ann"), Segment(1, 2, "yo", "Bob")]
    assert q._maybe_tidy("job1", segs) is not None

    statuses = [d for (_jid, ev, d) in published if ev == "status"]
    assert [s["progress"] for s in statuses] == [0.5, 1.0]
    assert all(s["status"] == qdb.STATUS_TIDYING for s in statuses)
    assert qdb.get_job(settings.db_path, "job1")["progress"] == 1.0


def test_mark_interrupted_completes_tidying_job(tmp_path):
    """A job stranded mid-tidy has its transcript persisted already, so restart
    recovery finishes it as DONE (readable view just missing) — not an error."""
    p = tmp_path / "t.db"
    qdb.init_db(p)
    qdb.create_job(p, {
        "id": "j1", "original_filename": "a.ogg", "stored_path": "a.ogg",
        "language": "auto", "status": qdb.STATUS_TIDYING, "model_name": "m",
    })
    qdb.create_job(p, {
        "id": "j2", "original_filename": "b.ogg", "stored_path": "b.ogg",
        "language": "auto", "status": qdb.STATUS_TRANSCRIBING, "model_name": "m",
    })
    assert qdb.mark_interrupted(p) == ["j2"]
    done = qdb.get_job(p, "j1")
    assert done["status"] == qdb.STATUS_DONE
    assert done["progress"] == 1.0
    assert done["error"] is None
    assert qdb.get_job(p, "j2")["status"] == qdb.STATUS_INTERRUPTED


class _FakeTranscriber:
    device = "cpu"
    compute_type = "int8"

    def transcribe(self, wav_path, duration, language=None, on_segment=None, hotwords=None):
        segs = [Segment(0.0, 1.0, "hi", "Ann"), Segment(1.0, 2.0, "yo", "Bob")]
        for i, s in enumerate(segs, 1):
            if on_segment:
                on_segment(s, i / len(segs))
        return segs, [], TranscribeInfo(language="en", duration=duration)


@pytest.mark.anyio
async def test_process_surfaces_tidy_phase(tmp_path, monkeypatch):
    """Full pipeline: the job is TIDYING in the DB while the LLM pass runs
    (visible to polling clients), SSE carries rising tidy progress, and the
    tidied event precedes done."""
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    qdb.init_db(settings.db_path)
    src = tmp_path / "a.ogg"
    src.write_bytes(b"x")
    qdb.create_job(settings.db_path, {
        "id": "job1", "original_filename": "a.ogg", "stored_path": str(src),
        "language": "auto", "status": qdb.STATUS_QUEUED, "model_name": "m",
    })
    monkeypatch.setattr(audio, "convert_to_wav", lambda s, d: Path(d).write_bytes(b""))
    monkeypatch.setattr(audio, "probe_duration", lambda p: 2.0)

    status_mid_tidy = []

    def fake_tidy(turns, base_url, timeout, language=None, on_progress=None):
        status_mid_tidy.append(qdb.get_job(settings.db_path, "job1")["status"])
        out = []
        for i, t in enumerate(turns, start=1):
            out.append({"speaker": t.speaker, "text": t.text.upper()})
            on_progress(i, len(turns))
        return out

    monkeypatch.setattr(qmod, "tidy_turns", fake_tidy)

    q = JobQueue(settings)
    q.transcriber = _FakeTranscriber()
    q.llm_server = _FakeLLM(available=True)
    q._loop = asyncio.get_running_loop()
    sub = q.subscribe("job1")
    await q._process("job1")
    await asyncio.sleep(0)  # flush call_soon_threadsafe callbacks

    events = []
    while not sub.empty():
        events.append(sub.get_nowait())

    assert status_mid_tidy == [qdb.STATUS_TIDYING]
    tidy_progress = [
        e["data"]["progress"] for e in events
        if e["event"] == "status" and e["data"]["status"] == qdb.STATUS_TIDYING
    ]
    assert tidy_progress == [0.0, 0.5, 1.0]
    names = [e["event"] for e in events]
    assert names.index("tidied") < names.index("done")

    job = qdb.get_job(settings.db_path, "job1")
    assert job["status"] == qdb.STATUS_DONE
    assert job["progress"] == 1.0
    assert json.loads(job["tidied_json"]) == [
        {"speaker": "Ann", "text": "HI"},
        {"speaker": "Bob", "text": "YO"},
    ]


@pytest.mark.anyio
async def test_stop_stops_llm_server(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    llm = _FakeLLM(available=True)
    q = _queue(settings, llm)
    await q.stop()
    assert llm.stopped is True
