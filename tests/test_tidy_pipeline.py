"""Best-effort tidy step: populated when LLM is up, skipped/safe when not."""
import os
import tempfile

os.environ.setdefault("TRANSCRIBE_DATA_DIR", tempfile.mkdtemp(prefix="fluister-tidy-"))

import pytest

import app.queue as qmod
from app.config import load_settings
from app.models import Segment
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
        lambda turns, base_url, timeout: [
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


@pytest.mark.anyio
async def test_stop_stops_llm_server(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    llm = _FakeLLM(available=True)
    q = _queue(settings, llm)
    await q.stop()
    assert llm.stopped is True
