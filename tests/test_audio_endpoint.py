"""Tests for the GET /api/jobs/{job_id}/audio endpoint.

The audio endpoint serves a job's 16 kHz mono WAV with HTTP Range support
(Starlette's FileResponse handles ranging). It only touches ``db`` + ``settings``
and never the JobQueue, so we deliberately construct ``TestClient(fastapi_app)`` WITHOUT
the ``with`` context-manager form: Starlette only runs the app ``lifespan``
(which loads a heavy GPU model and starts the queue) when the client is entered
as a context manager. Calling ``client.get(...)`` directly skips startup, so no
model loads.
"""

from __future__ import annotations

import wave

from fastapi.testclient import TestClient

from app import db
from app.config import load_settings
from app.main import app as fastapi_app


def _write_silent_wav(path: str) -> None:
    """~0.1 s of 16 kHz mono silence -> a small but valid WAV file."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence


def _make_job(test_settings, *, job_id: str, wav_path: str | None) -> None:
    db.create_job(
        test_settings.db_path,
        {
            "id": job_id,
            "original_filename": "clip.m4a",
            "stored_path": "/does/not/matter.m4a",
            "wav_path": wav_path,
            "language": "auto",
            "status": db.STATUS_DONE,
            "model_name": "large-v3",
            "created_at": db.now_iso(),
        },
    )


def _setup(monkeypatch, tmp_path):
    """Point the app at a throwaway DB under tmp_path and init it."""
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    test_settings = load_settings()
    monkeypatch.setattr("app.main.settings", test_settings)
    db.init_db(test_settings.db_path)
    return test_settings


def test_audio_ok_and_range(monkeypatch, tmp_path):
    test_settings = _setup(monkeypatch, tmp_path)
    wav = tmp_path / "job1.wav"
    _write_silent_wav(str(wav))
    _make_job(test_settings, job_id="job1", wav_path=str(wav))

    client = TestClient(fastapi_app)  # no `with` -> lifespan (model + queue) is skipped

    r = client.get("/api/jobs/job1/audio")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert len(r.content) > 0
    assert r.headers["accept-ranges"] == "bytes"

    r2 = client.get("/api/jobs/job1/audio", headers={"Range": "bytes=0-1"})
    assert r2.status_code == 206
    assert "content-range" in r2.headers


def test_audio_unknown_job_404(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    client = TestClient(fastapi_app)
    r = client.get("/api/jobs/does-not-exist/audio")
    assert r.status_code == 404


def test_audio_missing_file_404(monkeypatch, tmp_path):
    test_settings = _setup(monkeypatch, tmp_path)
    _make_job(test_settings, job_id="job2", wav_path=str(tmp_path / "gone.wav"))
    client = TestClient(fastapi_app)
    r = client.get("/api/jobs/job2/audio")
    assert r.status_code == 404
