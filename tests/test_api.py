"""End-to-end API tests using a fake transcriber and stubbed ffmpeg.

Exercises the real FastAPI app + async queue + worker + SSE, without a GPU or
model download.
"""

from __future__ import annotations

import asyncio
import json
import wave

import httpx
import pytest

from app import audio, db
from app.main import app, settings
from app.models import Segment, TranscribeInfo
from app.queue import JobQueue

pytestmark = pytest.mark.anyio


class FakeTranscriber:
    device = "cpu"
    compute_type = "int8"

    def transcribe(self, wav_path, duration, language=None, on_segment=None):
        lang = "nl" if language == "nl" else "en"
        segs = [
            Segment(0.0, 1.5, "Hallo daar." if lang == "nl" else "Hello there."),
            Segment(1.5, 3.0, "Dit is een test." if lang == "nl" else "This is a test."),
        ]
        for i, s in enumerate(segs, 1):
            if on_segment:
                on_segment(s, i / len(segs))
        return segs, [], TranscribeInfo(language=lang, duration=duration)


def _write_silent_wav(path: str) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)  # 1s of silence


@pytest.fixture
def patched(monkeypatch):
    # No real model, no real ffmpeg.
    monkeypatch.setattr(JobQueue, "_default_factory", lambda self: FakeTranscriber())
    monkeypatch.setattr(audio, "convert_to_wav", lambda src, dst: _write_silent_wav(str(dst)))
    monkeypatch.setattr(audio, "probe_duration", lambda path: 3.0)


async def _client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _wait_done(client, job_id, timeout=5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        job = (await client.get(f"/api/jobs/{job_id}")).json()
        if job["status"] in (db.STATUS_DONE, db.STATUS_ERROR):
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {job['status']}")


async def test_upload_transcribe_download_delete(patched):
    async with app.router.lifespan_context(app):
        async with await _client() as client:
            files = [("files", ("signal-2026-06-06-094449.aac", b"x", "audio/aac"))]
            r = await client.post("/api/jobs", files=files, data={"language": "nl"})
            assert r.status_code == 200
            jobs = r.json()
            assert len(jobs) == 1
            job_id = jobs[0]["id"]
            # filename timestamp was parsed
            assert jobs[0]["msg_timestamp"].startswith("2026-06-06T09:44:49")
            assert jobs[0]["msg_timestamp_source"] == "filename"

            job = await _wait_done(client, job_id)
            assert job["status"] == db.STATUS_DONE
            assert job["detected_language"] == "nl"
            assert "Hallo daar." in job["transcript_text"]

            # all four formats download
            for fmt in ("txt", "srt", "vtt", "json"):
                d = await client.get(f"/api/jobs/{job_id}/download/{fmt}")
                assert d.status_code == 200, fmt
                assert d.content
            srt = (await client.get(f"/api/jobs/{job_id}/download/srt")).text
            assert "-->" in srt and "1" in srt
            meta = json.loads((await client.get(f"/api/jobs/{job_id}/download/json")).text)
            assert meta["meta"]["language"] == "nl"

            # appears in history, then delete
            assert any(j["id"] == job_id for j in (await client.get("/api/jobs")).json())
            assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 200
            assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404


async def test_sse_emits_done_for_finished_job(patched):
    async with app.router.lifespan_context(app):
        async with await _client() as client:
            files = [("files", ("memo.m4a", b"x", "audio/mp4"))]
            job_id = (await client.post("/api/jobs", files=files)).json()[0]["id"]
            await _wait_done(client, job_id)

            events = []
            async with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        events.append(line.split(":", 1)[1].strip())
                    if "done" in events:
                        break
            assert "done" in events


async def test_unknown_job_404(patched):
    async with app.router.lifespan_context(app):
        async with await _client() as client:
            assert (await client.get("/api/jobs/nope")).status_code == 404
            assert (await client.get("/api/jobs/nope/download/txt")).status_code == 404
            assert (await client.delete("/api/jobs/nope")).status_code == 404


async def test_persons_api(patched):
    import numpy as np

    from app.main import settings as app_settings
    from app.speakers import Gallery

    async with app.router.lifespan_context(app):
        async with await _client() as client:
            g = Gallery(app_settings.db_path)
            id1, _ = g.assign_or_create(np.array([1, 0, 0], dtype="float32"))
            id2, _ = g.assign_or_create(np.array([0, 1, 0], dtype="float32"))

            listed = (await client.get("/api/persons")).json()
            ids = {p["id"] for p in listed}
            assert {id1, id2} <= ids
            # response carries no raw embedding blobs
            assert set(listed[0].keys()) == {"id", "name", "n_samples", "created_at"}

            # rename
            r = await client.put(f"/api/persons/{id1}", json={"name": "Marco"})
            assert r.status_code == 200 and r.json()["name"] == "Marco"

            # merge id2 -> id1 (samples combine, id2 disappears)
            r = await client.post("/api/persons/merge", json={"src": id2, "dst": id1})
            assert r.status_code == 200
            after = (await client.get("/api/persons")).json()
            remaining = {p["id"]: p for p in after}
            assert id2 not in remaining and remaining[id1]["n_samples"] == 2

            # cannot merge into self
            assert (
                await client.post("/api/persons/merge", json={"src": id1, "dst": id1})
            ).status_code == 400

            # delete + 404s
            assert (await client.delete(f"/api/persons/{id1}")).status_code == 200
            assert (await client.delete(f"/api/persons/{id1}")).status_code == 404
            assert (
                await client.put("/api/persons/nope", json={"name": "x"})
            ).status_code == 404
