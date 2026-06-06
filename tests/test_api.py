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

    def transcribe(self, wav_path, duration, language=None, on_segment=None, hotwords=None):
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
            data = r.json()
            assert data["duplicates"] == []
            created = data["created"]
            assert len(created) == 1
            job_id = created[0]["id"]
            # filename timestamp was parsed
            assert created[0]["msg_timestamp"].startswith("2026-06-06T09:44:49")
            assert created[0]["msg_timestamp_source"] == "filename"

            job = await _wait_done(client, job_id)
            assert job["status"] == db.STATUS_DONE
            assert job["detected_language"] == "nl"
            assert "Hallo daar." in job["transcript_text"]

            # structured transcript for the web UI
            tr = (await client.get(f"/api/jobs/{job_id}/transcript")).json()
            assert isinstance(tr["segments"], list) and len(tr["segments"]) >= 1
            assert "Hallo daar." in tr["segments"][0]["text"]

            # appears in history, then delete
            assert any(j["id"] == job_id for j in (await client.get("/api/jobs")).json())
            assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 200
            assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404


async def test_sse_emits_done_for_finished_job(patched):
    async with app.router.lifespan_context(app):
        async with await _client() as client:
            files = [("files", ("memo.m4a", b"x", "audio/mp4"))]
            job_id = (await client.post("/api/jobs", files=files)).json()["created"][0]["id"]
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
            assert (await client.get("/api/jobs/nope/transcript")).status_code == 404
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
            assert set(listed[0].keys()) == {"id", "name", "n_samples", "created_at", "keywords"}

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


async def test_persons_keywords_via_api(patched):
    import numpy as np

    from app.main import settings as app_settings
    from app.speakers import Gallery

    async with app.router.lifespan_context(app):
        async with await _client() as client:
            pid, _ = Gallery(app_settings.db_path).assign_or_create(
                np.array([1, 0, 0], dtype="float32")
            )

            # public shape now includes keywords (default null)
            me = next(p for p in (await client.get("/api/persons")).json() if p["id"] == pid)
            assert "keywords" in me and me["keywords"] is None

            # set keywords via PUT
            r = await client.put(f"/api/persons/{pid}", json={"keywords": "Xenos, Praxis"})
            assert r.status_code == 200 and r.json()["keywords"] == "Xenos, Praxis"

            # name + keywords together
            r = await client.put(
                f"/api/persons/{pid}", json={"name": "Jolis", "keywords": "Gouda"}
            )
            assert r.status_code == 200
            assert r.json()["name"] == "Jolis" and r.json()["keywords"] == "Gouda"

            # updating only the name leaves keywords intact
            r = await client.put(f"/api/persons/{pid}", json={"name": "Jolis2"})
            assert r.status_code == 200 and r.json()["keywords"] == "Gouda"

            await client.delete(f"/api/persons/{pid}")


async def test_delete_job_forgets_gallery_samples(patched):
    import numpy as np

    from app import db as _db
    from app.main import settings as app_settings
    from app.speakers import Gallery

    async with app.router.lifespan_context(app):
        async with await _client() as client:
            files = [("files", ("forget.m4a", b"q", "audio/mp4"))]
            jid = (await client.post("/api/jobs", files=files)).json()["created"][0]["id"]
            await _wait_done(client, jid)

            # A voice sample attributed to this job.
            g = Gallery(app_settings.db_path)
            pid, _ = g.assign_or_create(np.array([1, 0, 0], dtype="float32"), job_id=jid)
            assert len(_db.list_person_embeddings(app_settings.db_path, pid)) == 1

            # Deleting the job forgets its sample and empties the centroid.
            assert (await client.delete(f"/api/jobs/{jid}")).status_code == 200
            assert _db.list_person_embeddings(app_settings.db_path, pid) == []
            assert _db.get_person(app_settings.db_path, pid)["centroid"] is None

            await client.delete(f"/api/persons/{pid}")  # cleanup


async def test_duplicate_skipped(patched):
    async with app.router.lifespan_context(app):
        async with await _client() as client:
            content = b"hello-bytes"
            files = [("files", ("dup.m4a", content, "audio/mp4"))]
            first = (await client.post("/api/jobs", files=files)).json()
            assert len(first["created"]) == 1 and first["duplicates"] == []
            jid = first["created"][0]["id"]
            await _wait_done(client, jid)

            # same name + same size -> skipped
            files2 = [("files", ("dup.m4a", content, "audio/mp4"))]
            second = (await client.post("/api/jobs", files=files2)).json()
            assert second["created"] == []
            assert len(second["duplicates"]) == 1
            assert second["duplicates"][0]["duplicate_of"] == jid

            # same name, different size -> NOT a duplicate
            files3 = [("files", ("dup.m4a", content + b"x", "audio/mp4"))]
            third = (await client.post("/api/jobs", files=files3)).json()
            assert len(third["created"]) == 1 and third["duplicates"] == []


async def test_clear_all_keeps_persons(patched):
    import numpy as np

    from app.main import settings as app_settings
    from app.speakers import Gallery

    async with app.router.lifespan_context(app):
        async with await _client() as client:
            files = [("files", ("clearme.m4a", b"abc", "audio/mp4"))]
            jid = (await client.post("/api/jobs", files=files)).json()["created"][0]["id"]
            await _wait_done(client, jid)
            Gallery(app_settings.db_path).assign_or_create(np.array([1, 0, 0], dtype="float32"))

            r = await client.post("/api/jobs/clear")
            assert r.status_code == 200 and r.json()["deleted"] >= 1
            assert (await client.get("/api/jobs")).json() == []

            # persons (voice gallery) are kept
            persons = (await client.get("/api/persons")).json()
            assert len(persons) >= 1
            for p in persons:  # cleanup for other tests
                await client.delete(f"/api/persons/{p['id']}")


async def test_delete_person_scrubs_job_labels(patched):
    import json as _json

    from app import db as _db
    from app.main import settings as app_settings

    async with app.router.lifespan_context(app):
        async with await _client() as client:
            files = [("files", ("spk.m4a", b"zzz", "audio/mp4"))]
            jid = (await client.post("/api/jobs", files=files)).json()["created"][0]["id"]
            await _wait_done(client, jid)

            # Simulate diarization having labeled this job with a person.
            pid = "person-noise"
            _db.create_person(app_settings.db_path, {"id": pid, "name": "Noise", "n_samples": 1})
            _db.update_job(
                app_settings.db_path, jid,
                segments_json=_json.dumps(
                    [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "Noise", "person_id": pid}]
                ),
                speakers=_json.dumps({"SPEAKER_00": {"person_id": pid, "name": "Noise"}}),
            )

            # Deleting the person must scrub its label from the transcript.
            assert (await client.delete(f"/api/persons/{pid}")).status_code == 200
            tr = (await client.get(f"/api/jobs/{jid}/transcript")).json()
            assert tr["segments"][0]["speaker"] is None
            assert tr["segments"][0]["person_id"] is None
            job = (await client.get(f"/api/jobs/{jid}")).json()
            assert _json.loads(job["speakers"])["SPEAKER_00"]["person_id"] is None


async def test_startup_scrubs_orphan_speakers(patched):
    import json as _json

    from app import db as _db
    from app.main import settings as app_settings

    # A job referencing a person that doesn't exist, seeded before startup.
    _db.init_db(app_settings.db_path)
    jid = "orphan-job"
    _db.create_job(app_settings.db_path, {
        "id": jid, "original_filename": "o.m4a", "stored_path": "/x",
        "language": "auto", "status": _db.STATUS_DONE, "model_name": "m",
    })
    _db.update_job(
        app_settings.db_path, jid,
        segments_json=_json.dumps(
            [{"start": 0, "end": 1, "text": "hi", "speaker": "Ghost", "person_id": "ghost"}]
        ),
        speakers=_json.dumps({"SPEAKER_00": {"person_id": "ghost", "name": "Ghost"}}),
    )

    # Startup runs the scrub.
    async with app.router.lifespan_context(app):
        async with await _client() as client:
            tr = (await client.get(f"/api/jobs/{jid}/transcript")).json()
            assert tr["segments"][0]["person_id"] is None
            assert tr["segments"][0]["speaker"] is None
            await client.delete(f"/api/jobs/{jid}")
