"""FastAPI application: REST API + SSE + static web UI."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from pydantic import BaseModel

from app import db
from app.config import ensure_dirs, load_settings
from app.filename_time import parse_filename_timestamp
from app.queue import JobQueue
from app.speakers import Gallery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fluister")

settings = load_settings()
STATIC_DIR = Path(__file__).resolve().parent / "static"

LANGUAGES = {"auto", "nl", "en"}


class PersonUpdateBody(BaseModel):
    name: str | None = None
    keywords: str | None = None


class MergeBody(BaseModel):
    src: str
    dst: str


def _person_public(p: dict) -> dict:
    return {
        "id": p["id"],
        "name": p["name"],
        "n_samples": p["n_samples"],
        "created_at": p["created_at"],
        "keywords": p.get("keywords"),
    }


def _apply_person_change_to_jobs(
    person_id: str, *, new_name: str | None = None,
    new_person_id: str | None = None, remove: bool = False,
) -> None:
    """Propagate a person rename / merge / delete into stored job transcripts so
    speaker labels stay correct. Rename: update name. Merge: repoint to the
    merged person. Delete: drop the label (speaker -> None)."""
    # Scan every job (limit=-1), not just the most recent page — a rename/merge
    # must reach older transcripts too.
    for job in db.list_jobs(settings.db_path, limit=-1):
        speakers = json.loads(job["speakers"]) if job.get("speakers") else None
        segs = json.loads(job["segments_json"]) if job.get("segments_json") else None
        changed = False

        if speakers:
            for lbl, v in list(speakers.items()):
                if (v or {}).get("person_id") != person_id:
                    continue
                if remove:
                    speakers[lbl] = {"person_id": None, "name": None}
                elif new_person_id:
                    speakers[lbl] = {"person_id": new_person_id, "name": new_name}
                else:
                    speakers[lbl] = {"person_id": person_id, "name": new_name}
                changed = True

        if segs:
            for s in segs:
                if s.get("person_id") != person_id:
                    continue
                if remove:
                    s["person_id"], s["speaker"] = None, None
                elif new_person_id:
                    s["person_id"], s["speaker"] = new_person_id, new_name
                else:
                    s["speaker"] = new_name
                changed = True

        if changed:
            db.update_job(
                settings.db_path, job["id"],
                speakers=json.dumps(speakers) if speakers else None,
                segments_json=json.dumps(segs) if segs else None,
            )


def _scrub_orphan_speakers() -> None:
    """Drop speaker labels in jobs that point at persons which no longer exist
    (e.g. a person deleted before this propagation existed). Self-healing."""
    valid = {p["id"] for p in db.list_persons(settings.db_path)}
    for job in db.list_jobs(settings.db_path, limit=-1):  # every job, not just the latest page
        speakers = json.loads(job["speakers"]) if job.get("speakers") else None
        segs = json.loads(job["segments_json"]) if job.get("segments_json") else None
        changed = False
        if speakers:
            for lbl, v in list(speakers.items()):
                pid = (v or {}).get("person_id")
                if pid and pid not in valid:
                    speakers[lbl] = {"person_id": None, "name": None}
                    changed = True
        if segs:
            for s in segs:
                pid = s.get("person_id")
                if pid and pid not in valid:
                    s["person_id"], s["speaker"] = None, None
                    changed = True
        if changed:
            db.update_job(
                settings.db_path, job["id"],
                speakers=json.dumps(speakers) if speakers else None,
                segments_json=json.dumps(segs) if segs else None,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs(settings)
    db.init_db(settings.db_path)
    interrupted = db.mark_interrupted(settings.db_path)
    if interrupted:
        logger.info("Marked %d in-flight job(s) as interrupted", len(interrupted))
    _scrub_orphan_speakers()
    app.state.queue = JobQueue(settings)
    await app.state.queue.start()
    logger.info("fluister ready on http://%s:%s", settings.host, settings.port)
    yield
    await app.state.queue.stop()


app = FastAPI(title="fluister", lifespan=lifespan)


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """Tell the browser to revalidate the UI assets every load. Combined with
    StaticFiles' ETag/Last-Modified, an unchanged file still 304s, but an edited
    one is fetched fresh — so a normal reload shows UI changes without a manual
    hard-refresh. (This is a local single-user tool; freshness beats caching.)"""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _queue(request: Request) -> JobQueue:
    return request.app.state.queue


# ── API ──────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status(request: Request):
    q = _queue(request)
    t = q.transcriber
    return {
        "model_ready": q.model_ready,
        "model_name": settings.model_name,
        "device": getattr(t, "device", None),
        "compute_type": getattr(t, "compute_type", None),
    }


@app.post("/api/jobs")
async def create_jobs(
    request: Request,
    files: list[UploadFile] = File(...),
    language: str = Form("auto"),
):
    q = _queue(request)
    if language not in LANGUAGES:
        language = "auto"

    created = []
    duplicates = []
    for f in files:
        job_id = uuid.uuid4().hex
        original = f.filename or "audio"
        ext = Path(original).suffix
        stored = settings.uploads_dir / f"{job_id}{ext}"

        # Stream to disk, enforcing the configured size cap so a single huge
        # upload can't fill the disk.
        max_bytes = settings.max_upload_mb * 1024 * 1024
        written = 0
        with stored.open("wb") as out:
            while chunk := await f.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    out.close()
                    stored.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"“{original}” exceeds the {settings.max_upload_mb} MB upload limit",
                    )
                out.write(chunk)
        size = stored.stat().st_size

        # Skip files already transcribed (or in flight) with the same name + size.
        dup = db.find_duplicate(settings.db_path, original, size)
        if dup:
            stored.unlink(missing_ok=True)
            duplicates.append({"filename": original, "duplicate_of": dup["id"]})
            continue

        parsed = parse_filename_timestamp(original)
        job = {
            "id": job_id,
            "original_filename": original,
            "stored_path": str(stored),
            "language": language,
            "status": db.STATUS_QUEUED,
            "model_name": settings.model_name,
            "progress": 0,
            "created_at": db.now_iso(),
            "size": size,
            "msg_timestamp": parsed.dt.isoformat() if parsed else None,
            "msg_timestamp_source": parsed.source if parsed else None,
            "msg_has_time": int(parsed.has_time) if parsed else None,
        }
        row = db.create_job(settings.db_path, job)
        await q.enqueue(job_id)
        created.append(row)

    return {"created": created, "duplicates": duplicates}


@app.post("/api/jobs/clear")
async def clear_all_jobs():
    """Delete all transcriptions and their files. Keeps persons (voice gallery)."""
    for p in settings.uploads_dir.glob("*"):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    return {"deleted": db.clear_jobs(settings.db_path)}


@app.get("/api/jobs")
async def list_jobs():
    return db.list_jobs(settings.db_path)


@app.get("/api/search")
async def search(q: str = ""):
    """Full-text search across transcripts + filenames (FTS5, LIKE fallback)."""
    return db.search_jobs(settings.db_path, q, limit=20)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    if db.get_job(settings.db_path, job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    q = _queue(request)
    sub = q.subscribe(job_id)  # subscribe BEFORE the snapshot to avoid missing events

    async def gen():
        try:
            snap = db.get_job(settings.db_path, job_id)
            yield _sse("status", {
                "status": snap["status"],
                "progress": snap["progress"],
                "detected_language": snap["detected_language"],
            })
            if snap["status"] == db.STATUS_DONE:
                yield _sse("done", snap)
                return
            if snap["status"] in (db.STATUS_ERROR, db.STATUS_INTERRUPTED):
                yield _sse("error", {"message": snap["error"] or "Job failed"})
                return
            while True:
                event = await sub.get()
                yield _sse(event["event"], event["data"])
                if event["event"] in ("done", "error"):
                    break
        finally:
            q.unsubscribe(job_id, sub)

    return EventSourceResponse(gen())


@app.get("/api/jobs/{job_id}/transcript")
async def job_transcript(job_id: str):
    """Structured transcript (segments + speaker map) for the web UI."""
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    segments = json.loads(job["segments_json"]) if job.get("segments_json") else []
    speakers = json.loads(job["speakers"]) if job.get("speakers") else {}
    return {"segments": segments, "speakers": speakers}


@app.get("/api/jobs/{job_id}/audio")
async def job_audio(job_id: str):
    """Serve a job's 16 kHz mono WAV so reviewers can play/seek it.
    Starlette's FileResponse handles HTTP Range (206 + Accept-Ranges)."""
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    wav = job.get("wav_path")
    if not wav or not Path(wav).exists():
        raise HTTPException(status_code=404, detail="Audio not available")
    return FileResponse(wav, media_type="audio/wav")


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    # Forget this job's voice samples first so deleting (and re-uploading) a clip
    # doesn't leave orphaned/duplicate samples skewing the gallery centroids.
    Gallery(settings.db_path).forget_job(job_id)
    job = db.delete_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    for key in ("stored_path", "wav_path"):
        p = job.get(key)
        if p:
            Path(p).unlink(missing_ok=True)
    return {"ok": True}


# ── Persons (global voice gallery) ──────────────────────────────────────────
@app.get("/api/persons")
async def list_persons():
    return [_person_public(p) for p in db.list_persons(settings.db_path)]


@app.put("/api/persons/{person_id}")
async def update_person(person_id: str, body: PersonUpdateBody):
    if db.get_person(settings.db_path, person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    fields: dict = {}
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name must not be empty")
        fields["name"] = name
    if body.keywords is not None:
        # Empty/whitespace clears the list back to NULL.
        fields["keywords"] = body.keywords.strip() or None
    if fields:
        db.update_person(settings.db_path, person_id, **fields)
    # Only a name change needs propagating into stored transcripts; keywords
    # only affect future transcriptions.
    if "name" in fields:
        _apply_person_change_to_jobs(person_id, new_name=fields["name"])
    return _person_public(db.get_person(settings.db_path, person_id))


@app.delete("/api/persons/{person_id}")
async def delete_person(person_id: str):
    if db.delete_person(settings.db_path, person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    _apply_person_change_to_jobs(person_id, remove=True)
    return {"ok": True}


@app.post("/api/persons/merge")
async def merge_persons(body: MergeBody):
    if body.src == body.dst:
        raise HTTPException(status_code=400, detail="Cannot merge a person into itself")
    dst = db.get_person(settings.db_path, body.dst)
    if db.get_person(settings.db_path, body.src) is None or dst is None:
        raise HTTPException(status_code=404, detail="Person not found")
    Gallery(settings.db_path).merge(body.src, body.dst)
    _apply_person_change_to_jobs(body.src, new_person_id=body.dst, new_name=dst["name"])
    return {"ok": True}


@app.post("/api/jobs/{job_id}/rediarize")
async def rediarize(job_id: str, request: Request):
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.get("stored_path") or not Path(job["stored_path"]).exists():
        raise HTTPException(status_code=400, detail="Original audio no longer available")
    db.update_job(
        settings.db_path, job_id, status=db.STATUS_QUEUED, error=None, progress=0.0,
        diarized=0, speakers=None, transcript_text=None, detected_language=None,
        # Clear the previous run's structured + readable transcripts too, so a
        # rerun that produces no tidy pass (LLM down) can't keep serving a stale
        # readable view mismatched with the freshly re-diarized segments.
        segments_json=None, tidied_json=None, insights_json=None,
        started_at=None, finished_at=None,
    )
    await _queue(request).enqueue(job_id)
    return db.get_job(settings.db_path, job_id)


def _sse(event: str, data) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


# ── Static web UI ──────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
