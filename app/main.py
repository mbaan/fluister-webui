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


class RenameBody(BaseModel):
    name: str


class MergeBody(BaseModel):
    src: str
    dst: str


def _person_public(p: dict) -> dict:
    return {
        "id": p["id"],
        "name": p["name"],
        "n_samples": p["n_samples"],
        "created_at": p["created_at"],
    }


def _apply_person_change_to_jobs(
    person_id: str, *, new_name: str | None = None,
    new_person_id: str | None = None, remove: bool = False,
) -> None:
    """Propagate a person rename / merge / delete into stored job transcripts so
    speaker labels stay correct. Rename: update name. Merge: repoint to the
    merged person. Delete: drop the label (speaker -> None)."""
    for job in db.list_jobs(settings.db_path):
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
    for job in db.list_jobs(settings.db_path):
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

        with stored.open("wb") as out:
            while chunk := await f.read(1024 * 1024):
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


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
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
async def rename_person(person_id: str, body: RenameBody):
    if db.get_person(settings.db_path, person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name must not be empty")
    db.update_person(settings.db_path, person_id, name=name)
    _apply_person_change_to_jobs(person_id, new_name=name)
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
