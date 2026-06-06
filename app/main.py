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

FORMATS = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json; charset=utf-8",
}
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs(settings)
    db.init_db(settings.db_path)
    interrupted = db.mark_interrupted(settings.db_path)
    if interrupted:
        logger.info("Marked %d in-flight job(s) as interrupted", len(interrupted))
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
    for d in (settings.uploads_dir, settings.outputs_dir):
        for p in d.glob("*"):
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


@app.get("/api/jobs/{job_id}/download/{fmt}")
async def download(job_id: str, fmt: str):
    if fmt not in FORMATS:
        raise HTTPException(status_code=404, detail="Unknown format")
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    path = settings.outputs_dir / f"{job_id}.{fmt}"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript not ready")
    base = Path(job["original_filename"]).stem or "transcript"
    return FileResponse(path, media_type=FORMATS[fmt], filename=f"{base}.{fmt}")


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    job = db.delete_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    for key in ("stored_path", "wav_path"):
        p = job.get(key)
        if p:
            Path(p).unlink(missing_ok=True)
    for fmt in FORMATS:
        (settings.outputs_dir / f"{job_id}.{fmt}").unlink(missing_ok=True)
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
    return _person_public(db.get_person(settings.db_path, person_id))


@app.delete("/api/persons/{person_id}")
async def delete_person(person_id: str):
    if db.delete_person(settings.db_path, person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return {"ok": True}


@app.post("/api/persons/merge")
async def merge_persons(body: MergeBody):
    if body.src == body.dst:
        raise HTTPException(status_code=400, detail="Cannot merge a person into itself")
    if (
        db.get_person(settings.db_path, body.src) is None
        or db.get_person(settings.db_path, body.dst) is None
    ):
        raise HTTPException(status_code=404, detail="Person not found")
    Gallery(settings.db_path).merge(body.src, body.dst)
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
