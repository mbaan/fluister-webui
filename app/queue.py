"""Background job queue with a single GPU worker and per-job SSE pub/sub.

The GPU processes one job at a time, so a single worker coroutine drains an
asyncio queue. The blocking work (ffmpeg + faster-whisper) runs in a worker
thread via ``asyncio.to_thread``; segment events produced inside that thread are
published back to SSE subscribers using ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from contextlib import suppress
from typing import Any, Callable

from app import assign, audio, db, formats
from app.config import Settings
from app.models import TranscriptMeta
from app.speakers import Gallery

logger = logging.getLogger(__name__)

# An SSE event is just {"event": <name>, "data": <json-serialisable dict>}.
Event = dict[str, Any]


class JobQueue:
    def __init__(
        self, settings: Settings, transcriber_factory: Callable[[], Any] | None = None
    ) -> None:
        self.settings = settings
        self._factory = transcriber_factory or self._default_factory
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers: dict[str, set[asyncio.Queue[Event]]] = defaultdict(set)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker_task: asyncio.Task | None = None
        self._load_task: asyncio.Task | None = None
        self._model_ready = asyncio.Event()
        self.transcriber: Any | None = None
        self.diarizer: Any | None = None

    def _default_factory(self) -> Any:
        # Imported lazily so tests can inject a fake without importing faster_whisper.
        from app.transcriber import Transcriber

        s = self.settings
        return Transcriber(
            model_name=s.model_name,
            device=s.device,
            compute_type=s.compute_type,
            batch_size=s.batch_size,
            use_vad=s.use_vad,
        )

    # ── lifecycle ────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._load_task = asyncio.create_task(self._load_model())
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        for task in (self._worker_task, self._load_task):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def _load_model(self) -> None:
        try:
            self.transcriber = await asyncio.to_thread(self._factory)
            logger.info(
                "Transcription model ready (device=%s compute=%s)",
                getattr(self.transcriber, "device", "?"),
                getattr(self.transcriber, "compute_type", "?"),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load transcription model")

        # Diarizer is optional — its absence just disables speaker labels.
        try:
            if self.settings.diarize and self.settings.hf_token:
                from app.diarizer import Diarizer

                self.diarizer = await asyncio.to_thread(
                    lambda: Diarizer(
                        model_name=self.settings.diarize_model,
                        device=self.settings.device,
                        hf_token=self.settings.hf_token,
                    )
                )
                logger.info(
                    "Diarizer ready (device=%s)", getattr(self.diarizer, "device", "?")
                )
            elif self.settings.diarize and not self.settings.hf_token:
                logger.warning(
                    "TRANSCRIBE_DIARIZE is on but HF_TOKEN is unset — "
                    "speaker labels disabled."
                )
        except Exception:  # noqa: BLE001
            logger.exception("Diarizer load failed — speaker labels disabled.")
            self.diarizer = None
        finally:
            self._model_ready.set()

    @property
    def model_ready(self) -> bool:
        return self.transcriber is not None

    # ── queue ────────────────────────────────────────────────────────────
    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    async def _worker(self) -> None:
        await self._model_ready.wait()
        while True:
            job_id = await self._queue.get()
            try:
                await self._process(job_id)
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected error processing job %s", job_id)
            finally:
                self._queue.task_done()

    # ── pub/sub ──────────────────────────────────────────────────────────
    def subscribe(self, job_id: str) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers[job_id].add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[Event]) -> None:
        subs = self._subscribers.get(job_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subscribers.pop(job_id, None)

    def publish(self, job_id: str, event: str, data: dict) -> None:
        """Publish to subscribers. Must be called from the event loop thread."""
        for q in list(self._subscribers.get(job_id, ())):
            q.put_nowait({"event": event, "data": data})

    def publish_threadsafe(self, job_id: str, event: str, data: dict) -> None:
        """Publish from a worker thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.publish, job_id, event, data)

    # ── processing ───────────────────────────────────────────────────────
    async def _process(self, job_id: str) -> None:
        db_path = self.settings.db_path
        job = db.get_job(db_path, job_id)
        if job is None:
            return
        try:
            if self.transcriber is None:
                raise RuntimeError(
                    "Transcription model is not loaded — check server logs."
                )

            # 1. Convert to 16 kHz mono WAV.
            db.update_job(
                db_path, job_id, status=db.STATUS_CONVERTING,
                started_at=db.now_iso(), progress=0.0,
            )
            self.publish(job_id, "status", {
                "status": db.STATUS_CONVERTING, "progress": 0.0,
                "detected_language": None,
            })
            wav_path = str(self.settings.uploads_dir / f"{job_id}.wav")
            await asyncio.to_thread(audio.convert_to_wav, job["stored_path"], wav_path)
            duration = await asyncio.to_thread(audio.probe_duration, wav_path)
            db.update_job(
                db_path, job_id, wav_path=wav_path, duration=duration,
                status=db.STATUS_TRANSCRIBING,
            )
            self.publish(job_id, "status", {
                "status": db.STATUS_TRANSCRIBING, "progress": 0.0,
                "detected_language": None,
            })

            # 2. Transcribe, streaming segments to subscribers.
            def on_segment(seg, progress) -> None:
                self.publish_threadsafe(job_id, "segment", {
                    "start": seg.start, "end": seg.end, "text": seg.text,
                })
                self.publish_threadsafe(job_id, "status", {
                    "status": db.STATUS_TRANSCRIBING, "progress": progress,
                    "detected_language": None,
                })
                db.update_job(db_path, job_id, progress=progress)

            language = job.get("language") or "auto"
            segments, words, info = await asyncio.to_thread(
                self.transcriber.transcribe, wav_path, duration, language, on_segment
            )

            # 2b. Diarize + identify speakers (best-effort; never fails the job).
            segments, speakers_map, diarized = await asyncio.to_thread(
                self._diarize_and_identify, job_id, wav_path, segments, words
            )

            # 3. Write outputs + persist.
            meta = TranscriptMeta(
                filename=job["original_filename"],
                language=info.language,
                duration=info.duration,
                model=self.settings.model_name,
                msg_timestamp=job.get("msg_timestamp"),
                msg_timestamp_source=job.get("msg_timestamp_source"),
            )
            await asyncio.to_thread(
                self._write_outputs, job_id, segments, meta, speakers_map
            )
            transcript_text = "\n".join(s.text for s in segments if s.text)
            db.update_job(
                db_path, job_id, status=db.STATUS_DONE,
                detected_language=info.language, duration=info.duration,
                progress=1.0, transcript_text=transcript_text,
                diarized=1 if diarized else 0,
                speakers=json.dumps(speakers_map) if speakers_map else None,
                finished_at=db.now_iso(),
            )
            self.publish(job_id, "done", db.get_job(db_path, job_id))

        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            db.update_job(
                db_path, job_id, status=db.STATUS_ERROR, error=str(exc),
                finished_at=db.now_iso(),
            )
            self.publish(job_id, "error", {"message": str(exc)})

    def _write_outputs(
        self, job_id: str, segments, meta: TranscriptMeta, speakers: dict | None = None
    ) -> None:
        out = self.settings.outputs_dir
        (out / f"{job_id}.txt").write_text(formats.to_txt(segments, meta), encoding="utf-8")
        (out / f"{job_id}.srt").write_text(formats.to_srt(segments), encoding="utf-8")
        (out / f"{job_id}.vtt").write_text(formats.to_vtt(segments), encoding="utf-8")
        (out / f"{job_id}.json").write_text(
            formats.to_json(segments, meta, speakers or None), encoding="utf-8"
        )

    def _diarize_and_identify(self, job_id, wav_path, segments, words):
        """Best-effort diarization + global speaker identification.

        Returns ``(segments, speakers_map, diarized)``. On any failure or when
        diarization is disabled, returns the original segments unchanged.
        """
        if not (self.settings.diarize and self.diarizer is not None):
            return segments, {}, False
        try:
            turns, embeddings = self.diarizer.diarize(wav_path)
        except Exception:  # noqa: BLE001
            logger.exception("Diarization failed for job %s", job_id)
            return segments, {}, False
        if not turns:
            return segments, {}, False

        gallery = Gallery(
            self.settings.db_path, threshold=self.settings.speaker_threshold
        )
        # Total speaking time per local label, to identify the dominant voices
        # first (more stable assignment when several voices are present).
        label_dur: dict[str, float] = {}
        for t in turns:
            label_dur[t.speaker] = label_dur.get(t.speaker, 0.0) + (t.end - t.start)

        label_to_person: dict[str, str] = {}
        label_to_name: dict[str, str] = {}
        used: set[str] = set()
        for label in sorted(label_dur, key=label_dur.get, reverse=True):
            emb = embeddings.get(label)
            if emb is None:
                continue  # no embedding for this speaker; keep its local label
            pid, _created = gallery.assign_or_create(
                emb, job_id=job_id, exclude_ids=used
            )
            used.add(pid)
            person = db.get_person(self.settings.db_path, pid)
            label_to_person[label] = pid
            label_to_name[label] = person["name"] if person else label

        spk = (
            assign.words_to_speaker_segments(words, turns)
            if words
            else assign.segments_to_speaker_segments(segments, turns)
        )
        for s in spk:
            pid = label_to_person.get(s.speaker)
            if pid:
                s.person_id = pid
                s.speaker = label_to_name.get(s.speaker, s.speaker)

        speakers_map = {
            label: {
                "person_id": label_to_person.get(label),
                "name": label_to_name.get(label, label),
            }
            for label in label_dur
        }
        return spk, speakers_map, True
