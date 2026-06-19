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

from app import assign, audio, db
from app.config import Settings
from app.llm_server import LlamaServer
from app.speakers import Gallery, build_hotwords
from app.insights import generate_insights
from app.tidier import group_turns, tidy_turns

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
        self.llm_server: Any = self._default_llm_server()

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

    def _default_llm_server(self) -> Any:
        s = self.settings
        return LlamaServer(
            enabled=s.tidy_enabled,
            repo=s.llm_repo,
            file=s.llm_file,
            token=s.hf_token,
            model_path=s.llm_model,
            port=s.llm_port,
            ctx=s.llm_ctx,
            health_timeout=s.llm_health_timeout,
        )

    # ── lifecycle ────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._load_task = asyncio.create_task(self._load_model())
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        # Stop the LLM first so its VRAM frees promptly on shutdown.
        if self.llm_server is not None:
            self.llm_server.stop()
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

        # Transcription can start now — the tidier LLM is optional enrichment and
        # must NOT gate readiness (its weights may download from HF on first use).
        self._model_ready.set()

        # Start the tidier LLM best-effort, after readiness so a first-run model
        # download never blocks transcription.
        try:
            await asyncio.to_thread(self.llm_server.start)
        except Exception:  # noqa: BLE001
            logger.exception("llama-server start failed — readable view disabled.")

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
            # Progress is monotonic within a transcription pass; a drop means the
            # OOM fallback restarted from the top, so tell the client to discard
            # the partial segments it already received (otherwise they duplicate).
            stream = {"last": 0.0, "persisted": 0.0}

            def on_segment(seg, progress) -> None:
                if progress + 1e-6 < stream["last"]:
                    self.publish_threadsafe(job_id, "reset", {})
                    stream["persisted"] = 0.0
                stream["last"] = progress
                self.publish_threadsafe(job_id, "segment", {
                    "start": seg.start, "end": seg.end, "text": seg.text,
                })
                self.publish_threadsafe(job_id, "status", {
                    "status": db.STATUS_TRANSCRIBING, "progress": progress,
                    "detected_language": None,
                })
                # Persist progress sparingly: it only feeds the polling progress
                # bar (live clients get it over SSE), so a write per decoded
                # segment is needless DB churn on a long file. A >=1% step keeps
                # the polled bar fresh enough (poll interval is seconds).
                if progress - stream["persisted"] >= 0.01:
                    stream["persisted"] = progress
                    db.update_job(db_path, job_id, progress=progress)

            language = job.get("language") or "auto"
            # Bias the decoder toward known names/keywords (union across the
            # whole gallery — we don't know the speaker until diarization runs).
            hotwords = build_hotwords(db.list_persons(db_path))
            segments, words, info = await asyncio.to_thread(
                self.transcriber.transcribe,
                wav_path, duration, language, on_segment, hotwords,
            )

            # 2b. Diarize + identify speakers (best-effort; never fails the job).
            segments, speakers_map, diarized = await asyncio.to_thread(
                self._diarize_and_identify, job_id, wav_path, segments, words
            )

            # 3. Persist results (segments + speakers stored in the DB) before the
            # tidy pass, so an interruption mid-tidy can never lose the transcript
            # (startup recovery flips a stranded TIDYING job back to DONE). The
            # TIDYING status is persisted so polling clients see the readable view
            # being prepared instead of a silent DONE; progress restarts at 0 and
            # climbs per tidied turn.
            will_tidy = self.llm_server is not None and self.llm_server.available
            segments_payload = assign.attach_words_to_segments(segments, words)
            transcript_text = "\n".join(s.text for s in segments if s.text)
            db.update_job(
                db_path, job_id,
                status=db.STATUS_TIDYING if will_tidy else db.STATUS_DONE,
                detected_language=info.language, duration=info.duration,
                progress=0.0 if will_tidy else 1.0,
                transcript_text=transcript_text,
                segments_json=json.dumps(segments_payload, ensure_ascii=False),
                diarized=1 if diarized else 0,
                speakers=json.dumps(speakers_map) if speakers_map else None,
                finished_at=db.now_iso(),
            )

            # 4. Best-effort readable tidy. The transcript is already safe; the
            # `done` event is delayed so a watching client keeps its stream open
            # until the readable view arrives.
            if will_tidy:
                self.publish(job_id, "status", {
                    "status": db.STATUS_TIDYING, "progress": 0.0,
                    "detected_language": info.language,
                })
                tidied = await asyncio.to_thread(self._maybe_tidy, job_id, segments)
                if tidied is not None:
                    db.update_job(
                        db_path, job_id,
                        tidied_json=json.dumps(tidied, ensure_ascii=False),
                    )
                    self.publish(job_id, "tidied", {"tidied": tidied})

                # On-device insight pass (summary / key points / chapters). Also
                # best-effort and additive — it never touches the transcript. The
                # job stays TIDYING ("Polishing…") through this short window.
                insights = await asyncio.to_thread(self._maybe_insights, job_id, segments)
                fields: dict[str, Any] = {"status": db.STATUS_DONE, "progress": 1.0}
                if insights is not None:
                    fields["insights_json"] = json.dumps(insights, ensure_ascii=False)
                db.update_job(db_path, job_id, **fields)
                if insights is not None:
                    self.publish(job_id, "insights", {"insights": insights})

            self.publish(job_id, "done", db.get_job(db_path, job_id))

        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            db.update_job(
                db_path, job_id, status=db.STATUS_ERROR, error=str(exc),
                finished_at=db.now_iso(),
            )
            self.publish(job_id, "error", {"message": str(exc)})

    def _maybe_tidy(self, job_id: str, segments) -> list[dict] | None:
        """Best-effort readable tidy. Returns paragraphs or None (LLM down / error).
        Runs in a worker thread; per-turn progress goes to SSE subscribers and to
        the DB row (for polling clients)."""
        if not (self.llm_server is not None and self.llm_server.available):
            return None
        try:
            turns = group_turns(segments)
            if not turns:
                return None

            def on_progress(done: int, total: int) -> None:
                progress = done / total
                self.publish_threadsafe(job_id, "status", {
                    "status": db.STATUS_TIDYING, "progress": progress,
                    "detected_language": None,
                })
                # One DB write per turn is cheap next to the LLM call behind it.
                db.update_job(self.settings.db_path, job_id, progress=progress)

            return tidy_turns(
                turns, self.llm_server.base_url,
                timeout=self.settings.llm_request_timeout,
                on_progress=on_progress,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Tidy pass failed for job %s", job_id)
            return None

    def _maybe_insights(self, job_id: str, segments) -> dict | None:
        """Best-effort summary / key points / chapters via the same llama-server.
        Returns the insight dict or None (LLM down / error). Runs in a worker
        thread; never raises into the pipeline."""
        if not (self.llm_server is not None and self.llm_server.available):
            return None
        try:
            return generate_insights(
                segments, self.llm_server.base_url,
                timeout=self.settings.llm_request_timeout,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Insight pass failed for job %s", job_id)
            return None

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
        min_secs = self.settings.min_speaker_seconds
        # Total speaking time per local label, to identify the dominant voices
        # first (more stable assignment when several voices are present).
        label_dur: dict[str, float] = {}
        for t in turns:
            label_dur[t.speaker] = label_dur.get(t.speaker, 0.0) + (t.end - t.start)

        label_to_person: dict[str, str] = {}
        label_to_name: dict[str, str] = {}
        used: set[str] = set()
        for label in sorted(label_dur, key=label_dur.get, reverse=True):
            # Skip short/sporadic clusters (background noise, crosstalk): don't
            # enrol them as a person — their segments stay unlabeled.
            if label_dur[label] < min_secs:
                continue
            emb = embeddings.get(label)
            if emb is None:
                continue  # no embedding for this speaker
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
                s.speaker = label_to_name[s.speaker]
            else:
                s.speaker = None  # dropped/short speaker -> unlabeled text

        speakers_map = {
            label: {"person_id": label_to_person[label], "name": label_to_name[label]}
            for label in label_to_person
        }
        return spk, speakers_map, bool(label_to_person)
