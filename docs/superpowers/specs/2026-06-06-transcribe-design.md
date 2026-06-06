# fluister-webui — Local Transcription Service (Design)

Date: 2026-06-06

## Goal
A local, fully functional audio/video transcription service with a web UI,
supporting **Dutch and English**, that makes the most of the host GPU
(RTX 3080, 10 GB). Batch file upload (voice notes from messaging apps),
localhost only, no auth, no diarization.

## Host
Ryzen 7 9800X3D (16 threads), 60 GB RAM, RTX 3080 10 GB (CUDA), CachyOS,
`uv` + `ffmpeg` present. Python pinned to 3.12 (3.14 lacks ctranslate2 wheels).

## Stack
- **Engine:** `faster-whisper` `large-v3`, `float16`, `device=cuda`,
  `BatchedInferencePipeline` (`batch_size=8`), Silero VAD on. CPU `int8`
  fallback if CUDA is unavailable.
- **Backend:** FastAPI + uvicorn. Single background GPU worker draining an async
  job queue. SQLite history. SSE for live progress.
- **Audio:** `ffmpeg` normalizes any input → 16 kHz mono WAV.
- **Frontend:** static vanilla HTML/JS/CSS (no build step), served by FastAPI.
- **Deps:** managed with `uv`; CUDA libs via `nvidia-cublas-cu12` +
  `nvidia-cudnn-cu12` wheels; `run.sh` exports `LD_LIBRARY_PATH`.

## Module layout & contracts
Shared (already written): `app/models.py`, `app/config.py`, `app/db.py`.

### `app/models.py` (done)
- `Segment(start: float, end: float, text: str)`
- `TranscribeInfo(language: str, duration: float)`
- `TranscriptMeta(filename, language, duration, model, msg_timestamp=None, msg_timestamp_source=None)`

### `app/filename_time.py`
Extract the message timestamp from a filename; fall back to file mtime.
```python
@dataclass
class ParsedTime:
    dt: datetime          # timezone-naive local time
    source: str           # "filename" | "mtime"
    has_time: bool        # False when only a date was found

def parse_filename_timestamp(name: str) -> ParsedTime | None
def resolve_timestamp(path: str | Path) -> ParsedTime   # filename, else mtime
```
Patterns to cover (ordered, first match wins):
- `signal-2026-06-06-094449.aac`          → date + `HHMMSS`
- `signal-2026-06-04-16-21-28-808.m4a`    → date + `HH-MM-SS-mmm` (ms)
- `WhatsApp Audio 2026-06-06 at 09.44.49.opus`
- `PTT-20260606-WA0001.opus`              → date only (`has_time=False`)
- `audio_2026-06-06_09-44-49.ogg` (Telegram)
- generic `YYYYMMDD[_-]?HHMMSS`, `YYYY-MM-DD`, `YYYYMMDD`
Reject implausible values (month 1–12, day 1–31, hour <24, etc.).

### `app/audio.py`
```python
class AudioError(Exception): ...
def probe_duration(path: str | Path) -> float          # seconds (ffprobe)
def convert_to_wav(src: str | Path, dst: str | Path) -> None
    # ffmpeg -i src -ac 1 -ar 16000 -c:a pcm_s16le dst ; raise AudioError(stderr tail) on failure
```

### `app/formats.py`
```python
def to_txt(segments: list[Segment], meta: TranscriptMeta) -> str   # header + plain text
def to_srt(segments: list[Segment]) -> str
def to_vtt(segments: list[Segment]) -> str
def to_json(segments: list[Segment], meta: TranscriptMeta) -> str  # {meta, segments:[{start,end,text}]}
```
SRT times `HH:MM:SS,mmm`; VTT `HH:MM:SS.mmm` with `WEBVTT` header.

### `app/transcriber.py`
```python
class Transcriber:
    def __init__(self, model_name, device, compute_type, batch_size, use_vad): ...
        # resolve device "auto"->cuda if ctranslate2 sees a GPU else cpu;
        # compute_type "auto"->float16 on cuda, int8 on cpu.
        # Load WhisperModel once + wrap in BatchedInferencePipeline.
    def transcribe(self, wav_path, duration: float, language: str | None = None,
                   on_segment=None) -> tuple[list[Segment], TranscribeInfo]:
        # language None/"auto" -> autodetect. on_segment(seg, progress_0_1) per segment.
        # On CUDA OOM: retry smaller batch_size, then non-batched; surface clear error.
```

### `app/queue.py` + `app/main.py` (orchestrator)
- Async queue + single worker task running the blocking transcribe in a thread.
- Per-job SSE pub/sub (asyncio.Queue per subscriber).
- Worker: convert → transcribe (stream segments to SSE) → write outputs → done/error.
- On startup: `db.mark_interrupted`, `config.ensure_dirs`, load model.

### API
- `POST /api/jobs` (multipart, 1+ `files`, optional `language`) → `[{id,...}]`
- `GET /api/jobs` → history; `GET /api/jobs/{id}` → detail incl. transcript
- `GET /api/jobs/{id}/events` → SSE: `status` / `segment` / `done` / `error`
- `GET /api/jobs/{id}/download/{fmt}` → `txt|srt|vtt|json`
- `DELETE /api/jobs/{id}` → remove job + files
- `GET /` → SPA

### SSE event shapes
- `status`: `{status, progress, detected_language}`
- `segment`: `{start, end, text}`
- `done`: full job dict
- `error`: `{message}`

## DB job dict keys
`id, original_filename, stored_path, wav_path, msg_timestamp,
msg_timestamp_source, msg_has_time, language, detected_language, duration,
status, error, progress, transcript_text, model_name, created_at, started_at,
finished_at`. Outputs at `data/outputs/{id}.{txt,srt,vtt,json}`.

## Errors / resilience
ffmpeg & empty-audio → job `error` with message. CUDA OOM → retry/fallback.
No CUDA → CPU int8 with warning. Restart → in-flight jobs → `interrupted`.

## Testing
Unit: `filename_time` (table incl. the two Signal examples), `formats` (timestamp
formatting), `audio` (duration probe, generated tone). API: TestClient with a
mocked transcriber → upload → status transitions → downloads. Optional slow:
short clip through `tiny` model end-to-end.
