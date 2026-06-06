# fluister

Local, on-device audio & video transcription with a web UI — Dutch & English,
GPU-accelerated with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(Whisper `large-v3`). Drop in voice notes from messaging apps and get text back.
Nothing leaves your machine.

## Features

- **Web UI** — drag & drop upload, live transcription streaming (SSE), job
  history, one-click downloads, copy to clipboard.
- **Dutch + English** with automatic language detection (or force per upload).
- **Any common format** — m4a, aac, opus, ogg, mp3, wav, mp4, mov, webm, … —
  anything `ffmpeg` can read; auto-converted to 16 kHz mono.
- **Filename timestamps** — deduces when a message was sent from its filename
  (Signal / WhatsApp / Telegram patterns), falling back to the file date.
- **Speaker recognition** — optional [pyannote](https://github.com/pyannote/pyannote-audio)
  diarization labels who spoke when, and a **global voice gallery** recognizes
  the same person across files. Rename/merge people on the Speakers page; labels
  show as colored chips and in the outputs.
- **Duplicate detection** — re-uploading a file with the same name + size is
  skipped (with a notice) instead of being transcribed again.
- **Fast** — batched GPU inference (`large-v3`, float16) with automatic
  CUDA-OOM and CPU fallbacks.
- **Private** — binds to `127.0.0.1` only, no auth, no cloud.

## Requirements

- `ffmpeg` + `ffprobe` on `PATH`
- [`uv`](https://docs.astral.sh/uv/)
- NVIDIA GPU (CUDA) recommended; CPU works but is much slower. The CUDA runtime
  libs (cuBLAS / cuDNN) are installed automatically as pip wheels — no system
  CUDA toolkit needed.

## Setup

```bash
uv sync
```

The first transcription downloads the `large-v3` model (~3 GB) to your Hugging
Face cache; it's reused afterwards.

### Optional: speaker recognition

Diarization uses gated pyannote models, so it needs a (free) HuggingFace token:

1. Accept the terms at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. `cp .env.example .env` and set `HF_TOKEN=hf_...` (`.env` is gitignored).

Without a token, transcription still works — only speaker labels are disabled.
To add speakers to files transcribed earlier, use **Re-diarize**
(`POST /api/jobs/{id}/rediarize`).

## Run

```bash
./run.sh
```

Then open <http://127.0.0.1:8000>.

`run.sh` points `LD_LIBRARY_PATH` at the bundled cuBLAS/cuDNN libs; the app also
preloads them in-process, so `uv run uvicorn app.main:app` works as well.

## Configuration

All optional, via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `TRANSCRIBE_MODEL` | `large-v3` | faster-whisper model |
| `TRANSCRIBE_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `TRANSCRIBE_COMPUTE_TYPE` | `auto` | `auto` / `float16` / `int8` / … |
| `TRANSCRIBE_BATCH_SIZE` | `8` | batched-inference batch size |
| `TRANSCRIBE_LANGUAGE` | `auto` | default language (`auto` / `nl` / `en`) |
| `TRANSCRIBE_VAD` | `true` | Silero voice-activity filtering |
| `TRANSCRIBE_HOST` / `TRANSCRIBE_PORT` | `127.0.0.1` / `8000` | bind address |
| `TRANSCRIBE_DATA_DIR` | `./data` | uploads, outputs, SQLite db |
| `HF_TOKEN` | — | HuggingFace token; enables speaker diarization |
| `TRANSCRIBE_DIARIZE` | `true` | run diarization when a token is present |
| `TRANSCRIBE_DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | pyannote pipeline |
| `TRANSCRIBE_SPEAKER_THRESHOLD` | `0.45` | cosine similarity to match a known voice |

## Tests

```bash
uv run pytest                 # fast suite (no GPU, no model download)
RUN_SLOW_TESTS=1 uv run pytest # also runs an end-to-end tiny-model test
```

## How it works

```
upload → SQLite job (queued) → single GPU worker:
   ffmpeg → 16 kHz mono wav → faster-whisper (words) ─┐
                                   │                   ├→ assign speakers
        live segments ── SSE ──────┘→ browser          │   → identify vs gallery
                          pyannote diarize ────────────┘   → transcript (SQLite)
```

A single background worker processes one job at a time (the GPU is the
bottleneck); additional uploads queue up. After transcription, an optional
pyannote pass diarizes the audio, words are aligned to speaker turns, and each
turn's voice embedding is matched against the **global person gallery** (assign
to the closest match above the threshold, else create a new person). Modules:
`transcriber` (engine), `diarizer` (pyannote), `assign` (word↔speaker),
`speakers` (voice gallery), `audio` (ffmpeg),
`filename_time` (timestamp parser), `queue` (worker + SSE), `db` (SQLite),
`main` (FastAPI + static UI).
