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
- **Output formats** — TXT, SRT, VTT, JSON.
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

## Tests

```bash
uv run pytest                 # fast suite (no GPU, no model download)
RUN_SLOW_TESTS=1 uv run pytest # also runs an end-to-end tiny-model test
```

## How it works

```
upload → SQLite job (queued) → single GPU worker:
   ffmpeg → 16 kHz mono wav → faster-whisper (batched) → txt/srt/vtt/json
                                   │
        live segments ── SSE ──────┘→ browser
```

A single background worker processes one job at a time (the GPU is the
bottleneck); additional uploads queue up. Modules: `transcriber` (engine),
`audio` (ffmpeg), `formats` (writers), `filename_time` (timestamp parser),
`queue` (worker + SSE), `db` (SQLite), `main` (FastAPI + static UI).
