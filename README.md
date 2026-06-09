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
- **Readable transcripts** — an optional, best-effort local-LLM pass turns the
  raw transcript into a punctuated, paragraphed, filler-free **Readable** view
  (Dutch/English preserved). It *tidies, never fixes* — your words are kept, just
  cleaned up. Toggle back to **Raw** any time to see the untouched, timestamped,
  click-to-play transcript.
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

### Optional: readable transcripts (local LLM)

A best-effort post-pass cleans each transcript into a **Readable** view
(punctuation, paragraphs, "uhm" removal). It runs a local
[`llama-server`](https://github.com/ggml-org/llama.cpp) that **fluister starts
and stops itself**.

It's **on by default** and resolves its model the same way whisper does — by HF
identifier, into `~/.cache/huggingface` on first use (so it's listed by
`hf cache ls` and removable with `hf cache rm`). The default is
**Qwen3-30B-A3B-Instruct-2507** (Q4_K_M, ~18.6 GB) — a non-thinking MoE that fits
alongside whisper via `--cpu-moe` — so the **first run downloads ~18.6 GB**.

- Disable it: `TRANSCRIBE_TIDY=false`.
- Use a different HF GGUF: `TRANSCRIBE_LLM_REPO` + `TRANSCRIBE_LLM_FILE`.
- Use a local file instead of the cache: `TRANSCRIBE_LLM_MODEL=/abs/path/to.gguf`.

fluister launches it co-resident with whisper on the GPU using
`llama-server -m <model> --cpu-moe -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0`
(MoE experts offloaded to system RAM so it fits alongside whisper + diarizer on a
10 GB card). If the model isn't set or the server can't start, transcription is
unaffected — there's just no Readable view. The tidier *tidies, never fixes*: it
won't recover misheard words (bias names via per-person keywords instead).

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
| `TRANSCRIBE_BATCH_SIZE` | `2` | batched-inference batch size (kept low so whisper fits next to the co-resident tidy LLM; raise it if `TRANSCRIBE_TIDY=false`) |
| `TRANSCRIBE_LANGUAGE` | `auto` | default language (`auto` / `nl` / `en`) |
| `TRANSCRIBE_VAD` | `true` | Silero voice-activity filtering |
| `TRANSCRIBE_HOST` / `TRANSCRIBE_PORT` | `127.0.0.1` / `8000` | bind address |
| `TRANSCRIBE_DATA_DIR` | `./data` | uploads, outputs, SQLite db |
| `HF_TOKEN` | — | HuggingFace token; enables speaker diarization |
| `TRANSCRIBE_DIARIZE` | `true` | run diarization when a token is present |
| `TRANSCRIBE_DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | pyannote pipeline |
| `TRANSCRIBE_SPEAKER_THRESHOLD` | `0.45` | cosine similarity to match a known voice |
| `TRANSCRIBE_MIN_SPEAKER_SECONDS` | `2.0` | ignore diarized speakers with less total speech (filters noise) |
| `TRANSCRIBE_TIDY` | `true` | enable the readable LLM tidy pass |
| `TRANSCRIBE_LLM_REPO` | `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | HF repo for the tidier GGUF (cached like whisper) |
| `TRANSCRIBE_LLM_FILE` | `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` | GGUF filename within that repo |
| `TRANSCRIBE_LLM_MODEL` | — | optional local GGUF path; overrides repo/file |
| `TRANSCRIBE_LLM_PORT` | `8080` | port fluister runs `llama-server` on |
| `TRANSCRIBE_LLM_CTX` | `8192` | `llama-server` context size |
| `TRANSCRIBE_LLM_HEALTH_TIMEOUT` | `120` | seconds to wait for `llama-server` `/health` |
| `TRANSCRIBE_LLM_REQUEST_TIMEOUT` | `120` | seconds per tidy request |

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
to the closest match above the threshold, else create a new person). Finally, if
a tidier LLM is configured, a best-effort pass produces the Readable view — the
job is already marked `done` first, so this never blocks or strands it. Modules:
`transcriber` (engine), `diarizer` (pyannote), `assign` (word↔speaker),
`speakers` (voice gallery), `audio` (ffmpeg), `tidier` (LLM readable pass),
`llm_server` (llama-server lifecycle), `filename_time` (timestamp parser),
`queue` (worker + SSE), `db` (SQLite), `main` (FastAPI + static UI).
