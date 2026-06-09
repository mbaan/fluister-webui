# LLM tidy pass → readable transcript view — design

Date: 2026-06-09

## Goal

Raw `large-v3` transcripts of long voice notes are wall-of-text: sparse
punctuation, no paragraphs, full of "uhm"/false starts. Make them *readable*
without touching the audio-accurate transcript we already trust.

The June-2026 ASR eval settled the approach (see memory `model-evaluation-2026-06`):
faster-whisper punctuation is baked in, `initial_prompt` is harmful (it leaks into
the transcript and triggers repetition loops), so the readability fix is a
**local-LLM post-pass that *tidies, never fixes*** — punctuation, capitalization,
paragraphs, filler removal — while preserving every meaningful word and Dutch⇄EN
code-switching. It does **not** recover acoustic mishearings (that's the ASR
stage's job, already handled via per-person `hotwords`).

## Decisions (locked)

- **Engine:** keep faster-whisper `large-v3` untouched. The tidier is a separate
  LLM step. Never quantize whisper to make room (accuracy guardrail).
- **Model:** Qwen3-30B-A3B (GGUF, Q4_K_M) via `llama-server`, run with `--cpu-moe`
  so MoE experts live in the 60 GB system RAM and the GPU footprint stays small.
  Validated for Dutch/EN tidying in the June eval; 30B over-corrects less than 7B.
- **Trigger:** automatic, **best-effort** — every job tidies after diarization *if*
  the LLM is reachable; if not, the job still completes and simply has no readable
  view (mirrors the diarizer's degrade-gracefully posture).
- **Scope:** baseline punctuation + capitalization + paragraph breaks, **plus
  filler/false-start removal**. No name/term glossary (kept maximally conservative).
- **Two views, raw is source of truth:** the existing timestamped + diarized
  click-to-play transcript is never overwritten. The tidied text is a *second view*
  the user toggles to, so they can always audit what the LLM changed.
- **Lifecycle:** fluister **spawns** `llama-server` on startup and **stops** it on
  shutdown — full ownership, with a kernel backstop so it can never orphan.
- **Co-residence (hard requirement):** llama-server + whisper + diarizer must stay
  resident together under 10 GB VRAM. Verified manually with `nvidia-smi`.

## Non-goals / parked

- No error/fact correction, rephrasing, translation, or summarization.
- No glossary/name-spelling pass (parked; person `keywords` already bias the ASR).
- No streaming of the tidied text — it's produced once, after the full transcript.
- No editing of the tidied text in the UI.
- No re-tidy button; re-running the file (existing re-diarize action) regenerates
  it. If the raw transcript changes, the tidied view is stale until re-run — out of
  scope to auto-invalidate.

## VRAM budget (RTX 3080, 10 GB)

| Resident | VRAM |
|---|---|
| Desktop idle | ~1.0 GB |
| faster-whisper large-v3 (fp16) | ~4.7 GB |
| pyannote diarizer | ~1.5 GB |
| **Headroom for the LLM** | **~2.8 GB** |

Qwen3-30B-A3B `--cpu-moe` measured ~3.4 GB standalone, so it must be squeezed under
the headroom. Launch flags: `--cpu-moe -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0`
(experts to RAM, all non-expert layers on GPU, 8k context, quantized KV cache).
If it still doesn't fit, the documented fallback is a smaller quant / lower `-ngl`
— **never** quantizing whisper.

## 1. LLM subprocess supervisor (`app/llm_server.py`)

New `LlamaServer` class, no whisper/torch deps. Owns the full lifecycle:

- `start()` — if `TRANSCRIBE_TIDY` is on and the GGUF path exists, spawn
  `llama-server -m <model> --port <port> --cpu-moe -ngl 99 -c <ctx> -ctk q8_0
  -ctv q8_0`. Poll `GET /health` until ready or a timeout (~120 s). On any failure
  (binary missing, model missing, OOM, health timeout) set `available = False` and
  log — never raise into the app.
- **Spawn isolation:** `start_new_session=True` (own process group) so the whole
  group can be signalled together.
- **Orphan backstop (Linux):** `preexec_fn` sets `PR_SET_PDEATHSIG` (SIGKILL) so if
  fluister dies abnormally (SIGKILL / crash / OOM-killer) the kernel reaps the
  child — covers what a shutdown handler can't.
- **Stale-port guard:** if the configured port is already bound at `start()`, do
  not spawn a second copy — log and leave `available = False`. Never stack two
  LLMs in VRAM; the user resolves the stale process, then restarts.
- `stop()` — `SIGTERM` the process group, wait a ~10 s grace, then `SIGKILL`.
  Always safe to call, even after a partial/failed start.
- Properties: `available: bool`, `base_url: str`.

## 2. Tidier (`app/tidier.py`)

Pure, HTTP-only, testable against a fake endpoint — no whisper/torch deps.

```python
def group_turns(segments: list[Segment]) -> list[Turn]        # (speaker, text)
def tidy_turns(turns, base_url, *, timeout) -> list[dict]      # [{speaker, text}]
```

- `group_turns`: merge **consecutive** segments sharing a speaker into one turn
  (single-speaker note → one turn). Preserves attribution; the LLM never sees two
  speakers at once, so it can't move words across them.
- Long-turn guard: split a turn at segment boundaries if its text exceeds a token/
  char budget (keep prompts well inside the 8k context); tidy each chunk and
  concatenate.
- `tidy_turns`: for each turn (chunk), POST to the OpenAI-compatible
  `/v1/chat/completions` with temperature ~0.1 and the system prompt below; collect
  `{speaker, text}`. On a per-turn HTTP/parse error, fall back to that turn's raw
  text (degrade gracefully rather than drop content).

**System prompt (the tidier-not-fixer contract):**

> You clean up speech-to-text transcripts for readability. Add punctuation,
> capitalization, and paragraph breaks. Remove filler words (uh, um, like, you
> know) and false starts / repeated restarts. Do NOT change, add, remove
> (other than fillers), reorder, translate, or "correct" any meaningful word.
> Preserve the original language(s) exactly, including Dutch⇄English
> code-switching. Output only the cleaned text, nothing else.

## 3. Pipeline integration (`app/queue.py`)

Order so a restart can **never strand a job mid-tidy**: persist the transcript as
`DONE` first (the note is immediately usable), then enrich with the tidied view
best-effort, delaying only the *live* `done` SSE so a watching client keeps its
stream open until the readable view arrives.

In `_process`, after diarization:

1. `update_job(..., status=DONE, transcript_text, segments_json, speakers, ...)`
   exactly as today — but **do not publish `done` yet** if tidying will run.
2. If `self.llm_server is not None and self.llm_server.available`:
   publish a transient `status` event with `status="tidying"` (this is an SSE-only
   signal, **not** a persisted job status — the row is already `DONE`), then
   `tidied = await asyncio.to_thread(tidy_turns, group_turns(segments), base_url)`,
   `update_job(tidied_json=...)`, and finally publish `done` (its payload is
   `db.get_job(...)`, now carrying `tidied_json`).
3. If the LLM is unavailable, publish `done` immediately after step 1.
4. Wrap the tidy block in try/except: on failure log, leave `tidied_json` NULL, and
   still publish `done`. Same best-effort contract as `_diarize_and_identify`.

Because the row is `DONE` before tidying starts, a restart mid-tidy leaves a valid
job that simply lacks a readable view — no stuck state, no recovery special-casing.

`JobQueue` gains a `LlamaServer`, created/started in `start()` (alongside model
load) and `stop()`ped in `stop()`. The FastAPI lifespan shutdown must call
`JobQueue.stop()` so teardown always runs.

## 4. Data model (`app/db.py`)

- Add nullable `tidied_json TEXT` to `jobs`; extend `_migrate()` with the same
  `PRAGMA table_info` guard used for the other late columns.
- `tidied_json` shape: `[{"speaker": str | null, "text": str}, ...]` (speaker-
  labeled paragraphs). Raw `transcript_text` / `segments_json` are never modified.
- No new persisted job status. `"tidying"` is an **SSE-only** signal (§3); the job
  row goes straight to `DONE` before tidying, so no recovery/`ACTIVE_STATUSES`
  change is needed.

## 5. Config (`app/config.py`)

New `Settings` fields + env:

- `TRANSCRIBE_TIDY` (bool, default `true`) → `tidy_enabled`
- `TRANSCRIBE_LLM_MODEL` (GGUF path; if missing/unset → tidy disabled) → `llm_model`
- `TRANSCRIBE_LLM_PORT` (default `8080`) → `llm_port`
- `TRANSCRIBE_LLM_CTX` (default `8192`) → `llm_ctx`
- `TRANSCRIBE_LLM_HEALTH_TIMEOUT` (default `120`) → `llm_health_timeout`

Document the resolved `llama-server` launch command + the new env in
`.env.example` and `README.md`.

## 6. Frontend (`app/static/app.js`, `index.html`, `style.css`)

- A **Raw / Readable** toggle on the note view.
- **Readable** renders `tidied_json`: speaker-labeled paragraphs, no per-word
  timestamps, no click-to-play. **Default to Readable when `tidied_json` exists**,
  else Raw.
- **Raw** is the existing timestamped + diarized + click-to-play transcript,
  unchanged. Toggle always available so the user can audit the LLM's edits.
- Handle the `tidied` SSE event (and the `tidying` status) so a note being viewed
  live gains its readable view when the pass completes.

## 7. Testing (TDD)

- `tidier.group_turns`: consecutive same-speaker merge; speaker-change splits;
  single-speaker → one turn; long-turn splitting at segment boundaries.
- `tidier.tidy_turns`: builds the right prompt and parses the response against a
  **fake HTTP server**; per-turn error falls back to raw text.
- `llm_server.LlamaServer`: disabled when model path missing / tidy off;
  `stop()` invokes teardown on a dummy/fake process; stale-port guard. No real
  `llama-server` spawned in tests.
- `queue._process`: job reaches `done` with `tidied_json` NULL when the LLM is
  unavailable; `tidied_json` populated when a fake tidier is injected; a `tidying`
  status event is published while it runs.
- `db`: migration adds `tidied_json`; round-trips.
- `config`: new fields parse from env with correct defaults.

## 8. Manual verification (hard requirement)

With the model present, start fluister, transcribe a real note, and confirm via
`nvidia-smi` that llama-server + whisper + diarizer stay co-resident under 10 GB.
Kill fluister with `SIGKILL` and confirm (via `nvidia-smi` / `pgrep llama-server`)
that the child does not orphan. Confirm the readable view drops fillers and reads
cleanly, and that toggling to Raw shows the untouched original with click-to-play
still working.
