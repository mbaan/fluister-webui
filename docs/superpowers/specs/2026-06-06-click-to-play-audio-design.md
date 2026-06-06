# Click-to-play audio at word offset — design

Date: 2026-06-06

## Goal

In the transcript view, clicking a word plays the recording from that word's
offset, with the word under the playhead highlighted, so the reviewer can
instantly hear what was actually said. Motivation: ASR mishearings (names,
idioms, homophones) are unavoidable and **not** fixable after the fact — a
June-2026 bake-off showed no engine swap (large-v2/v3, Voxtral) and no LLM
correction pass recovers them, because the acoustic signal is gone from the
text. The highest-leverage response is therefore one-click audio verification,
not error prevention.

## Scope

- Word-level click-to-seek **and** play.
- Live highlight of the word under the playhead while audio plays.
- Minimal sticky player: play/pause + `current / duration`.
- No model change, no LLM, no change to diarization.

## Non-goals / parked

- Per-person hotwords (own future spec).
- Scrubber / playback-speed controls (chose the lightweight player).
- Backfilling existing transcripts — the user will re-transcribe everything, so
  **no backward-compat path**. A job that somehow lacks stored words degrades to
  plain, non-clickable text (defensive only, not a feature).

## 1. Persistence (`app/queue.py`, `app/assign.py`)

The transcriber already returns a word list (`Word{start,end,word}`) which is
currently used for diarization and then discarded. We keep it.

- New pure helper in `app/assign.py`:
  `attach_words_to_segments(segments, words) -> list[dict]`. Assigns each `Word`
  to the segment whose `[start, end]` contains the word's midpoint; returns the
  segment payload dicts, each with a `words: [{start, end, word}]` list. Words
  outside every segment are dropped. Empty `words` → every segment gets `[]`.
- `_process` builds `segments_payload` via this helper (replacing the current
  inline comprehension), so each persisted segment carries its words.
- Persisted segment shape:
  ```json
  { "start", "end", "text", "speaker", "person_id",
    "words": [ { "start", "end", "word" }, … ] }
  ```
- **No DB migration** — `segments_json` is already a JSON blob.
  `GET /api/jobs/{id}/transcript` returns the `words` field automatically.

## 2. Audio endpoint (`app/main.py`)

`GET /api/jobs/{job_id}/audio`:

- 404 if the job is unknown or `wav_path` is absent / not on disk.
- Otherwise `FileResponse(wav_path, media_type="audio/wav")`. Starlette's
  `FileResponse` honors HTTP `Range` (responds `206` + `Accept-Ranges: bytes`),
  which is what lets the browser seek.
- Localhost-only, no auth — consistent with the rest of the app.

Serving the 16 kHz mono WAV (not the original upload) guarantees browser
playback and an exact timeline match with the word timestamps (which were
computed from that WAV).

## 3. Frontend (`app/static/app.js`, `index.html`, stylesheet)

- **Render words:** in the transcript render paths (plain and diarized), when a
  segment has `words`, render its tokens as
  `<span class="word" data-start="S">word</span>` joined by spaces, preserving
  the existing speaker grouping/colors. No words → plain text node (unchanged).
- **Player:** lazily create one `<audio>` per opened job card, plus a small
  sticky bar: play/pause button + `mm:ss / mm:ss`. `audio.src` =
  `/api/jobs/{id}/audio`.
- **Click a word:** `audio.currentTime = max(0, start - 0.25)` then
  `audio.play()` (0.25 s lead so the word's onset isn't clipped). If audio isn't
  ready yet, set `src`, await `canplay`, then seek + play.
- **Live highlight:** on `timeupdate`, find the word whose `[start, nextStart)`
  contains `currentTime`; move `.word--active` to it (cleared on pause/end).
- **Lifecycle:** collapsing / closing / switching a job stops and resets its
  audio.

## 4. Error handling

- Missing audio → the bar shows "audio unavailable"; words still render, clicks
  are no-ops.
- Click before metadata is loaded → handled by the `canplay` await.
- A word missing `end` → bounded by the next word's `start` (last word: segment
  `end`).

## 5. Testing

- **Unit (pytest):** `attach_words_to_segments` — correct segment assignment,
  boundary cases, empty `words`, words outside all segments.
- **API (FastAPI `TestClient`):** audio endpoint → `200` + `Accept-Ranges` for a
  job with a wav; a `Range` request → `206` partial; `404` for unknown job /
  missing file.
- **Persistence:** the segments payload built in `_process` carries `words` per
  segment (covered via the helper unit test; optionally assert in the existing
  slow end-to-end test).
- **Frontend:** no JS test harness (vanilla JS) → manual verification: run the
  app, open a transcript, click words, confirm seek + highlight + play/pause.

## Files touched

- `app/assign.py` — `attach_words_to_segments` helper.
- `app/queue.py` — use the helper when building `segments_payload`.
- `app/main.py` — `GET /api/jobs/{id}/audio`.
- `app/static/app.js` — word spans, player, highlight, lifecycle.
- `app/static/index.html` + stylesheet — player markup + `.word` / `.word--active` styles.
- `tests/` — helper unit test, audio endpoint test.
