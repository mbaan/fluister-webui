# Per-person hotwords — design

Date: 2026-06-06

## Goal

Stop faster-whisper from mishearing names. A June-2026 bake-off showed misheard
**names** (Jolis, Tijn, Xenos, Praxis, Energiehaven) are the biggest and only
acoustically-fixable error bucket — once mis-transcribed the audio signal is gone,
so no LLM/glossary post-pass recovers them. The fix has to happen *at the ASR
stage*, by biasing the decoder via faster-whisper's `hotwords=` parameter (NOT
`initial_prompt`, which leaks into the transcript and triggers repetition loops).

Each person in the voice gallery gets a keyword list — the people, places, and
jargon they habitually mention. At transcription time we feed the **union** of
all persons' keywords (plus their own names) as hotwords.

## Scope

- A per-person freetext `keywords` field, edited on the existing Speakers page.
- A person's display name is auto-included as a hotword (placeholder `Person N`
  names excluded).
- The union of all persons' names + keywords is fed as `hotwords` on every
  transcription.

## Non-goals / parked

- **No global keyword list** — place/brand/jargon words ride along on whichever
  person mentions them.
- **No per-person two-pass scoping** (diarize-then-retranscribe). We don't know
  the speaker at transcription time; the union is the pragmatic single-pass start.
  If the union ever gets noisy as the gallery grows, revisit then.
- No keyword auto-suggestion.
- No backfill action — hotwords only bias future transcriptions. The existing
  re-diarize action already re-runs the *full* pipeline (it re-transcribes), so
  re-running a file transparently picks up current keywords.

## 1. Data model (`app/db.py`)

Add a nullable `keywords TEXT` column to the `persons` table.

- Extend `_migrate()` — which today only inspects the `jobs` table — to also add
  `keywords` to `persons` if missing, using the same `PRAGMA table_info` check.
- Add `keywords` to `_PERSON_COLS` so create/update round-trip it.

## 2. Union builder (`app/speakers.py`)

New pure, DB-free function:

```python
build_hotwords(persons: list[dict]) -> str | None
```

For each person dict: include `name` **unless** it matches the placeholder
pattern `^Person \d+$`; then split `keywords` on commas and newlines, stripping
blanks. Collect all terms, dedupe **case-insensitively** preserving first-seen
order, and join with `", "`. Return `None` when nothing qualifies (so we pass no
bias rather than an empty string). Pure function over a list of dicts — no `db`
import, unit-testable in isolation.

## 3. Engine plumbing (`app/transcriber.py`, `app/queue.py`)

- `Transcriber.transcribe` gains `hotwords: str | None = None`, threaded through
  `_run` into **both** `pipeline.transcribe(...)` and `model.transcribe(...)` as
  `hotwords=hotwords`. The OOM fallback ladder is otherwise untouched.
- In `_process` (`app/queue.py`), just before the
  `asyncio.to_thread(self.transcriber.transcribe, ...)` call, compute
  `hotwords = build_hotwords(db.list_persons(db_path))` and pass it through.

## 4. Editing API (`app/main.py`)

Extend `PUT /api/persons/{id}` to carry keywords as well as name.

- Replace `RenameBody` with an update body exposing `name: str | None = None`
  and `keywords: str | None = None` (both optional).
- Update whichever fields are present. Run `_apply_person_change_to_jobs` (name
  propagation into stored transcripts) **only when `name` actually changes** —
  keywords need no propagation since they only affect future transcriptions.
- `_person_public()` gains `keywords` so the UI can render and edit it.
- Fix existing integrations: callers/tests referencing `RenameBody` or expecting
  the old `_person_public` shape, and the JS rename call.

## 5. UI (`app/static/app.js`, `style.css`)

In `buildPersonRow`, add a small keyword text input under the person's name
(placeholder: *"keywords this person mentions…"*), saved on blur via the extended
PUT. Reuse existing person-row styling; minimal additions.

## 6. Testing

- `build_hotwords`: name included; `Person N` placeholder excluded; keyword
  comma/newline splitting; case-insensitive dedupe; all-empty → `None`.
- `transcriber`: a fake model asserts `hotwords=` reaches both the batched and
  non-batched `transcribe` calls.
- `queue`: `_process` builds the union from the gallery and passes it to
  `transcribe` (fake transcriber captures the arg).
- `db`/API: migration adds the column; `PUT` updates keywords (and still renames);
  `_person_public` includes `keywords`.
