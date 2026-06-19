# fluister UI rework — design

Date: 2026-06-19

## Goal

Replace the dark-only, muted UI with a colorful, opinionated, themeable interface
(auto light/dark + manual override, four color themes), do a deeper UX rework, and
add four capabilities. Frontend stays vanilla JS — **no build step, fully offline**.
Backend gains a search index and an insight pass with **no new runtime deps**.

## Signature look

Procedurally-generated **halftone dot-matrix soundwave** hero banner on a `<canvas>`,
recolored per theme and subtly reactive while transcribing, with a small frosted
brand+nav panel floating on it. Default color theme: **Spectrum**.

## Theme system

- Tokens as CSS custom properties; two base sets (light + dark).
- **Mode**: default follows `prefers-color-scheme`; `<html data-theme="light|dark">`
  overrides; persisted as `fluister.theme = auto|light|dark`.
- **Color theme**: `<html data-accent="spectrum|aurora|sunset|meadow">` sets
  `--accent`, `--accent-2`, and the wave hue stops; persisted as `fluister.accent`
  (default `spectrum`).
- Status colors and the per-speaker hue system get light-mode-tuned lightness so
  chips read on white.
- Controls: compact cluster top-right of the hero — mode toggle (auto/light/dark),
  palette popover (4 swatches), search icon.

## Frontend file structure

Split the 1.8k-line `app.js` into ES modules under `app/static/js/`, loaded via
`<script type="module">` (no bundler needed, same-origin):

- `dom.js` — `el()` + shared helpers (escapeHtml, time/format, hueFor, parseTs).
- `theme.js` — mode/accent resolution, persistence, palette UI, accent stops.
- `wave.js` — halftone canvas generator + reactive animation.
- `jobs.js` — job list, cards, poll, SSE (core, ported from current app.js).
- `player.js` — audio element, player bar, power tools, word highlight.
- `reading.js` — focused reading route, in-transcript search, insight/chapters panel.
- `search.js` — global search palette (calls `/api/search`).
- `speakers.js` — speakers page + chip keyword editor.
- `app.js` — boot, hash router, orchestration.

`style.css` stays a single organized file.

## Routing

Hash-based: `#/` (list), `#/speakers`, `#/read/<id>`. Browser back works. Global-search
results deep-link to `#/read/<id>?q=<term>`.

## Features

### Focused reading view (`#/read/<id>`)
Full-viewport surface over a dimmed app. Top bar: back · filename/meta · Raw/Readable
toggle · in-transcript search. Body: Overview panel (insights, when present), then
prose in Literata at a ~70ch measure, larger type. Sticky player at the bottom.
Click a word → seek + play.

### In-transcript search
Client-side over the rendered transcript: highlight matches, ↑/↓ step, `n/total`
count, Enter cycles. Raw match → seek+play; Readable match → scroll into view.

### Global search (FTS5)
Subtle hero search icon (also `/`) opens a centered command-palette overlay.
Debounced query → `GET /api/search?q=` → ranked hits `{job_id, filename, snippet}`
with the term highlighted. Enter/click opens the reading view at that job, scrolled
to the first match. Esc closes.

Backend: FTS5 virtual table `jobs_fts(job_id UNINDEXED, filename, body)`, synced on
job create / completion / delete via explicit upserts in the db layer. Endpoint
returns top-N with `snippet()`. If the SQLite build lacks FTS5 (detected at init),
fall back to a `LIKE` scan.

### Chip keyword editor (speakers)
Parse the existing comma string into sorted, de-duped, removable chips; add via input
(Enter or comma), remove via ×. Serialize back to a comma string on change →
`PUT /api/persons/{id}`. No API change.

### Playback power tools
On the reading-view player: speed (0.75/1/1.25/1.5/2), ±10s, A–B loop, and
skip-silence (jump when the gap between consecutive segments exceeds ~0.8s). All
derived client-side from segment timestamps.

### Reactive hero wave
While any job is active (converting/transcribing/tidying), animate the wave's
amplitude envelope with a slow breathing factor via throttled rAF; idle renders once,
static. Disabled under `prefers-reduced-motion`.

### On-device insight pass (backend)
New `app/insights.py`: given the transcript turns, call llama-server (reuse the
existing `llm_server` lifecycle/config) to produce JSON
`{summary: str, key_points: [str], chapters: [{title, start_seconds}]}`. Best-effort —
any failure stores nothing and the job still completes. Pipeline: runs **after** the
tidy pass. New DB column `insights_json TEXT` (idempotent migration). New SSE
`insights` event carries the JSON. Reading view renders the Overview panel + a
clickable chapter list (chapter start mapped to the nearest segment for scroll).
Labeled "AI summary · generated on-device." **Never alters `transcript_text`,
`segments_json`, or `tidied_json`** — honoring the "tidier, not fixer" guardrail.

Prompt: concise summary (≤3 sentences), 3–7 key points/action items, 3–8 chapters with
short titles + the start timestamp of the turn they begin; language follows the
transcript. Parse defensively (strict JSON extraction, like the tidier).

## Phasing (incremental, each independently shippable)

1. **Foundation** — theme system, hero canvas + reactive wave, full restyle of list /
   cards / speakers / dropzone / empty+error states. *(frontend; delivers the core
   "colorful + light/dark" ask on its own)*
2. **Reading & control** — focused reading view, in-transcript search, chip keyword
   editor, playback power tools. *(frontend)*
3. **Search backend** — FTS5 table + `/api/search` + palette wiring. *(backend+frontend)*
4. **Insight pass** — `insights.py` + pipeline + storage + SSE + reading panel.
   *(backend+frontend)*

## Testing

- Backend (pytest, matching existing style): FTS5 index sync + query + LIKE fallback;
  insights JSON parsing / defensive handling; migration adds columns idempotently.
- Frontend: verified by running the app (no JS test harness exists). Confirm poll/SSE
  still work, themes persist across reload, reduced-motion is honored.

## Constraints / guardrails

- Fully offline; no CDNs; self-hosted assets only.
- No new Python runtime deps (FTS5 is stdlib sqlite3; insight reuses llama-server).
- Respect the model guardrails (whisper large-v3, no `initial_prompt`, LLM = tidier
  not fixer): the insight pass is additive + labeled and never edits the transcript.
- GPU co-residence: insight runs inside the llama-server window (tidy phase) — no
  extra server process.
- Preserve the job poll/SSE lifecycle.

## Out of scope

Waveform-as-player, export (SRT/VTT/MD), in-browser recording, editable transcript,
job-list filename filter, full keyboard-shortcut set.
