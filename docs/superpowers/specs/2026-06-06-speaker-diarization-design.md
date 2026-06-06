# fluister — Speaker Diarization + Global Voice Recognition (Design)

Date: 2026-06-06

## Goal
Recognize *who* is speaking in every transcribed file and tag segments with a
person. **Persons are global**: a voice auto-created from one file is the same
person entity across all files; future files with that voice match the existing
person and improve its voiceprint. Users can rename/merge persons; names apply
everywhere.

Decisions (from brainstorming): pyannote + HF token; **assign-to-closest-match**
above a similarity threshold else auto-create; inline speaker labels **plus** a
Speakers management page; diarization runs automatically on **every** file
("autosample").

## Approach
Keep the working faster-whisper engine; add pyannote as a **post-pass**.
Transcribe with **word-level timestamps**, then run pyannote
`speaker-diarization-3.1` (returns diarization turns **and** one embedding per
speaker via `return_embeddings=True`). Assign each word to a speaker by time
overlap, regroup into speaker turns, identify each speaker against the global
gallery. (Rejected: WhisperX — replaces the engine, brittle pins;
segment-level-only — too coarse.)

## Pipeline (worker, per file)
```
convert → transcribe (segments + words) → diarize (turns + per-speaker embeddings)
  → assign words→speaker turns → identify each speaker vs global gallery
  → rewrite labels to person names → write outputs + persist
```
Diarization is a post-pass, so live SSE shows plain text; the card re-renders
with speaker labels once done. If diarization is unavailable/fails, the job
still completes as a plain transcript.

## Shared model changes (`app/models.py`)
```python
@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None     # display name (person name or local label)
    person_id: str | None = None   # global person id, if identified

@dataclass
class Word:
    start: float
    end: float
    word: str
```

## Modules & contracts

### `app/diarizer.py`
```python
@dataclass
class DiarTurn:
    start: float; end: float; speaker: str  # local label, e.g. "SPEAKER_00"

class Diarizer:
    def __init__(self, model_name="pyannote/speaker-diarization-3.1",
                 device="auto", hf_token: str | None = None): ...
        # device auto -> cuda if torch.cuda.is_available() else cpu.
        # Load Pipeline.from_pretrained(model_name, use_auth_token=hf_token) once.
    def diarize(self, wav_path) -> tuple[list[DiarTurn], dict[str, list[float]]]:
        # pipeline(wav, return_embeddings=True) -> (annotation, embeddings ndarray)
        # returns (turns, {local_label: embedding as list[float]})
```
On CUDA OOM, retry on CPU. Raise a clear error if the model can't load (caller
degrades to no-diarization).

### `app/assign.py` (pure logic)
```python
def words_to_speaker_segments(words: list[Word], turns: list[DiarTurn]) -> list[Segment]:
    # each word -> turn with max time overlap (fallback: nearest turn);
    # group contiguous same-speaker words into Segments; Segment.speaker = local label.
def segments_to_speaker_segments(segments: list[Segment], turns: list[DiarTurn]) -> list[Segment]:
    # fallback when no words: assign each segment to the dominant overlapping turn.
```

### `app/speakers.py` (global gallery; numpy + db, temp-db testable)
```python
def cosine(a, b) -> float

class Gallery:
    def __init__(self, db_path, threshold: float = 0.45): ...
    def identify(self, embedding) -> tuple[str | None, float]:   # (person_id|None, best_sim)
    def assign_or_create(self, embedding, job_id=None,
                         exclude_ids: set[str] = frozenset()) -> tuple[str, bool]:
        # best person centroid (excluding exclude_ids); if best_sim >= threshold ->
        # assign + add_sample (recompute centroid); else create "Person N" + add_sample.
        # exclude_ids prevents two speakers in the SAME file collapsing to one person.
    def add_sample(self, person_id, embedding, job_id=None): ...
    def rename(self, person_id, name): ...
    def merge(self, src_id, dst_id): ...   # move samples, recompute centroid, delete src
    def delete(self, person_id): ...
    def list(self) -> list[dict]: ...
```
Embeddings stored as float32 bytes; centroid = mean of samples.

### `app/formats.py` (extend)
When `Segment.speaker` is set, prefix output lines with the speaker:
- TXT: `Name: text` per turn.
- SRT/VTT: `Name: text` inside the cue.
- JSON: each segment includes `speaker` and `person_id`; top-level `speakers`
  map (`local_label -> {person_id, name}`) when diarized.

### `app/transcriber.py` (extend)
Enable `word_timestamps=True`; return words alongside segments:
`transcribe(...) -> tuple[list[Segment], list[Word], TranscribeInfo]`. If batched
word timestamps misbehave, fall back to non-batched for the words.

## Data model (`app/db.py`)
New tables:
- `persons(id TEXT PK, name TEXT, created_at TEXT, centroid BLOB, n_samples INT, dim INT)`
- `person_embeddings(id TEXT PK, person_id TEXT, job_id TEXT, embedding BLOB, created_at TEXT)`
- `jobs`: add `diarized INTEGER DEFAULT 0`, `speakers TEXT` (JSON label→person map).

New db functions: `create_person, get_person, list_persons, update_person,
delete_person, add_person_embedding, list_person_embeddings,
delete_person_embeddings`.

## API (`app/main.py`)
- `GET /api/persons` — list (id, name, n_samples, created_at)
- `PUT /api/persons/{id}` — `{name}` rename
- `DELETE /api/persons/{id}`
- `POST /api/persons/merge` — `{src, dst}`
- `POST /api/jobs/{id}/rediarize` — re-run diarization on the stored wav and update
- existing `GET /api/jobs/{id}/download/json` carries per-segment speakers

## UI (`app/static/*`)
- **Top nav**: Transcriptions | Speakers.
- **Transcript (done)**: fetch the job JSON, render **speaker turns** with a
  color-coded `Name:` chip (stable color per `person_id`).
- **Speakers page**: list persons with editable name, sample count; **merge**
  two persons; delete. Changes propagate (re-fetch jobs/json).

## Config (`app/config.py`)
- Load `.env` from project root at startup into `os.environ` (don't override real
  env). Lets `HF_TOKEN` + `TRANSCRIBE_*` be set in `.env`. Ship `.env.example`;
  gitignore `.env`.
- New settings: `hf_token` (`HF_TOKEN`), `diarize` (`TRANSCRIBE_DIARIZE`, default
  true), `diarize_model`, `speaker_threshold` (`TRANSCRIBE_SPEAKER_THRESHOLD`,
  default 0.45).

## Dependencies
Add `pyannote.audio`, `torch` (CUDA build), `numpy`. Both ctranslate2 (large-v3
fp16 ~3 GB) and pyannote (~few hundred MB) fit in 10 GB VRAM.

## Errors / graceful degradation
- No `HF_TOKEN` / terms not accepted / pyannote load fails → diarization OFF,
  transcription still works; UI notes "speakers unavailable".
- Per-file diarization error → plain transcript + note, job still `done`.
- pyannote CUDA OOM → retry on CPU.
- Same-file two-speaker collision → `exclude_ids` forces distinct persons.

## Testing
- `speakers` — cosine, threshold assign/create, `exclude_ids`, merge/centroid,
  rename, delete (temp db + synthetic embeddings).
- `assign` — word↔turn overlap grouping; segment fallback.
- `formats` — speaker prefixes in txt/srt/json.
- API — persons CRUD + merge on temp db with fake embeddings; rediarize mocked.
- `diarizer` — optional slow test (needs HF token + GPU).

## Verification
End-to-end on the three real Signal files once `HF_TOKEN` is set. Expectation:
the two English notes likely cluster into **one** global person, the Dutch note
into another — confirming global recognition.
