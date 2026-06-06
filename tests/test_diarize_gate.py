"""Minimum-speaker-duration gate: short/noise clusters must not become persons."""

import os
import tempfile

os.environ.setdefault("TRANSCRIBE_DATA_DIR", tempfile.mkdtemp(prefix="fluister-gate-"))

from app import db
from app.config import load_settings
from app.models import DiarTurn, Word
from app.queue import JobQueue


class _FakeDiarizer:
    """One dominant speaker (10 s) and one brief noise cluster (0.5 s)."""

    def diarize(self, wav):
        turns = [
            DiarTurn(0.0, 10.0, "SPEAKER_00"),
            DiarTurn(10.0, 10.5, "SPEAKER_01"),
        ]
        emb = {"SPEAKER_00": [1.0, 0.0, 0.0], "SPEAKER_01": [0.0, 1.0, 0.0]}
        return turns, emb


def test_short_speaker_is_not_enrolled(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRANSCRIBE_MIN_SPEAKER_SECONDS", "2")
    settings = load_settings()
    db.init_db(settings.db_path)

    q = JobQueue(settings)
    q.diarizer = _FakeDiarizer()

    words = [Word(0.0, 9.0, "hello"), Word(10.0, 10.4, "blip")]
    segs, speakers_map, diarized = q._diarize_and_identify("job1", "x.wav", [], words)

    persons = db.list_persons(settings.db_path)
    assert len(persons) == 1          # only the dominant speaker enrolled
    assert diarized is True
    assert len(speakers_map) == 1

    dom = next(s for s in segs if "hello" in s.text)
    noise = next(s for s in segs if "blip" in s.text)
    assert dom.person_id == persons[0]["id"]
    assert noise.speaker is None and noise.person_id is None
