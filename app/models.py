"""Shared dataclasses used across modules.

Keep this module dependency-free (stdlib only) so every other module can import
it without pulling in heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    """One transcribed segment, times in seconds from start of audio.

    ``speaker`` / ``person_id`` are populated only when diarization ran:
    ``speaker`` is the display name (person name, or a local label like
    "SPEAKER_00"); ``person_id`` is the global person id when identified.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None
    person_id: str | None = None


@dataclass
class Word:
    """One transcribed word with timing, used to align speakers to text."""

    start: float
    end: float
    word: str


@dataclass
class DiarTurn:
    """A diarization turn: one speaker active over a time span.

    ``speaker`` is a file-local label (e.g. "SPEAKER_00"); it gets mapped to a
    global person during identification.
    """

    start: float
    end: float
    speaker: str


@dataclass
class TranscribeInfo:
    """Metadata returned by the transcriber about a completed run."""

    language: str  # detected or forced language code, e.g. "nl" / "en"
    duration: float  # audio duration in seconds
