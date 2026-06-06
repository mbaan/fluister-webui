"""Shared dataclasses used across modules.

Keep this module dependency-free (stdlib only) so every other module can import
it without pulling in heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    """One transcribed segment, times in seconds from start of audio."""

    start: float
    end: float
    text: str


@dataclass
class TranscribeInfo:
    """Metadata returned by the transcriber about a completed run."""

    language: str  # detected or forced language code, e.g. "nl" / "en"
    duration: float  # audio duration in seconds


@dataclass
class TranscriptMeta:
    """Header metadata embedded into txt/json outputs."""

    filename: str
    language: str
    duration: float
    model: str
    msg_timestamp: str | None = None  # ISO-8601, when the message was sent
    msg_timestamp_source: str | None = None  # "filename" | "mtime"
