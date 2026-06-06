"""Tests for app.filename_time."""

import os
from datetime import datetime
from pathlib import Path

import pytest

from app.filename_time import ParsedTime, parse_filename_timestamp, resolve_timestamp


# ── table-driven tests for parse_filename_timestamp ──────────────────────────

CASES = [
    # (filename, expected_dt, expected_has_time, description)

    # Signal with run-together time
    (
        "signal-2026-06-06-094449.aac",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "signal run-together HH MM SS",
    ),
    # Signal with dash-separated time + ms
    (
        "signal-2026-06-04-16-21-28-808.m4a",
        datetime(2026, 6, 4, 16, 21, 28),
        True,
        "signal dash-separated HH-MM-SS-ms",
    ),
    # WhatsApp Audio
    (
        "WhatsApp Audio 2026-06-06 at 09.44.49.opus",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "WhatsApp Audio at HH.MM.SS",
    ),
    # WhatsApp Image
    (
        "WhatsApp Image 2026-06-06 at 09.44.49.jpg",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "WhatsApp Image at HH.MM.SS",
    ),
    # WhatsApp Video
    (
        "WhatsApp Video 2026-06-06 at 09.44.49.mp4",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "WhatsApp Video at HH.MM.SS",
    ),
    # PTT (WhatsApp push-to-talk, date only)
    (
        "PTT-20260606-WA0001.opus",
        datetime(2026, 6, 6, 0, 0, 0),
        False,
        "PTT-YYYYMMDD-WA#### date only",
    ),
    # IMG compact
    (
        "IMG-20260606-WA0042.jpg",
        datetime(2026, 6, 6, 0, 0, 0),
        False,
        "IMG-YYYYMMDD-WA#### date only",
    ),
    # VID compact
    (
        "VID-20260606-WA0007.mp4",
        datetime(2026, 6, 6, 0, 0, 0),
        False,
        "VID-YYYYMMDD-WA#### date only",
    ),
    # Telegram
    (
        "audio_2026-06-06_09-44-49.ogg",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "Telegram audio_YYYY-MM-DD_HH-MM-SS",
    ),
    # Generic dashed date + colon-separated time
    (
        "recording_2026-06-06_09:44:49.wav",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "generic YYYY-MM-DD HH:MM:SS",
    ),
    # Generic dashed date + dot-separated time
    (
        "note-2026-06-06-09.44.49.m4a",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "generic YYYY-MM-DD-HH.MM.SS",
    ),
    # Generic dashed date + run-together time
    (
        "voice-2026-06-06-094449.mp3",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "generic YYYY-MM-DD-HHMMSS run-together",
    ),
    # Compact datetime YYYYMMDD_HHMMSS
    (
        "20260606_094449.wav",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "compact YYYYMMDD_HHMMSS",
    ),
    # Compact datetime YYYYMMDD-HHMMSS
    (
        "20260606-094449.wav",
        datetime(2026, 6, 6, 9, 44, 49),
        True,
        "compact YYYYMMDD-HHMMSS",
    ),
    # Generic dashed date only
    (
        "2026-06-06.ogg",
        datetime(2026, 6, 6, 0, 0, 0),
        False,
        "dashed date only YYYY-MM-DD",
    ),
    # Compact date only
    (
        "20260606.ogg",
        datetime(2026, 6, 6, 0, 0, 0),
        False,
        "compact date only YYYYMMDD",
    ),
]


@pytest.mark.parametrize("filename,expected_dt,expected_has_time,desc", CASES, ids=[c[3] for c in CASES])
def test_parse_filename_timestamp(filename, expected_dt, expected_has_time, desc):
    result = parse_filename_timestamp(filename)
    assert result is not None, f"Expected a match for {filename!r} ({desc})"
    assert result.source == "filename"
    assert result.dt == expected_dt, f"{desc}: got {result.dt}, expected {expected_dt}"
    assert result.has_time == expected_has_time, f"{desc}: has_time mismatch"


# ── no-match cases ────────────────────────────────────────────────────────────

NO_MATCH_CASES = [
    "voice memo.m4a",
    "recording.mp3",
    "untitled.ogg",
    "note.wav",
    "audio.aac",
]


@pytest.mark.parametrize("filename", NO_MATCH_CASES)
def test_no_match(filename):
    result = parse_filename_timestamp(filename)
    assert result is None, f"Expected None for {filename!r}, got {result}"


# ── resolve_timestamp mtime fallback ─────────────────────────────────────────

def test_resolve_timestamp_mtime_fallback(tmp_path):
    """A file with no timestamp in its name falls back to mtime."""
    f = tmp_path / "voice memo.m4a"
    f.write_bytes(b"")
    # Set a specific mtime: 2025-03-15 12:30:00 local
    target = datetime(2025, 3, 15, 12, 30, 0)
    mtime = target.timestamp()
    os.utime(f, (mtime, mtime))

    result = resolve_timestamp(f)
    assert result.source == "mtime"
    assert result.has_time is True
    assert result.dt == target


def test_resolve_timestamp_uses_filename_when_available(tmp_path):
    """A file with a timestamp in its name uses filename, not mtime."""
    f = tmp_path / "signal-2026-06-06-094449.aac"
    f.write_bytes(b"")
    # Set a different mtime so we can distinguish
    os.utime(f, (0, 0))

    result = resolve_timestamp(f)
    assert result.source == "filename"
    assert result.dt == datetime(2026, 6, 6, 9, 44, 49)
    assert result.has_time is True


# ── validation: impossible dates should return None ──────────────────────────

def test_impossible_date_rejected():
    """Feb 30 is not a valid date and should not match."""
    result = parse_filename_timestamp("audio_2026-02-30_09-44-49.ogg")
    assert result is None


def test_invalid_month_rejected():
    result = parse_filename_timestamp("2026-13-01.ogg")
    assert result is None


def test_invalid_hour_falls_back_to_date():
    # The time part 25:00:00 is invalid, but the date 2026-06-06 is valid.
    # Expect a date-only match (has_time=False) rather than None.
    result = parse_filename_timestamp("audio_2026-06-06_25-00-00.ogg")
    assert result is not None
    assert result.has_time is False
    assert result.dt == datetime(2026, 6, 6, 0, 0, 0)
