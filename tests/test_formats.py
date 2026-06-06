"""Tests for app/formats.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.formats import to_json, to_srt, to_txt, to_vtt, _format_srt_time, _format_vtt_time
from app.models import Segment, TranscriptMeta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def segments() -> list[Segment]:
    return [
        Segment(start=0.0, end=3.5, text="Goedemorgen, dit is een test."),
        Segment(start=3.5, end=7.8, text="  café au lait smaakt lëkker.  "),
        Segment(start=7.8, end=3661.5, text="Tot ziens!"),
    ]


@pytest.fixture
def meta() -> TranscriptMeta:
    return TranscriptMeta(
        filename="opname.m4a",
        language="nl",
        duration=3661.5,
        model="whisper-large-v3",
        msg_timestamp="2026-06-06T10:00:00+02:00",
        msg_timestamp_source="filename",
    )


@pytest.fixture
def meta_no_ts() -> TranscriptMeta:
    return TranscriptMeta(
        filename="opname.m4a",
        language="nl",
        duration=120.0,
        model="whisper-large-v3",
    )


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------

class TestTimestampFormatting:
    def test_srt_over_one_hour(self):
        assert _format_srt_time(3661.5) == "01:01:01,500"

    def test_srt_zero(self):
        assert _format_srt_time(0.0) == "00:00:00,000"

    def test_srt_sub_minute(self):
        assert _format_srt_time(3.5) == "00:00:03,500"

    def test_vtt_over_one_hour(self):
        assert _format_vtt_time(3661.5) == "01:01:01.500"

    def test_vtt_zero(self):
        assert _format_vtt_time(0.0) == "00:00:00.000"

    def test_vtt_uses_dot_separator(self):
        ts = _format_vtt_time(61.25)
        assert "." in ts
        assert "," not in ts

    def test_srt_uses_comma_separator(self):
        ts = _format_srt_time(61.25)
        assert "," in ts
        assert ts.count(",") == 1


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------

class TestSRT:
    def test_first_cue_index(self, segments):
        output = to_srt(segments)
        first_line = output.strip().split("\n")[0]
        assert first_line == "1"

    def test_contains_arrow(self, segments):
        output = to_srt(segments)
        assert " --> " in output

    def test_segment_text_present(self, segments):
        output = to_srt(segments)
        assert "Goedemorgen, dit is een test." in output
        assert "café au lait smaakt lëkker." in output

    def test_cue_count(self, segments):
        output = to_srt(segments)
        # Split on double newlines, filter empty blocks
        blocks = [b for b in output.split("\n\n") if b.strip()]
        assert len(blocks) == len(segments)

    def test_second_cue_index(self, segments):
        output = to_srt(segments)
        blocks = [b for b in output.split("\n\n") if b.strip()]
        assert blocks[1].startswith("2\n")

    def test_timestamp_in_cue(self, segments):
        output = to_srt(segments)
        assert "00:00:00,000 --> 00:00:03,500" in output


# ---------------------------------------------------------------------------
# VTT
# ---------------------------------------------------------------------------

class TestVTT:
    def test_starts_with_webvtt(self, segments):
        output = to_vtt(segments)
        assert output.startswith("WEBVTT")

    def test_contains_arrow(self, segments):
        output = to_vtt(segments)
        assert " --> " in output

    def test_segment_text_present(self, segments):
        output = to_vtt(segments)
        assert "Goedemorgen, dit is een test." in output

    def test_vtt_dot_separator_in_timestamps(self, segments):
        output = to_vtt(segments)
        # All timestamps use '.' not ','
        lines = output.split("\n")
        ts_lines = [l for l in lines if " --> " in l]
        assert ts_lines, "no timestamp lines found"
        for line in ts_lines:
            assert "," not in line

    def test_non_ascii_text_present(self, segments):
        output = to_vtt(segments)
        assert "lëkker" in output


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class TestJSON:
    def test_valid_json(self, segments, meta):
        output = to_json(segments, meta)
        data = json.loads(output)
        assert "meta" in data
        assert "segments" in data

    def test_meta_fields(self, segments, meta):
        data = json.loads(to_json(segments, meta))
        m = data["meta"]
        assert m["filename"] == meta.filename
        assert m["language"] == meta.language
        assert m["duration"] == meta.duration
        assert m["model"] == meta.model
        assert m["msg_timestamp"] == meta.msg_timestamp
        assert m["msg_timestamp_source"] == meta.msg_timestamp_source

    def test_segment_fields(self, segments, meta):
        data = json.loads(to_json(segments, meta))
        segs = data["segments"]
        assert len(segs) == len(segments)
        assert segs[0]["start"] == 0.0
        assert segs[0]["end"] == 3.5
        assert segs[0]["text"] == "Goedemorgen, dit is een test."
        assert "speaker" in segs[0]
        assert "person_id" in segs[0]

    def test_segment_speaker_person_id_null_when_unset(self, segments, meta):
        data = json.loads(to_json(segments, meta))
        for seg in data["segments"]:
            assert seg["speaker"] is None
            assert seg["person_id"] is None

    def test_segment_speaker_person_id_values(self, meta):
        segs = [Segment(start=0.0, end=1.0, text="Hi", speaker="Anna", person_id="p1")]
        data = json.loads(to_json(segs, meta))
        assert data["segments"][0]["speaker"] == "Anna"
        assert data["segments"][0]["person_id"] == "p1"

    def test_speakers_top_level_present_when_passed(self, segments, meta):
        speakers_map = {"SPEAKER_00": {"person_id": "p1", "name": "Anna"}}
        data = json.loads(to_json(segments, meta, speakers=speakers_map))
        assert "speakers" in data
        assert data["speakers"] == speakers_map

    def test_speakers_top_level_absent_when_not_passed(self, segments, meta):
        data = json.loads(to_json(segments, meta))
        assert "speakers" not in data

    def test_non_ascii_survives(self, segments, meta):
        output = to_json(segments, meta)
        assert "ë" in output
        assert "café" in output
        data = json.loads(output)
        texts = [s["text"] for s in data["segments"]]
        assert any("ë" in t for t in texts)
        assert any("café" in t for t in texts)

    def test_null_timestamp_fields(self, segments, meta_no_ts):
        data = json.loads(to_json(segments, meta_no_ts))
        assert data["meta"]["msg_timestamp"] is None
        assert data["meta"]["msg_timestamp_source"] is None


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------

class TestTXT:
    def test_contains_transcript_text(self, segments, meta):
        output = to_txt(segments, meta)
        assert "Goedemorgen, dit is een test." in output
        assert "café au lait smaakt lëkker." in output
        assert "Tot ziens!" in output

    def test_contains_msg_timestamp_when_set(self, segments, meta):
        output = to_txt(segments, meta)
        assert "2026-06-06T10:00:00+02:00" in output
        assert "filename" in output

    def test_omits_sent_line_when_timestamp_none(self, segments, meta_no_ts):
        output = to_txt(segments, meta_no_ts)
        assert "Sent:" not in output

    def test_contains_filename(self, segments, meta):
        output = to_txt(segments, meta)
        assert "opname.m4a" in output

    def test_contains_language(self, segments, meta):
        output = to_txt(segments, meta)
        assert "nl" in output

    def test_contains_model(self, segments, meta):
        output = to_txt(segments, meta)
        assert "whisper-large-v3" in output

    def test_contains_duration(self, segments, meta):
        # duration 3661.5 seconds = 1:01:01
        output = to_txt(segments, meta)
        assert "1:01:01" in output

    def test_skips_empty_segments(self, meta):
        segs = [
            Segment(start=0.0, end=1.0, text="Hello"),
            Segment(start=1.0, end=2.0, text="   "),
            Segment(start=2.0, end=3.0, text="World"),
        ]
        output = to_txt(segs, meta)
        lines = output.split("\n")
        # Only non-empty text lines should appear after the separator
        sep_idx = next(i for i, l in enumerate(lines) if l.startswith("---"))
        body_lines = [l for l in lines[sep_idx + 1:] if l]
        assert body_lines == ["Hello", "World"]


# ---------------------------------------------------------------------------
# Speaker labels
# ---------------------------------------------------------------------------

class TestTXTSpeakerLabels:
    def test_speaker_prefix_on_first_segment(self, meta):
        segs = [Segment(start=0.0, end=1.0, text="Hello", speaker="Anna")]
        output = to_txt(segs, meta)
        assert "Anna: Hello" in output

    def test_speaker_prefix_does_not_repeat_consecutive(self, meta):
        segs = [
            Segment(start=0.0, end=1.0, text="Hello", speaker="Anna"),
            Segment(start=1.0, end=2.0, text="World", speaker="Anna"),
        ]
        output = to_txt(segs, meta)
        lines = output.split("\n")
        sep_idx = next(i for i, l in enumerate(lines) if l.startswith("---"))
        body_lines = [l for l in lines[sep_idx + 1:] if l]
        # First line has prefix, second does not
        assert body_lines[0] == "Anna: Hello"
        assert body_lines[1] == "World"

    def test_speaker_prefix_reappears_on_speaker_change(self, meta):
        segs = [
            Segment(start=0.0, end=1.0, text="Hi", speaker="Anna"),
            Segment(start=1.0, end=2.0, text="Hey", speaker="Bob"),
            Segment(start=2.0, end=3.0, text="Bye", speaker="Anna"),
        ]
        output = to_txt(segs, meta)
        lines = output.split("\n")
        sep_idx = next(i for i, l in enumerate(lines) if l.startswith("---"))
        body_lines = [l for l in lines[sep_idx + 1:] if l]
        assert body_lines == ["Anna: Hi", "Bob: Hey", "Anna: Bye"]

    def test_no_speaker_prefix_when_none(self, meta):
        segs = [Segment(start=0.0, end=1.0, text="Hello")]
        output = to_txt(segs, meta)
        assert "None:" not in output
        assert "Hello" in output
        lines = output.split("\n")
        sep_idx = next(i for i, l in enumerate(lines) if l.startswith("---"))
        body_lines = [l for l in lines[sep_idx + 1:] if l]
        assert body_lines == ["Hello"]


class TestSRTSpeakerLabels:
    def test_cue_text_has_speaker_prefix(self):
        segs = [Segment(start=0.0, end=1.0, text="Hello", speaker="Anna")]
        output = to_srt(segs)
        assert "Anna: Hello" in output

    def test_no_speaker_prefix_when_none(self):
        segs = [Segment(start=0.0, end=1.0, text="Hello")]
        output = to_srt(segs)
        assert "None:" not in output
        assert "Hello" in output

    def test_each_cue_gets_own_prefix(self):
        segs = [
            Segment(start=0.0, end=1.0, text="Hi", speaker="Anna"),
            Segment(start=1.0, end=2.0, text="Hey", speaker="Anna"),
        ]
        output = to_srt(segs)
        # Both cues should have the prefix (no de-duplication in SRT)
        assert output.count("Anna: ") == 2


class TestVTTSpeakerLabels:
    def test_cue_text_has_speaker_prefix(self):
        segs = [Segment(start=0.0, end=1.0, text="Hello", speaker="Bob")]
        output = to_vtt(segs)
        assert "Bob: Hello" in output

    def test_no_speaker_prefix_when_none(self):
        segs = [Segment(start=0.0, end=1.0, text="Hello")]
        output = to_vtt(segs)
        assert "None:" not in output
        assert "Hello" in output

    def test_each_cue_gets_own_prefix(self):
        segs = [
            Segment(start=0.0, end=1.0, text="Hi", speaker="Bob"),
            Segment(start=1.0, end=2.0, text="Hey", speaker="Bob"),
        ]
        output = to_vtt(segs)
        assert output.count("Bob: ") == 2
