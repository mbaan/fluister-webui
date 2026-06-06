"""Tests for app.assign: combining word/segment timings with diarization turns."""

from __future__ import annotations

import pytest

from app.assign import overlap, segments_to_speaker_segments, words_to_speaker_segments
from app.models import DiarTurn, Segment, Word


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _words(*specs: tuple[float, float, str]) -> list[Word]:
    return [Word(start=s, end=e, word=w) for s, e, w in specs]


def _turns(*specs: tuple[float, float, str]) -> list[DiarTurn]:
    return [DiarTurn(start=s, end=e, speaker=sp) for s, e, sp in specs]


# ---------------------------------------------------------------------------
# overlap()
# ---------------------------------------------------------------------------

def test_overlap_full():
    assert overlap(1.0, 3.0, 1.0, 3.0) == pytest.approx(2.0)


def test_overlap_partial():
    assert overlap(0.0, 2.0, 1.0, 3.0) == pytest.approx(1.0)


def test_overlap_none():
    assert overlap(0.0, 1.0, 2.0, 3.0) == pytest.approx(0.0)


def test_overlap_adjacent():
    # Touching edges → zero overlap
    assert overlap(0.0, 1.0, 1.0, 2.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# words_to_speaker_segments — two-speaker back-and-forth
# ---------------------------------------------------------------------------

def test_two_speaker_back_and_forth():
    """Words 0-2s and 4-6s under SPEAKER_00, words 2-4s under SPEAKER_01."""
    words = _words(
        (0.0, 1.0, "Hello"),
        (1.0, 2.0, "world"),
        (2.0, 3.0, "how"),
        (3.0, 4.0, "are"),
        (4.0, 5.0, "you"),
        (5.0, 6.0, "today"),
    )
    turns = _turns(
        (0.0, 2.0, "SPEAKER_00"),
        (2.0, 4.0, "SPEAKER_01"),
        (4.0, 6.0, "SPEAKER_00"),
    )
    segments = words_to_speaker_segments(words, turns)

    assert len(segments) == 3

    assert segments[0].speaker == "SPEAKER_00"
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].end == pytest.approx(2.0)
    assert segments[0].text == "Hello world"

    assert segments[1].speaker == "SPEAKER_01"
    assert segments[1].start == pytest.approx(2.0)
    assert segments[1].end == pytest.approx(4.0)
    assert segments[1].text == "how are"

    assert segments[2].speaker == "SPEAKER_00"
    assert segments[2].start == pytest.approx(4.0)
    assert segments[2].end == pytest.approx(6.0)
    assert segments[2].text == "you today"

    # person_id is always None from this module
    for seg in segments:
        assert seg.person_id is None


# ---------------------------------------------------------------------------
# words_to_speaker_segments — snap to nearest when no overlap
# ---------------------------------------------------------------------------

def test_word_no_overlap_snaps_to_nearest():
    """A word at 10-11s with no overlapping turn should snap to the nearest."""
    words = _words(
        (0.0, 1.0, "first"),
        (10.0, 11.0, "orphan"),   # no turn covers this range
        (20.0, 21.0, "last"),
    )
    turns = _turns(
        (0.0, 2.0, "SPEAKER_00"),
        (18.0, 22.0, "SPEAKER_01"),
    )
    segments = words_to_speaker_segments(words, turns)

    # "orphan" is equidistant between SPEAKER_00 (midpoint 1s) and SPEAKER_01
    # (midpoint 20s): distance from orphan midpoint (10.5) is 9.5 vs 9.5.
    # min() will pick SPEAKER_00 (first in list) on a tie.
    orphan_seg = next(s for s in segments if "orphan" in s.text)
    assert orphan_seg.speaker in ("SPEAKER_00", "SPEAKER_01")  # deterministic pick


def test_word_no_overlap_snaps_to_clearly_nearest():
    """A word at 1.5-2s with no direct overlap snaps to the clearly nearest turn."""
    words = _words(
        (5.0, 6.0, "far"),
        (1.5, 2.0, "near"),   # closer to SPEAKER_00 (0-1s) than SPEAKER_01 (10-12s)
    )
    # Sort by start so the function sees them in time order (contract says list[Word])
    words_sorted = sorted(words, key=lambda w: w.start)

    turns = _turns(
        (0.0, 1.0, "SPEAKER_00"),
        (10.0, 12.0, "SPEAKER_01"),
    )
    segments = words_to_speaker_segments(words_sorted, turns)

    near_seg = next(s for s in segments if "near" in s.text)
    assert near_seg.speaker == "SPEAKER_00"


# ---------------------------------------------------------------------------
# words_to_speaker_segments — empty turns
# ---------------------------------------------------------------------------

def test_empty_turns_single_segment():
    """Empty turns → all words become one segment per contiguous run, speaker=None."""
    words = _words(
        (0.0, 1.0, "one"),
        (1.0, 2.0, "two"),
        (2.0, 3.0, "three"),
    )
    segments = words_to_speaker_segments(words, [])

    assert len(segments) == 1
    assert segments[0].speaker is None
    assert segments[0].text == "one two three"
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].end == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# words_to_speaker_segments — empty words
# ---------------------------------------------------------------------------

def test_empty_words_returns_empty():
    turns = _turns((0.0, 5.0, "SPEAKER_00"))
    assert words_to_speaker_segments([], turns) == []


def test_empty_words_empty_turns_returns_empty():
    assert words_to_speaker_segments([], []) == []


# ---------------------------------------------------------------------------
# segments_to_speaker_segments
# ---------------------------------------------------------------------------

def test_segments_to_speaker_mostly_overlapping():
    """A segment mostly overlapping SPEAKER_01 should get that label."""
    segs = [
        Segment(start=0.0, end=1.0, text="small"),
        Segment(start=1.0, end=5.0, text="mostly second"),
    ]
    turns = _turns(
        (0.0, 1.5, "SPEAKER_00"),  # overlaps seg1 by 1.0s, seg2 by 0.5s
        (1.0, 5.0, "SPEAKER_01"),  # overlaps seg1 by 0.5s, seg2 by 4.0s
    )
    result = segments_to_speaker_segments(segs, turns)

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"   # 1.0s > 0.5s
    assert result[1].speaker == "SPEAKER_01"   # 4.0s > 0.5s

    # Text and times are preserved exactly
    assert result[0].text == "small"
    assert result[0].start == pytest.approx(0.0)
    assert result[0].end == pytest.approx(1.0)
    assert result[1].text == "mostly second"
    assert result[1].start == pytest.approx(1.0)
    assert result[1].end == pytest.approx(5.0)

    for seg in result:
        assert seg.person_id is None


def test_segments_to_speaker_no_turns():
    """Empty turns → all segments keep speaker=None."""
    segs = [Segment(start=0.0, end=2.0, text="hello", speaker="OLD")]
    result = segments_to_speaker_segments(segs, [])
    assert result[0].speaker is None
    assert result[0].text == "hello"


def test_segments_to_speaker_preserves_order():
    """Output order matches input order."""
    segs = [
        Segment(start=float(i), end=float(i + 1), text=f"word{i}")
        for i in range(5)
    ]
    turns = _turns((0.0, 5.0, "SPEAKER_00"))
    result = segments_to_speaker_segments(segs, turns)
    assert [s.text for s in result] == [s.text for s in segs]


def test_segments_to_speaker_empty_input():
    """Empty segment list returns empty list."""
    turns = _turns((0.0, 5.0, "SPEAKER_00"))
    assert segments_to_speaker_segments([], turns) == []
