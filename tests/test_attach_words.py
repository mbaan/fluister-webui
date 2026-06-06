"""Tests for app.assign.attach_words_to_segments: word→segment attachment."""

from __future__ import annotations

from app.assign import attach_words_to_segments
from app.models import Segment, Word

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _words(*specs: tuple[float, float, str]) -> list[Word]:
    return [Word(start=s, end=e, word=w) for s, e, w in specs]


# ---------------------------------------------------------------------------
# Distribution by midpoint
# ---------------------------------------------------------------------------


def test_words_distributed_across_segments_by_midpoint():
    """Each word attaches to the segment whose span contains its midpoint."""
    segments = [
        Segment(start=0.0, end=2.0, text="hello world"),
        Segment(start=2.0, end=4.0, text="how are"),
    ]
    words = _words(
        (0.0, 1.0, "hello"),   # midpoint 0.5 → seg 0
        (1.0, 2.0, "world"),   # midpoint 1.5 → seg 0
        (2.0, 3.0, "how"),     # midpoint 2.5 → seg 1
        (3.0, 4.0, "are"),     # midpoint 3.5 → seg 1
    )
    result = attach_words_to_segments(segments, words)

    assert len(result) == 2
    assert [w["word"] for w in result[0]["words"]] == ["hello", "world"]
    assert [w["word"] for w in result[1]["words"]] == ["how", "are"]
    # Word dicts carry start/end/word only.
    assert result[0]["words"][0] == {"start": 0.0, "end": 1.0, "word": "hello"}


def test_words_stay_in_input_order_within_segment():
    segments = [Segment(start=0.0, end=10.0, text="a b c")]
    words = _words(
        (0.0, 1.0, "a"),
        (1.0, 2.0, "b"),
        (2.0, 3.0, "c"),
    )
    result = attach_words_to_segments(segments, words)
    assert [w["word"] for w in result[0]["words"]] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Words whose midpoint falls in no segment are dropped
# ---------------------------------------------------------------------------


def test_word_outside_all_segments_is_dropped():
    segments = [
        Segment(start=0.0, end=2.0, text="in"),
        Segment(start=5.0, end=7.0, text="also in"),
    ]
    words = _words(
        (0.0, 1.0, "in"),       # midpoint 0.5 → seg 0
        (3.0, 4.0, "orphan"),   # midpoint 3.5 → gap, dropped
        (5.5, 6.5, "also"),     # midpoint 6.0 → seg 1
    )
    result = attach_words_to_segments(segments, words)

    assert [w["word"] for w in result[0]["words"]] == ["in"]
    assert [w["word"] for w in result[1]["words"]] == ["also"]
    # "orphan" appears nowhere.
    all_words = [w["word"] for seg in result for w in seg["words"]]
    assert "orphan" not in all_words


# ---------------------------------------------------------------------------
# Empty words
# ---------------------------------------------------------------------------


def test_empty_words_gives_empty_lists():
    segments = [
        Segment(start=0.0, end=2.0, text="one"),
        Segment(start=2.0, end=4.0, text="two"),
    ]
    result = attach_words_to_segments(segments, [])
    assert len(result) == 2
    assert all(seg["words"] == [] for seg in result)


def test_segment_with_no_matching_words_gets_empty_list():
    segments = [
        Segment(start=0.0, end=2.0, text="has"),
        Segment(start=10.0, end=12.0, text="empty"),
    ]
    words = _words((0.0, 1.0, "has"))  # midpoint 0.5 → seg 0 only
    result = attach_words_to_segments(segments, words)
    assert [w["word"] for w in result[0]["words"]] == ["has"]
    assert result[1]["words"] == []


# ---------------------------------------------------------------------------
# Preserves speaker / person_id and segment order
# ---------------------------------------------------------------------------


def test_preserves_speaker_person_id_and_order():
    segments = [
        Segment(start=0.0, end=2.0, text="alpha", speaker="Alice", person_id="p1"),
        Segment(start=2.0, end=4.0, text="beta", speaker="Bob", person_id="p2"),
        Segment(start=4.0, end=6.0, text="gamma", speaker=None, person_id=None),
    ]
    words = _words((0.5, 1.5, "alpha"))
    result = attach_words_to_segments(segments, words)

    assert [seg["text"] for seg in result] == ["alpha", "beta", "gamma"]
    assert result[0]["speaker"] == "Alice"
    assert result[0]["person_id"] == "p1"
    assert result[1]["speaker"] == "Bob"
    assert result[1]["person_id"] == "p2"
    assert result[2]["speaker"] is None
    assert result[2]["person_id"] is None
    # start/end mirror the segment.
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 2.0


# ---------------------------------------------------------------------------
# Boundary between two adjacent segments
# ---------------------------------------------------------------------------


def test_word_midpoint_on_segment_boundary():
    """A word whose midpoint sits exactly on the shared boundary lands in the
    first segment that contains it (inclusive [start, end])."""
    segments = [
        Segment(start=0.0, end=2.0, text="left"),
        Segment(start=2.0, end=4.0, text="right"),
    ]
    # midpoint = (1.5 + 2.5) / 2 = 2.0, exactly on the boundary shared by both.
    words = _words((1.5, 2.5, "edge"))
    result = attach_words_to_segments(segments, words)

    # 2.0 is in [0.0, 2.0] (first match wins) → left segment.
    assert [w["word"] for w in result[0]["words"]] == ["edge"]
    assert result[1]["words"] == []


def test_word_clearly_in_second_segment_after_boundary():
    """Sanity: a midpoint just past the boundary lands in the second segment."""
    segments = [
        Segment(start=0.0, end=2.0, text="left"),
        Segment(start=2.0, end=4.0, text="right"),
    ]
    words = _words((2.4, 2.6, "after"))  # midpoint 2.5 → only seg 1
    result = attach_words_to_segments(segments, words)
    assert result[0]["words"] == []
    assert [w["word"] for w in result[1]["words"]] == ["after"]
