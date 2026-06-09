from app.models import Segment
from app.tidier import Turn, group_turns


def _seg(text, speaker=None):
    return Segment(start=0.0, end=1.0, text=text, speaker=speaker)


def test_single_speaker_is_one_turn():
    segs = [_seg("hello"), _seg("there"), _seg("friend")]
    turns = group_turns(segs)
    assert turns == [Turn(speaker=None, text="hello there friend")]


def test_speaker_change_splits():
    segs = [_seg("hi", "Ann"), _seg("yo", "Ann"), _seg("hello", "Bob")]
    turns = group_turns(segs)
    assert turns == [Turn("Ann", "hi yo"), Turn("Bob", "hello")]


def test_blank_segments_skipped():
    segs = [_seg("  "), _seg("real"), _seg("")]
    assert group_turns(segs) == [Turn(None, "real")]


def test_long_turn_splits_at_segment_boundary():
    segs = [_seg("a" * 30, "Ann"), _seg("b" * 30, "Ann"), _seg("c" * 30, "Ann")]
    turns = group_turns(segs, max_chars=50)
    assert len(turns) == 3
    assert all(t.speaker == "Ann" for t in turns)
    assert [t.text for t in turns] == ["a" * 30, "b" * 30, "c" * 30]
