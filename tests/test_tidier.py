import app.tidier as tidier_mod
from app.models import Segment
from app.tidier import SYSTEM_PROMPT, Turn, group_turns, tidy_turns


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


def test_tidy_turns_builds_prompt_and_parses(monkeypatch):
    captured = []

    def fake_chat(base_url, messages, *, model, temperature, timeout):
        captured.append((base_url, messages, temperature, timeout))
        return "Cleaned: " + messages[-1]["content"]

    monkeypatch.setattr(tidier_mod, "chat_completion", fake_chat)
    turns = [Turn("Ann", "um hello"), Turn(None, "like yeah")]
    out = tidy_turns(turns, "http://x:8080", timeout=42)

    assert out == [
        {"speaker": "Ann", "text": "Cleaned: um hello"},
        {"speaker": None, "text": "Cleaned: like yeah"},
    ]
    assert captured[0][1][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert captured[0][1][1]["content"] == "um hello"
    assert captured[0][3] == 42  # request timeout threaded through


def test_tidy_turns_falls_back_to_raw_on_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("server down")

    monkeypatch.setattr(tidier_mod, "chat_completion", boom)
    out = tidy_turns([Turn(None, "raw text")], "http://x:8080", timeout=5)
    assert out == [{"speaker": None, "text": "raw text"}]


def test_tidy_turns_reports_progress_per_turn(monkeypatch):
    monkeypatch.setattr(tidier_mod, "chat_completion", lambda *a, **k: "ok")
    calls = []
    turns = [Turn("Ann", "a"), Turn("Bob", "b"), Turn("Ann", "c")]
    tidy_turns(turns, "http://x:8080", on_progress=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_tidy_turns_survives_progress_callback_error(monkeypatch):
    monkeypatch.setattr(tidier_mod, "chat_completion", lambda *a, **k: "ok")

    def bad_progress(done, total):
        raise RuntimeError("loop closed")

    out = tidy_turns([Turn(None, "a"), Turn(None, "b")], "http://x:8080", on_progress=bad_progress)
    assert out == [{"speaker": None, "text": "ok"}, {"speaker": None, "text": "ok"}]
