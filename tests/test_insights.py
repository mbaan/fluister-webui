"""Insight pass: prompt building, defensive JSON parsing, timestamp coercion,
and graceful failure — all without a real llama-server."""
from types import SimpleNamespace

from app import insights


def _segs():
    return [
        SimpleNamespace(start=0.0, text="Welcome everyone to the planning call."),
        SimpleNamespace(start=65.0, text="Let's review the quarterly budget."),
        SimpleNamespace(start=185.0, text="Finally, action items and owners."),
    ]


def test_parse_time_forms():
    assert insights._parse_time("01:05") == 65.0
    assert insights._parse_time("1:02:03") == 3723.0
    assert insights._parse_time(42) == 42.0
    assert insights._parse_time("90") == 90.0
    assert insights._parse_time("nope") is None


def test_build_block_has_timestamps_and_skips_blanks():
    segs = [SimpleNamespace(start=0, text=""), SimpleNamespace(start=5, text="Hi there")]
    block = insights.build_transcript_block(segs)
    assert "[00:05] Hi there" in block
    assert block.count("\n") == 0  # only one non-blank line


def test_extract_json_plain_and_fenced():
    assert insights._extract_json('{"a": 1}') == {"a": 1}
    assert insights._extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert insights._extract_json("here you go: {\"a\": 3} cheers") == {"a": 3}
    assert insights._extract_json("not json at all") is None


def test_generate_insights_happy_path(monkeypatch):
    reply = (
        '{"summary": "A short planning call.",'
        ' "key_points": ["Review budget", "Assign owners", "  "],'
        ' "chapters": [{"time": "00:00", "title": "Intro"},'
        '              {"time": "01:05", "title": "Budget"},'
        '              {"title": "missing time"}]}'
    )
    monkeypatch.setattr(insights, "chat_completion", lambda *a, **k: reply)
    out = insights.generate_insights(_segs(), "http://x")
    assert out["summary"] == "A short planning call."
    assert out["key_points"] == ["Review budget", "Assign owners"]  # blank dropped
    assert out["chapters"] == [
        {"title": "Intro", "start": 0.0},
        {"title": "Budget", "start": 65.0},
    ]  # the title-without-time chapter is dropped


def test_generate_insights_bad_json_returns_none(monkeypatch):
    monkeypatch.setattr(insights, "chat_completion", lambda *a, **k: "sorry, no JSON here")
    assert insights.generate_insights(_segs(), "http://x") is None


def test_generate_insights_empty_segments_skips_llm(monkeypatch):
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("should not be called")

    monkeypatch.setattr(insights, "chat_completion", _boom)
    assert insights.generate_insights([], "http://x") is None
    assert called["n"] == 0


def test_generate_insights_llm_error_returns_none(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(insights, "chat_completion", _raise)
    assert insights.generate_insights(_segs(), "http://x") is None
