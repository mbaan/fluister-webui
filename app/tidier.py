"""Readability tidier: turns a raw transcript into punctuated, paragraphed,
filler-free text via a local llama-server. Pure + HTTP-only — no torch/whisper
imports, so it stays cheap to import and easy to test.

Contract: the LLM *tidies, never fixes* — it must not change, add, translate,
reorder, or "correct" meaningful words; it only punctuates, paragraphs, and
drops fillers. See
docs/superpowers/specs/2026-06-09-llm-tidy-readable-transcript-design.md.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    """A contiguous block of one speaker's text to tidy in isolation."""

    speaker: str | None
    text: str


def group_turns(segments: Iterable, max_chars: int = 4000) -> list[Turn]:
    """Merge consecutive same-speaker segments into turns, splitting a turn at a
    segment boundary when it would exceed ``max_chars`` (keeps prompts well inside
    the model context). Blank segments are skipped. ``segments`` are objects with
    ``.text`` and ``.speaker`` (e.g. ``app.models.Segment``)."""
    turns: list[Turn] = []
    cur_speaker = None
    cur_parts: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur_parts, cur_len
        if cur_parts:
            turns.append(Turn(speaker=cur_speaker, text=" ".join(cur_parts)))
        cur_parts = []
        cur_len = 0

    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        speaker = getattr(seg, "speaker", None)
        new_speaker = speaker != cur_speaker
        too_long = bool(cur_parts) and (cur_len + 1 + len(text)) > max_chars
        if new_speaker or too_long:
            flush()
            cur_speaker = speaker
        cur_parts.append(text)
        cur_len += (1 if cur_len else 0) + len(text)
    flush()
    return turns


SYSTEM_PROMPT = (
    "You clean up speech-to-text transcripts for readability. Add punctuation, "
    "capitalization, and paragraph breaks. Remove filler words (uh, um, like, you "
    "know) and false starts / repeated restarts. Do NOT change, add, remove (other "
    "than fillers), reorder, translate, or 'correct' any meaningful word. Preserve "
    "the original language(s) exactly, including Dutch-English code-switching. "
    "Output only the cleaned text, nothing else."
)


def chat_completion(
    base_url: str,
    messages: list[dict],
    *,
    model: str = "local",
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    """POST an OpenAI-style chat completion to llama-server; return the content.
    Raises on transport/HTTP/parse errors (callers handle best-effort)."""
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def tidy_turns(
    turns: list[Turn],
    base_url: str,
    *,
    model: str = "local",
    temperature: float = 0.1,
    timeout: int = 120,
) -> list[dict]:
    """Tidy each turn independently. On a per-turn failure, fall back to that
    turn's raw text (degrade gracefully — never drop content). Returns a list of
    ``{"speaker": str | None, "text": str}`` paragraphs."""
    out: list[dict] = []
    for turn in turns:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": turn.text},
        ]
        try:
            text = chat_completion(
                base_url, messages, model=model, temperature=temperature, timeout=timeout
            )
        except Exception:  # noqa: BLE001
            logger.warning("Tidy failed for a turn; keeping raw text", exc_info=True)
            text = turn.text
        out.append({"speaker": turn.speaker, "text": text})
    return out
