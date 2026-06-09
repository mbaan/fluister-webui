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
