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
from typing import Callable, Iterable

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
    "You clean up speech-to-text transcripts for readability. Add punctuation and "
    "capitalization, and break the text into paragraphs only at clear shifts in "
    "topic — group related sentences together and avoid one- or two-sentence "
    "paragraphs. Remove filler words (uh, um, like, you know) and false starts / "
    "repeated restarts. Do NOT change, add, remove (other than fillers), reorder, "
    "translate, or 'correct' any meaningful word. Preserve the original language(s) "
    "exactly, including Dutch-English code-switching. Output only the cleaned text, "
    "nothing else."
)

# A Dutch transcript tidied under the English prompt above tends to come back
# translated into English: the small local model quietly answers in the language
# it's *addressed* in. The reliable fix is to address it in the transcript's own
# language, so we keep a localised prompt per language. SYSTEM_PROMPT is the
# English entry; add a translation here to anchor another language natively.
SYSTEM_PROMPT_NL = (
    "Je maakt automatisch gegenereerde transcripties beter leesbaar. Voeg "
    "interpunctie en hoofdletters toe en splits de tekst alleen in alinea's bij "
    "een duidelijke wisseling van onderwerp — groepeer zinnen die bij elkaar "
    "horen en vermijd alinea's van één of twee zinnen. Verwijder stopwoorden "
    "(eh, ehm, zeg maar, weet je) en valse starts / herhaalde herstarts. "
    "Verander, voeg toe, verwijder (behalve stopwoorden), herorden, vertaal of "
    "'corrigeer' GEEN enkel betekenisvol woord. Behoud de oorspronkelijke taal "
    "of talen exact, inclusief Nederlands-Engelse code-switching. Geef alleen de "
    "opgeschoonde tekst terug, verder niets."
)

SYSTEM_PROMPTS = {"en": SYSTEM_PROMPT, "nl": SYSTEM_PROMPT_NL}

# ISO 639-1 codes faster-whisper emits → English language names, so that for a
# language we have no localised prompt for we can still name it explicitly in the
# fallback prompt ("write your output in German") instead of staying vague.
LANGUAGE_NAMES = {
    "en": "English", "nl": "Dutch", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "pl": "Polish",
    "ru": "Russian", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
    "id": "Indonesian", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "nb": "Norwegian", "nn": "Norwegian", "fi": "Finnish", "cs": "Czech",
    "sk": "Slovak", "el": "Greek", "he": "Hebrew", "ro": "Romanian",
    "hu": "Hungarian", "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian",
    "ca": "Catalan", "th": "Thai", "vi": "Vietnamese", "fa": "Persian",
}


def language_name(code: str | None) -> str | None:
    """Map a whisper language code (``"nl"``) to an English name (``"Dutch"``).
    An unknown but non-empty code returns itself; ``None``, ``""`` or ``"auto"``
    (language not yet detected) return ``None``."""
    code = (code or "").strip().lower().split("-")[0]
    if not code or code == "auto":
        return None
    return LANGUAGE_NAMES.get(code, code)


def build_system_prompt(language: str | None = None) -> str:
    """The tidy system prompt for the transcript's language. Prefer addressing
    the model in that language (a localised prompt) so it answers in kind; for a
    language we have no translation for, fall back to the English prompt plus an
    explicit "write in <language>" anchor; with no known language, the plain
    English prompt."""
    code = (language or "").strip().lower().split("-")[0]
    if code in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[code]
    name = language_name(code)
    if name is None:
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT} The transcript is in {name}; write your entire output "
        f"in {name} and never translate any part of it into another language."
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
    language: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Tidy each turn independently. On a per-turn failure, fall back to that
    turn's raw text (degrade gracefully — never drop content). Returns a list of
    ``{"speaker": str | None, "text": str}`` paragraphs.

    ``language`` is the whisper-detected language code; it picks the system
    prompt so the model cleans up in that language rather than translating.

    ``on_progress(done, total)`` is called after each turn so callers can
    surface progress; callback errors are swallowed — progress reporting must
    never cost us the tidied text."""
    out: list[dict] = []
    total = len(turns)
    system_prompt = build_system_prompt(language)
    for i, turn in enumerate(turns, start=1):
        messages = [
            {"role": "system", "content": system_prompt},
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
        if on_progress is not None:
            try:
                on_progress(i, total)
            except Exception:  # noqa: BLE001
                logger.debug("Tidy progress callback failed", exc_info=True)
    return out
