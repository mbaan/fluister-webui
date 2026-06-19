"""On-device insight pass: a brief summary, key points, and timestamped
chapters derived from a finished transcript via the local llama-server.

Pure + HTTP-only (reuses the tidier's chat_completion) so it stays cheap to
import and easy to test. Best-effort: every caller treats a None return as
"no insights" and continues.

This is *additive comprehension*, not transcript editing — it never changes the
transcript, segments, or readable text. See the design doc and the project's
"LLM = tidier, not fixer" guardrail.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.tidier import chat_completion

logger = logging.getLogger(__name__)


INSIGHT_SYSTEM = (
    "You analyze a transcript and produce a brief, faithful overview. The "
    "transcript is given as lines each prefixed with a timestamp like [mm:ss]. "
    "Return ONLY a JSON object (no prose, no code fence) with keys: "
    '"summary" (a 2-3 sentence plain overview), '
    '"key_points" (3-7 short bullet strings: the most important points or action items), '
    '"chapters" (3-8 objects {"time": "mm:ss", "title": "short topic title"} marking where '
    "each new topic starts; use timestamps that actually appear in the transcript). "
    "Write the summary, key_points and titles in the same language as the transcript. "
    "Do not invent anything — base everything strictly on the transcript."
)


def _mmss(sec: float) -> str:
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_time(v: Any) -> float | None:
    """Accept a number of seconds, "mm:ss", or "h:mm:ss"."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        if ":" in v:
            try:
                parts = [int(p) for p in v.split(":")]
            except ValueError:
                return None
            sec = 0
            for p in parts:
                sec = sec * 60 + p
            return float(sec)
        try:
            return float(v)
        except ValueError:
            return None
    return None


def build_transcript_block(segments, max_chars: int = 16000) -> str:
    """Render segments as timestamped lines, capped so the prompt stays inside
    the model context. Segments are objects with ``.text`` and ``.start``."""
    lines: list[str] = []
    total = 0
    for s in segments:
        text = (getattr(s, "text", "") or "").strip()
        if not text:
            continue
        line = f"[{_mmss(getattr(s, 'start', 0) or 0)}] {text}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _extract_json(raw: str) -> Any:
    """Pull the first JSON object out of a model reply, tolerating code fences
    and surrounding prose."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0 or j < i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def generate_insights(
    segments,
    base_url: str,
    *,
    model: str = "local",
    temperature: float = 0.2,
    timeout: int = 180,
) -> dict | None:
    """Summarise a transcript into {summary, key_points, chapters}. Returns None
    on empty input or any LLM/parse failure (callers stay best-effort)."""
    block = build_transcript_block(segments)
    if not block.strip():
        return None
    messages = [
        {"role": "system", "content": INSIGHT_SYSTEM},
        {"role": "user", "content": block},
    ]
    try:
        raw = chat_completion(
            base_url, messages, model=model, temperature=temperature, timeout=timeout
        )
    except Exception:  # noqa: BLE001
        logger.warning("Insight LLM call failed", exc_info=True)
        return None

    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None

    summary = str(data.get("summary") or "").strip()
    key_points = [
        str(x).strip() for x in (data.get("key_points") or []) if str(x).strip()
    ][:8]
    chapters: list[dict] = []
    for ch in (data.get("chapters") or []):
        if not isinstance(ch, dict):
            continue
        title = str(ch.get("title") or "").strip()
        start = _parse_time(ch.get("time", ch.get("start")))
        if title and start is not None:
            chapters.append({"title": title, "start": round(float(start), 2)})
    chapters = chapters[:10]

    if not (summary or key_points or chapters):
        return None
    return {"summary": summary, "key_points": key_points, "chapters": chapters}
