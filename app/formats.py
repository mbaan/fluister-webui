"""Output format serialisers for transcription results.

Supported formats: plain text (TXT), SubRip (SRT), WebVTT (VTT), JSON.
Stdlib only – no third-party dependencies.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

from app.models import Segment, TranscriptMeta


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm."""
    return _format_time(seconds, ms_sep=",")


def _format_vtt_time(seconds: float) -> str:
    """Convert seconds to VTT timestamp format: HH:MM:SS.mmm."""
    return _format_time(seconds, ms_sep=".")


def _format_time(seconds: float, ms_sep: str) -> str:
    """Shared time formatter for SRT and VTT."""
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}{ms_sep}{ms:03d}"


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS (e.g. 1:02:03)."""
    total_s = int(math.floor(seconds))
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_txt(segments: list[Segment], meta: TranscriptMeta) -> str:
    """Return a human-readable plain-text transcript with a header."""
    lines: list[str] = []

    lines.append(f"File: {meta.filename}")
    if meta.msg_timestamp is not None:
        source = meta.msg_timestamp_source or ""
        lines.append(f"Sent: {meta.msg_timestamp} ({source})")
    lines.append(
        f"Language: {meta.language}   Duration: {_format_duration(meta.duration)}"
        f"   Model: {meta.model}"
    )
    lines.append("")
    lines.append("-" * 60)

    last_speaker: str | None = None
    for seg in segments:
        text = seg.text.strip()
        if text:
            if seg.speaker is not None and seg.speaker != last_speaker:
                lines.append(f"{seg.speaker}: {text}")
                last_speaker = seg.speaker
            elif seg.speaker is not None:
                lines.append(text)
            else:
                lines.append(text)

    return "\n".join(lines)


def to_srt(segments: list[Segment]) -> str:
    """Return a SubRip (.srt) formatted transcript."""
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _format_srt_time(seg.start)
        end = _format_srt_time(seg.end)
        text = seg.text.strip()
        if seg.speaker is not None:
            text = f"{seg.speaker}: {text}"
        block = f"{i}\n{start} --> {end}\n{text}"
        blocks.append(block)
    # Each block separated by a blank line; trailing blank line at end.
    return "\n\n".join(blocks) + "\n\n"


def to_vtt(segments: list[Segment]) -> str:
    """Return a WebVTT (.vtt) formatted transcript."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _format_vtt_time(seg.start)
        end = _format_vtt_time(seg.end)
        text = seg.text.strip()
        if seg.speaker is not None:
            text = f"{seg.speaker}: {text}"
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_json(segments: list[Segment], meta: TranscriptMeta, speakers: dict | None = None) -> str:
    """Return a JSON string with meta and segments.

    When *speakers* is provided, a top-level ``"speakers"`` key is included.
    Each segment dict always contains ``"speaker"`` and ``"person_id"`` fields
    (values may be null).
    """
    meta_dict = {
        "filename": meta.filename,
        "language": meta.language,
        "duration": meta.duration,
        "model": meta.model,
        "msg_timestamp": meta.msg_timestamp,
        "msg_timestamp_source": meta.msg_timestamp_source,
    }
    segments_list = [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "speaker": seg.speaker,
            "person_id": seg.person_id,
        }
        for seg in segments
    ]
    payload: dict = {"meta": meta_dict, "segments": segments_list}
    if speakers is not None:
        payload["speakers"] = speakers
    return json.dumps(payload, ensure_ascii=False, indent=2)
