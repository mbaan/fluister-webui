"""Extract a timestamp from an audio/video filename, with mtime fallback."""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ParsedTime:
    dt: datetime        # timezone-naive local datetime
    source: str         # "filename" | "mtime"
    has_time: bool      # False when only a date was found (time becomes 00:00:00)


def _try_datetime(year, month, day, hour=0, minute=0, second=0) -> datetime | None:
    """Validate ranges and construct datetime, returning None on failure."""
    try:
        y, mo, d, h, mi, s = int(year), int(month), int(day), int(hour), int(minute), int(second)
    except (TypeError, ValueError):
        return None
    if not (2000 <= y <= 2100):
        return None
    if not (1 <= mo <= 12):
        return None
    if not (1 <= d <= 31):
        return None
    if not (0 <= h <= 23):
        return None
    if not (0 <= mi <= 59):
        return None
    if not (0 <= s <= 59):
        return None
    try:
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


# Separator character classes used in patterns
_SEP = r'[-_.\s]'   # generic separator
_TSEP = r'[-:.\s]'  # time component separator


# ── ordered patterns ─────────────────────────────────────────────────────────

# Each entry: (compiled regex, extractor function)
# Extractor receives match object, returns (datetime, has_time) or None.

def _extract_signal_dash(m) -> tuple[datetime, bool] | None:
    """signal-YYYY-MM-DD-HH-MM-SS[-ms]"""
    dt = _try_datetime(m.group('Y'), m.group('Mo'), m.group('D'),
                       m.group('H'), m.group('Mi'), m.group('S'))
    return (dt, True) if dt else None

def _extract_whatsapp_at(m) -> tuple[datetime, bool] | None:
    """WhatsApp Audio/Image/Video YYYY-MM-DD at HH.MM.SS"""
    dt = _try_datetime(m.group('Y'), m.group('Mo'), m.group('D'),
                       m.group('H'), m.group('Mi'), m.group('S'))
    return (dt, True) if dt else None

def _extract_wa_compact(m) -> tuple[datetime, bool] | None:
    """PTT/IMG/VID-YYYYMMDD-WA####  (date only)"""
    dt = _try_datetime(m.group('Y'), m.group('Mo'), m.group('D'))
    return (dt, False) if dt else None

def _extract_telegram(m) -> tuple[datetime, bool] | None:
    """audio_YYYY-MM-DD_HH-MM-SS"""
    dt = _try_datetime(m.group('Y'), m.group('Mo'), m.group('D'),
                       m.group('H'), m.group('Mi'), m.group('S'))
    return (dt, True) if dt else None

def _extract_generic_date_time(m) -> tuple[datetime, bool] | None:
    """YYYY-MM-DD <sep> HH:MM:SS  (generic dashed date + time)"""
    dt = _try_datetime(m.group('Y'), m.group('Mo'), m.group('D'),
                       m.group('H'), m.group('Mi'), m.group('S'))
    return (dt, True) if dt else None

def _extract_generic_date_only(m) -> tuple[datetime, bool] | None:
    """YYYY-MM-DD  (date only, dashed)"""
    dt = _try_datetime(m.group('Y'), m.group('Mo'), m.group('D'))
    return (dt, False) if dt else None

def _extract_compact_datetime(m) -> tuple[datetime, bool] | None:
    """YYYYMMDD[_-]HHMMSS"""
    dt = _try_datetime(m.group('Y'), m.group('Mo'), m.group('D'),
                       m.group('H'), m.group('Mi'), m.group('S'))
    return (dt, True) if dt else None

def _extract_compact_date_only(m) -> tuple[datetime, bool] | None:
    """YYYYMMDD  (date only, compact)"""
    dt = _try_datetime(m.group('Y'), m.group('Mo'), m.group('D'))
    return (dt, False) if dt else None


_PATTERNS: list[tuple[re.Pattern, callable]] = [
    # 1. Signal: signal-YYYY-MM-DD-HH-MM-SS[-ms]
    #    signal-2026-06-06-094449.aac  (time run-together) OR
    #    signal-2026-06-04-16-21-28-808.m4a (time dash-separated)
    (
        re.compile(
            r'(?i)signal'
            r'-(?P<Y>\d{4})-(?P<Mo>\d{2})-(?P<D>\d{2})'
            r'-(?P<H>\d{2})[-.]?(?P<Mi>\d{2})[-.]?(?P<S>\d{2})'
            r'(?:-\d+)?'           # optional ms
        ),
        _extract_signal_dash,
    ),

    # 2. WhatsApp: "WhatsApp Audio/Image/Video YYYY-MM-DD at HH.MM.SS"
    (
        re.compile(
            r'(?i)whatsapp\s+(?:audio|image|video|ptt)'
            r'\s+(?P<Y>\d{4})-(?P<Mo>\d{2})-(?P<D>\d{2})'
            r'\s+at\s+'
            r'(?P<H>\d{2})\.(?P<Mi>\d{2})\.(?P<S>\d{2})'
        ),
        _extract_whatsapp_at,
    ),

    # 3. WhatsApp compact (PTT/IMG/VID-YYYYMMDD-WA####)
    (
        re.compile(
            r'(?i)(?:PTT|IMG|VID)'
            r'-(?P<Y>\d{4})(?P<Mo>\d{2})(?P<D>\d{2})'
            r'-WA\d+'
        ),
        _extract_wa_compact,
    ),

    # 4. Telegram style: audio_YYYY-MM-DD_HH-MM-SS
    (
        re.compile(
            r'(?i)(?:audio|video|voice|file)'
            r'_(?P<Y>\d{4})-(?P<Mo>\d{2})-(?P<D>\d{2})'
            r'_(?P<H>\d{2})-(?P<Mi>\d{2})-(?P<S>\d{2})'
        ),
        _extract_telegram,
    ),

    # 5. Generic dashed date + time (YYYY-MM-DD <sep> HH[:.\-_ ]MM[:.\-_ ]SS)
    (
        re.compile(
            r'(?P<Y>\d{4})-(?P<Mo>\d{2})-(?P<D>\d{2})'
            r'[-_T\s]+'
            r'(?P<H>\d{2})[:\.\-_ ](?P<Mi>\d{2})[:\.\-_ ](?P<S>\d{2})'
        ),
        _extract_generic_date_time,
    ),

    # 5b. Generic dashed date + run-together time (YYYY-MM-DD-HHMMSS)
    (
        re.compile(
            r'(?P<Y>\d{4})-(?P<Mo>\d{2})-(?P<D>\d{2})'
            r'[-_]'
            r'(?P<H>\d{2})(?P<Mi>\d{2})(?P<S>\d{2})'
            r'(?!\d)'   # not followed by another digit
        ),
        _extract_generic_date_time,
    ),

    # 6. Compact datetime: YYYYMMDD[_-]HHMMSS
    (
        re.compile(
            r'(?P<Y>\d{4})(?P<Mo>\d{2})(?P<D>\d{2})'
            r'[_-]'
            r'(?P<H>\d{2})(?P<Mi>\d{2})(?P<S>\d{2})'
            r'(?!\d)'
        ),
        _extract_compact_datetime,
    ),

    # 7. Generic dashed date only (YYYY-MM-DD)
    (
        re.compile(r'(?P<Y>\d{4})-(?P<Mo>\d{2})-(?P<D>\d{2})'),
        _extract_generic_date_only,
    ),

    # 8. Compact date only (YYYYMMDD)
    (
        re.compile(
            r'(?<!\d)(?P<Y>\d{4})(?P<Mo>\d{2})(?P<D>\d{2})(?!\d)'
        ),
        _extract_compact_date_only,
    ),
]


def parse_filename_timestamp(name: str) -> ParsedTime | None:
    """Parse a timestamp from a filename (basename, may include extension).
    Return ParsedTime(source='filename') or None if no plausible timestamp."""
    # Strip extension for matching, but keep full name for context
    stem = Path(name).stem

    for pattern, extractor in _PATTERNS:
        m = pattern.search(stem) or pattern.search(name)
        if m:
            result = extractor(m)
            if result is not None:
                dt, has_time = result
                return ParsedTime(dt=dt, source="filename", has_time=has_time)
    return None


def resolve_timestamp(path: str | Path) -> ParsedTime:
    """Try parse_filename_timestamp on the basename; if None, use the file's
    mtime -> ParsedTime(source='mtime', has_time=True)."""
    p = Path(path)
    parsed = parse_filename_timestamp(p.name)
    if parsed is not None:
        return parsed
    mtime = os.stat(p).st_mtime
    dt = datetime.fromtimestamp(mtime).replace(microsecond=0)
    return ParsedTime(dt=dt, source="mtime", has_time=True)
