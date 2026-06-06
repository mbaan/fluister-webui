"""Audio utilities: probe duration and convert to WAV using ffmpeg/ffprobe."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class AudioError(Exception):
    """Raised when an audio operation fails."""


def _require_binary(name: str) -> str:
    """Return the path to *name* or raise AudioError if not found."""
    path = shutil.which(name)
    if path is None:
        raise AudioError(
            f"Required binary '{name}' was not found on PATH. "
            "Please install ffmpeg."
        )
    return path


def probe_duration(path: str | Path) -> float:
    """Return audio duration in seconds using ffprobe.

    Runs:
        ffprobe -v error -show_entries format=duration
                -of default=noprint_wrappers=1:nokey=1 <path>

    Raises AudioError on non-zero exit code or unparseable output.
    """
    ffprobe = _require_binary("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-1000:]
        raise AudioError(
            f"ffprobe failed for '{path}' (exit {result.returncode}):\n{stderr_tail}"
        )
    output = result.stdout.strip()
    try:
        return float(output)
    except ValueError:
        raise AudioError(
            f"ffprobe returned unparseable output for '{path}': {output!r}"
        )


def convert_to_wav(src: str | Path, dst: str | Path) -> None:
    """Normalise any audio input to 16 kHz mono 16-bit PCM WAV.

    Runs:
        ffmpeg -y -i <src> -ac 1 -ar 16000 -c:a pcm_s16le -vn <dst>

    Creates dst's parent directory if it does not exist.
    Raises AudioError (with the last ~1000 chars of stderr) on non-zero exit.
    """
    ffmpeg = _require_binary("ffmpeg")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i", str(src),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            "-vn",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-1000:]
        raise AudioError(
            f"ffmpeg conversion failed for '{src}' -> '{dst}' "
            f"(exit {result.returncode}):\n{stderr_tail}"
        )
