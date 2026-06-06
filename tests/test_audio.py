"""Tests for app.audio — probe_duration and convert_to_wav."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from app.audio import AudioError, convert_to_wav, probe_duration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
_SKIP_NO_FFMPEG = pytest.mark.skipif(
    not _FFMPEG_AVAILABLE, reason="ffmpeg is not installed"
)


def _make_source_audio(path):
    """Generate a 2-second stereo 44100 Hz m4a test file using ffmpeg lavfi."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=2",
            "-ac", "2",
            "-ar", "44100",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Failed to create test audio file:\n{result.stderr[-500:]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_SKIP_NO_FFMPEG
def test_convert_to_wav_creates_nonempty_file(tmp_path):
    """convert_to_wav produces a non-empty WAV file at the destination."""
    src = tmp_path / "src.m4a"
    dst = tmp_path / "out" / "dst.wav"
    _make_source_audio(src)

    convert_to_wav(src, dst)

    assert dst.exists(), "dst WAV file should exist after conversion"
    assert dst.stat().st_size > 0, "dst WAV file should not be empty"


@_SKIP_NO_FFMPEG
def test_probe_duration_approximately_two_seconds(tmp_path):
    """probe_duration returns ~2.0 for a 2-second synthesised file."""
    src = tmp_path / "src.m4a"
    dst = tmp_path / "probe.wav"
    _make_source_audio(src)
    convert_to_wav(src, dst)

    duration = probe_duration(dst)

    assert abs(duration - 2.0) < 0.3, (
        f"Expected duration ~2.0 s, got {duration}"
    )


@_SKIP_NO_FFMPEG
def test_convert_to_wav_raises_on_non_audio_file(tmp_path):
    """convert_to_wav raises AudioError when the source is not audio."""
    bad_src = tmp_path / "not_audio.txt"
    bad_src.write_text("not audio")
    dst = tmp_path / "out.wav"

    with pytest.raises(AudioError):
        convert_to_wav(bad_src, dst)


@_SKIP_NO_FFMPEG
def test_probe_duration_raises_on_garbage_file(tmp_path):
    """probe_duration raises AudioError when the file contains garbage."""
    bad_file = tmp_path / "garbage.wav"
    bad_file.write_bytes(b"\x00\x01\x02\x03garbage data that is not a valid audio file")

    with pytest.raises(AudioError):
        probe_duration(bad_file)


@_SKIP_NO_FFMPEG
def test_convert_to_wav_creates_parent_directory(tmp_path):
    """convert_to_wav creates the destination parent directory if needed."""
    src = tmp_path / "src.m4a"
    dst = tmp_path / "nested" / "deeply" / "output.wav"
    _make_source_audio(src)

    convert_to_wav(src, dst)

    assert dst.exists()
    assert dst.stat().st_size > 0
