"""Tests for app.transcriber.

Fast unit tests run unconditionally.
The slow end-to-end test (``@pytest.mark.slow``) requires ``RUN_SLOW_TESTS=1``
in the environment and downloads/uses the tiny Whisper model on CPU.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from app.models import Segment
from app.transcriber import Transcriber, resolve_compute_type, resolve_device


# ---------------------------------------------------------------------------
# Unit tests – resolve_device
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_auto_returns_cuda_when_gpu_present(self):
        with patch("ctranslate2.get_cuda_device_count", return_value=1):
            assert resolve_device("auto") == "cuda"

    def test_auto_returns_cpu_when_no_gpu(self):
        with patch("ctranslate2.get_cuda_device_count", return_value=0):
            assert resolve_device("auto") == "cpu"

    def test_explicit_cuda_passthrough(self):
        assert resolve_device("cuda") == "cuda"

    def test_explicit_cpu_passthrough(self):
        assert resolve_device("cpu") == "cpu"

    def test_arbitrary_device_passthrough(self):
        assert resolve_device("cuda:1") == "cuda:1"


# ---------------------------------------------------------------------------
# Unit tests – resolve_compute_type
# ---------------------------------------------------------------------------


class TestResolveComputeType:
    def test_auto_cuda_gives_float16(self):
        assert resolve_compute_type("auto", "cuda") == "float16"

    def test_auto_cpu_gives_int8(self):
        assert resolve_compute_type("auto", "cpu") == "int8"

    def test_explicit_passthrough_float32(self):
        assert resolve_compute_type("float32", "cuda") == "float32"

    def test_explicit_passthrough_int8(self):
        assert resolve_compute_type("int8", "cuda") == "int8"

    def test_explicit_passthrough_on_cpu(self):
        assert resolve_compute_type("float32", "cpu") == "float32"


# ---------------------------------------------------------------------------
# Unit tests – VAD over-filter retry
#
# When VAD is on and the transcript covers only a tiny fraction of the audio,
# the recording is most likely noisy speech that Silero VAD wrongly discarded;
# the transcriber should retry once with VAD off. These tests stub the
# per-pass helper ``_transcribe_once`` so no model is loaded.
# ---------------------------------------------------------------------------


def _bare_transcriber(use_vad: bool = True, vad_min_coverage: float = 0.5) -> Transcriber:
    """A Transcriber instance with the retry knobs set but no model loaded."""
    t = Transcriber.__new__(Transcriber)
    t.use_vad = use_vad
    t.batch_size = 8
    t.vad_min_coverage = vad_min_coverage
    return t


def _info(language: str = "en", duration: float = 200.0):
    return types.SimpleNamespace(language=language, duration=duration)


class TestVadCoverageRetry:
    def _stub(self, t: Transcriber, results: dict):
        """Make ``_transcribe_once`` return results[use_vad] and record calls."""
        calls: list[bool] = []

        def fake_once(wav_path, lang, duration, on_segment, hotwords, use_vad):
            calls.append(use_vad)
            return results[use_vad]

        t._transcribe_once = fake_once
        return calls

    def test_low_coverage_retries_without_vad(self):
        t = _bare_transcriber()
        vad = ([Segment(0.0, 2.0, "they're being hello")], [], _info(duration=200.0))
        novad = ([Segment(0.0, 198.0, "full transcript")], [], _info(duration=200.0))
        calls = self._stub(t, {True: vad, False: novad})

        segs, _words, info = t.transcribe("x.wav", duration=200.0)

        assert calls == [True, False]  # retried with VAD off
        assert segs[0].end == 198.0  # returned the no-VAD result
        assert info.duration == 200.0

    def test_good_coverage_does_not_retry(self):
        t = _bare_transcriber()
        good = ([Segment(0.0, 196.0, "full")], [], _info(duration=200.0))
        calls = self._stub(t, {True: good})

        segs, _words, _inf = t.transcribe("x.wav", duration=200.0)

        assert calls == [True]  # no retry
        assert segs[0].end == 196.0

    def test_short_file_does_not_retry(self):
        t = _bare_transcriber()
        short = ([Segment(0.0, 0.5, "hi")], [], _info(duration=5.0))
        calls = self._stub(t, {True: short})

        t.transcribe("x.wav", duration=5.0)

        assert calls == [True]  # too short to bother retrying

    def test_vad_already_off_never_retries(self):
        t = _bare_transcriber(use_vad=False)
        res = ([Segment(0.0, 2.0, "hi")], [], _info(duration=200.0))
        calls = self._stub(t, {False: res})

        t.transcribe("x.wav", duration=200.0)

        assert calls == [False]  # no VAD pass to retry around


# ---------------------------------------------------------------------------
# Slow end-to-end test (skipped unless RUN_SLOW_TESTS=1)
# ---------------------------------------------------------------------------

slow = pytest.mark.slow

_SKIP_SLOW = pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run slow model tests",
)


def _make_tone_wav(path: str, duration_s: float = 3.0, freq: int = 440) -> None:
    """Generate a sine-tone WAV using ffmpeg (lavfi source)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={duration_s}",
            "-ar",
            "16000",
            "-ac",
            "1",
            path,
        ],
        check=True,
        capture_output=True,
    )


@slow
@_SKIP_SLOW
def test_transcriber_tiny_cpu_smoke():
    """End-to-end smoke test: load tiny model on CPU and transcribe a tone.

    The tone has no speech, so segments/text may be empty or garbage — we only
    assert the return types and shapes are correct.
    """
    from app.models import Segment, TranscribeInfo
    from app.transcriber import Transcriber

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = str(Path(tmpdir) / "tone.wav")
        duration = 3.0
        _make_tone_wav(wav_path, duration_s=duration)

        t = Transcriber(
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            batch_size=8,
            use_vad=True,
        )

        collected_segs: list = []
        collected_progress: list[float] = []

        def on_seg(seg, progress):
            collected_segs.append(seg)
            collected_progress.append(progress)

        segments, words, info = t.transcribe(
            wav_path, duration=duration, on_segment=on_seg
        )

        # Return type assertions
        assert isinstance(segments, list), "segments must be a list"
        assert isinstance(words, list), "words must be a list"
        assert isinstance(info, TranscribeInfo), "info must be TranscribeInfo"
        assert isinstance(info.language, str), "language must be a string"
        assert isinstance(info.duration, float), "duration must be a float"
        assert info.duration > 0, "duration must be positive"

        # on_segment callback consistency
        assert len(collected_segs) == len(segments)
        for seg, prog in zip(collected_segs, collected_progress):
            assert isinstance(seg, Segment)
            assert 0.0 <= prog <= 1.0

        # Segment field types
        for seg in segments:
            assert isinstance(seg.start, float)
            assert isinstance(seg.end, float)
            assert isinstance(seg.text, str)


@slow
@_SKIP_SLOW
def test_transcriber_language_auto_and_none_equivalent():
    """Both language=None and language='auto' should work without error."""
    from app.transcriber import Transcriber

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = str(Path(tmpdir) / "tone.wav")
        _make_tone_wav(wav_path, duration_s=2.0)

        t = Transcriber(model_name="tiny", device="cpu", compute_type="int8")

        segs_none, _w1, info_none = t.transcribe(wav_path, duration=2.0, language=None)
        segs_auto, _w2, info_auto = t.transcribe(wav_path, duration=2.0, language="auto")

        assert isinstance(segs_none, list)
        assert isinstance(segs_auto, list)
        assert info_none.language == info_auto.language
