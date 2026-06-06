"""Tests for app.diarizer.

Default test run (no env vars needed):
    - Import smoke-test: module imports without loading any model.
    - resolve_device logic.
    - DiarizationError is a proper exception subclass.

Slow/integration tests (require a GPU + HF token + network):
    Guarded by @pytest.mark.slow and skipped unless both
    RUN_SLOW_TESTS=1 and HF_TOKEN are set in the environment.
"""

from __future__ import annotations

import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLOW = pytest.mark.slow

_slow_reason = (
    "Skipped: set RUN_SLOW_TESTS=1 and HF_TOKEN to run integration tests "
    "that load the gated pyannote model."
)


def _should_run_slow() -> bool:
    return (
        os.environ.get("RUN_SLOW_TESTS", "0") == "1"
        and bool(os.environ.get("HF_TOKEN"))
    )


# ---------------------------------------------------------------------------
# Fast tests — no model load
# ---------------------------------------------------------------------------


def test_import_without_model_load():
    """Importing the module must not trigger any model download."""
    from app.diarizer import Diarizer, DiarizationError, resolve_device  # noqa: F401

    assert callable(Diarizer)
    assert callable(resolve_device)
    assert issubclass(DiarizationError, Exception)


def test_resolve_device_auto():
    from app.diarizer import resolve_device

    result = resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert result == expected


def test_resolve_device_explicit_cuda():
    from app.diarizer import resolve_device

    assert resolve_device("cuda") == "cuda"


def test_resolve_device_explicit_cpu():
    from app.diarizer import resolve_device

    assert resolve_device("cpu") == "cpu"


def test_resolve_device_passthrough_arbitrary():
    from app.diarizer import resolve_device

    assert resolve_device("mps") == "mps"


def test_diarization_error_is_exception():
    from app.diarizer import DiarizationError

    err = DiarizationError("something went wrong")
    assert isinstance(err, Exception)
    assert "something went wrong" in str(err)


def test_diarizer_raises_on_bad_token():
    """Diarizer.__init__ should raise DiarizationError when from_pretrained
    returns None (simulates missing/invalid token or unaccepted model terms).
    """
    import unittest.mock as mock

    from app.diarizer import DiarizationError, Diarizer

    with mock.patch("pyannote.audio.Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = None
        with pytest.raises(DiarizationError, match="returned None"):
            Diarizer(
                model_name="pyannote/speaker-diarization-3.1",
                hf_token="invalid-token",
            )


def test_diarizer_raises_on_exception_from_pretrained(monkeypatch):
    """Diarizer.__init__ wraps arbitrary exceptions from from_pretrained."""
    import unittest.mock as mock

    from app.diarizer import DiarizationError, Diarizer

    with mock.patch("pyannote.audio.Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.side_effect = OSError("network error")
        with pytest.raises(DiarizationError, match="network error"):
            Diarizer(model_name="pyannote/speaker-diarization-3.1")


# ---------------------------------------------------------------------------
# Slow / integration tests
# ---------------------------------------------------------------------------


@_SLOW
@pytest.mark.skipif(not _should_run_slow(), reason=_slow_reason)
def test_diarize_real_audio(tmp_path):
    """Full round-trip: load model, diarize a synthetic wav, check output."""
    import numpy as np
    import soundfile as sf

    from app.diarizer import Diarizer
    from app.models import DiarTurn

    # Create a short stereo-silence wav — just enough to exercise the pipeline.
    sample_rate = 16_000
    duration_s = 5
    samples = np.zeros((sample_rate * duration_s,), dtype=np.float32)
    wav_path = tmp_path / "silence.wav"
    sf.write(str(wav_path), samples, sample_rate)

    hf_token = os.environ["HF_TOKEN"]
    d = Diarizer(hf_token=hf_token)
    turns, embeddings = d.diarize(wav_path)

    assert isinstance(turns, list)
    assert all(isinstance(t, DiarTurn) for t in turns)
    assert isinstance(embeddings, dict)
    for label, emb in embeddings.items():
        assert isinstance(label, str)
        assert isinstance(emb, list)
        assert len(emb) > 0
