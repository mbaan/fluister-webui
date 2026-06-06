"""Core transcription engine wrapping faster-whisper.

Uses BatchedInferencePipeline for throughput; falls back to the non-batched
WhisperModel on CUDA OOM, progressively shrinking batch_size before giving up.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

import ctranslate2
from faster_whisper import BatchedInferencePipeline, WhisperModel

from app.models import Segment, TranscribeInfo, Word

logger = logging.getLogger(__name__)

# Exception text fragments that indicate a CUDA out-of-memory condition.
_OOM_MARKERS = ("out of memory", "oom", "cuda failed", "cublas")


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _OOM_MARKERS)


def resolve_device(device: str) -> str:
    """Resolve "auto" to "cuda" or "cpu"; pass anything else through unchanged."""
    if device == "auto":
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    return device


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve "auto" to "float16" on CUDA or "int8" on CPU; pass-through otherwise."""
    if compute_type == "auto":
        return "float16" if device == "cuda" else "int8"
    return compute_type


class Transcriber:
    """Load a Whisper model once and expose a simple ``transcribe`` method."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        batch_size: int = 8,
        use_vad: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self.batch_size = batch_size
        self.use_vad = use_vad

        if self.device == "cuda":
            # Preload bundled cuBLAS/cuDNN so ctranslate2 can dlopen them by
            # soname without requiring LD_LIBRARY_PATH to be set.
            from app.cuda_libs import preload

            preload()

        logger.info(
            "Loading WhisperModel %s on %s (%s)",
            model_name,
            self.device,
            self.compute_type,
        )
        self.model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        self.pipeline = BatchedInferencePipeline(model=self.model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _consume(
        self,
        segments_iter: Iterable,
        info,
        duration: float,
        on_segment: Callable | None,
    ) -> tuple[list[Segment], list[Word]]:
        """Fully consume a segments generator, fire on_segment, return
        (segments, words). Words are collected when word_timestamps is on."""
        segments: list[Segment] = []
        words: list[Word] = []
        audio_duration = info.duration or duration
        denom = max(audio_duration, 0.001)

        for raw in segments_iter:
            seg = Segment(start=raw.start, end=raw.end, text=raw.text.strip())
            progress = min(seg.end / denom, 1.0)
            if on_segment is not None:
                on_segment(seg, progress)
            segments.append(seg)
            for w in getattr(raw, "words", None) or []:
                if w.start is None or w.end is None:
                    continue
                # faster-whisper word tokens carry a leading space; strip it so
                # assign.py can re-join speaker turns with single spaces.
                words.append(Word(start=w.start, end=w.end, word=(w.word or "").strip()))

        return segments, words

    def _run(
        self,
        wav_path,
        lang: str | None,
        batch_size: int,
        batched: bool,
        duration: float,
        on_segment: Callable | None,
    ) -> tuple[list[Segment], list[Word], object]:
        """Run transcription with given settings; returns (segments, words, info)."""
        if batched:
            segments_iter, info = self.pipeline.transcribe(
                wav_path,
                language=lang,
                batch_size=batch_size,
                vad_filter=self.use_vad,
                word_timestamps=True,
            )
        else:
            segments_iter, info = self.model.transcribe(
                wav_path,
                language=lang,
                vad_filter=self.use_vad,
                word_timestamps=True,
            )

        segments, words = self._consume(segments_iter, info, duration, on_segment)
        return segments, words, info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        wav_path,
        duration: float,
        language: str | None = None,
        on_segment: Callable | None = None,
    ) -> tuple[list[Segment], list[Word], TranscribeInfo]:
        """Transcribe *wav_path* and return ``(segments, words, TranscribeInfo)``.

        Parameters
        ----------
        wav_path:
            Path to a WAV file (or any audio format faster-whisper accepts).
        duration:
            Expected audio duration in seconds; used for progress calculation
            when ``info.duration`` is not yet available.
        language:
            ISO-639-1 code (e.g. ``"en"``) or ``None``/``"auto"`` for
            auto-detection.
        on_segment:
            Optional callback ``(seg: Segment, progress: float) -> None``
            called after each segment is decoded.
        """
        # Normalise language: "auto" or empty string -> None (auto-detect)
        lang = language if (language and language != "auto") else None

        # Build the fallback ladder: batched with shrinking batch_size, then
        # non-batched as the final attempt.
        batch_sizes = []
        bs = self.batch_size
        while bs >= 1:
            batch_sizes.append(bs)
            bs //= 2
        # Remove duplicates while preserving order (e.g. 1->0 would duplicate)
        seen: set[int] = set()
        unique_bs: list[int] = []
        for b in batch_sizes:
            if b not in seen:
                seen.add(b)
                unique_bs.append(b)

        strategies: list[tuple[int, bool]] = [(b, True) for b in unique_bs]
        strategies.append((1, False))  # final non-batched fallback

        last_exc: Exception | None = None
        for batch_size, batched in strategies:
            label = f"batched(batch_size={batch_size})" if batched else "non-batched"
            try:
                logger.debug("Transcribing with %s", label)
                segments, words, info = self._run(
                    wav_path, lang, batch_size, batched, duration, on_segment
                )
                return (
                    segments,
                    words,
                    TranscribeInfo(
                        language=info.language,
                        duration=info.duration or duration,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                if _is_oom(exc):
                    logger.warning(
                        "OOM with %s — trying next fallback: %s", label, exc
                    )
                    last_exc = exc
                    continue
                raise  # non-OOM errors bubble up immediately

        raise RuntimeError(
            f"Transcription failed: all OOM fallback strategies exhausted "
            f"(tried {[s for s in strategies]}). Last error: {last_exc}"
        )
