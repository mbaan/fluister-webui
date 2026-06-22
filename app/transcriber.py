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

# Files shorter than this (seconds) never trigger the VAD over-filter retry:
# on a brief clip a low coverage ratio is noisy and the retry buys little.
_VAD_RETRY_MIN_DURATION = 30.0


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
        vad_min_coverage: float = 0.5,
    ) -> None:
        self.device = resolve_device(device)
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self.batch_size = batch_size
        self.use_vad = use_vad
        self.vad_min_coverage = vad_min_coverage

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
        hotwords: str | None,
        use_vad: bool,
    ) -> tuple[list[Segment], list[Word], object]:
        """Run transcription with given settings; returns (segments, words, info)."""
        if batched:
            segments_iter, info = self.pipeline.transcribe(
                wav_path,
                language=lang,
                batch_size=batch_size,
                vad_filter=use_vad,
                word_timestamps=True,
                hotwords=hotwords,
            )
        else:
            segments_iter, info = self.model.transcribe(
                wav_path,
                language=lang,
                vad_filter=use_vad,
                word_timestamps=True,
                hotwords=hotwords,
            )

        segments, words = self._consume(segments_iter, info, duration, on_segment)
        return segments, words, info

    def _transcribe_once(
        self,
        wav_path,
        lang: str | None,
        duration: float,
        on_segment: Callable | None,
        hotwords: str | None,
        use_vad: bool,
    ) -> tuple[list[Segment], list[Word], object]:
        """One full transcription pass with the OOM fallback ladder.

        With VAD on, tries batched inference with a shrinking ``batch_size``,
        then a final non-batched attempt, stepping down only on CUDA OOM. With
        VAD off, only the non-batched path is used: ``BatchedInferencePipeline``
        derives its batches from VAD clip timestamps and refuses to run without
        them. Returns the raw ``(segments, words, info)`` from the first strategy
        that succeeds.
        """
        if not use_vad:
            # Batched inference needs VAD clip timestamps, so VAD-off must go
            # through the plain (non-batched) model.
            strategies: list[tuple[int, bool]] = [(1, False)]
        else:
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

            strategies = [(b, True) for b in unique_bs]
            strategies.append((1, False))  # final non-batched fallback

        last_exc: Exception | None = None
        for batch_size, batched in strategies:
            label = f"batched(batch_size={batch_size})" if batched else "non-batched"
            try:
                logger.debug("Transcribing with %s (vad=%s)", label, use_vad)
                return self._run(
                    wav_path, lang, batch_size, batched,
                    duration, on_segment, hotwords, use_vad,
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        wav_path,
        duration: float,
        language: str | None = None,
        on_segment: Callable | None = None,
        hotwords: str | None = None,
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
        hotwords:
            Optional space/comma-separated terms to bias the decoder toward
            (e.g. names the recogniser tends to mishear). ``None`` applies no bias.
        """
        # Normalise language: "auto" or empty string -> None (auto-detect)
        lang = language if (language and language != "auto") else None

        segments, words, info = self._transcribe_once(
            wav_path, lang, duration, on_segment, hotwords, use_vad=self.use_vad
        )

        # VAD over-filter rescue: a loud, continuously-noisy recording (e.g.
        # speech recorded in a moving car) can fool Silero VAD into discarding
        # almost the whole file, so whisper transcribes only the first chunk and
        # "stops". When VAD was on and the transcript covers only a small
        # fraction of a non-trivially-long file, retry once with VAD off — the
        # speech is usually there and decodes fine without the filter.
        if self.use_vad:
            segments, words, info = self._maybe_retry_without_vad(
                wav_path, lang, duration, on_segment, hotwords,
                segments, words, info,
            )

        return (
            segments,
            words,
            TranscribeInfo(
                language=info.language,
                duration=info.duration or duration,
            ),
        )

    # ------------------------------------------------------------------
    # VAD over-filter rescue
    # ------------------------------------------------------------------

    def _maybe_retry_without_vad(
        self,
        wav_path,
        lang: str | None,
        duration: float,
        on_segment: Callable | None,
        hotwords: str | None,
        segments: list[Segment],
        words: list[Word],
        info,
    ) -> tuple[list[Segment], list[Word], object]:
        """Re-run with VAD off when the VAD pass covered too little of the audio.

        Returns the no-VAD result when it triggers, otherwise the originals
        unchanged. The decision compares how far the last decoded segment reaches
        against the audio duration; below ``vad_min_coverage`` on a file longer
        than ``_VAD_RETRY_MIN_DURATION`` seconds, the VAD almost certainly
        dropped real (noisy) speech.
        """
        audio_duration = info.duration or duration
        if audio_duration < _VAD_RETRY_MIN_DURATION:
            return segments, words, info

        covered = segments[-1].end if segments else 0.0
        coverage = covered / max(audio_duration, 1e-3)
        if coverage >= self.vad_min_coverage:
            return segments, words, info

        logger.warning(
            "VAD kept only %.1f%% of %.0fs audio (last segment ends at %.1fs) — "
            "retrying without VAD.",
            coverage * 100, audio_duration, covered,
        )
        return self._transcribe_once(
            wav_path, lang, duration, on_segment, hotwords, use_vad=False
        )
