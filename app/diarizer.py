"""Speaker diarization module.

Uses pyannote.audio 4.x SpeakerDiarization pipeline.

In pyannote 4.x the pipeline returns a ``DiarizeOutput`` dataclass (not a
tuple) with:
    - ``speaker_diarization``   : pyannote.core.Annotation
    - ``speaker_embeddings``    : np.ndarray, shape (num_speakers, dim),
                                  rows ordered by ``annotation.labels()``

There is no ``return_embeddings`` kwarg; embeddings are always in the output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from app.models import DiarTurn

logger = logging.getLogger(__name__)


class DiarizationError(Exception):
    """Raised when the diarization pipeline cannot be loaded or run."""


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` or ``"cpu"``; passthrough otherwise."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Diarizer:
    """Wrapper around the pyannote speaker-diarization pipeline.

    Parameters
    ----------
    model_name:
        HuggingFace model id or local path.
    device:
        ``"auto"`` (default), ``"cuda"``, or ``"cpu"``.
    hf_token:
        HuggingFace access token.  Required for gated models; can also be
        provided via the ``HF_TOKEN`` environment variable by passing
        ``hf_token=None`` (pyannote will pick it up automatically).
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "auto",
        hf_token: str | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model_name = model_name

        try:
            from pyannote.audio import Pipeline  # local import keeps module light

            pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
        except Exception as exc:
            raise DiarizationError(
                f"Failed to load diarization pipeline '{model_name}': {exc}\n"
                "Make sure you have:\n"
                "  1. A valid HuggingFace token (HF_TOKEN env var or hf_token arg).\n"
                "  2. Accepted the model terms at https://huggingface.co/{model_name}."
            ) from exc

        if pipeline is None:
            raise DiarizationError(
                f"Pipeline.from_pretrained('{model_name}') returned None.\n"
                "Possible causes:\n"
                "  • Missing or invalid HuggingFace token (set HF_TOKEN or pass hf_token=).\n"
                f"  • You have not accepted the model terms at "
                f"https://huggingface.co/{model_name}."
            )

        pipeline.to(torch.device(self.device))
        self._pipeline = pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diarize(
        self,
        wav_path: str | Path,
    ) -> tuple[list[DiarTurn], dict[str, list[float]]]:
        """Run speaker diarization on *wav_path*.

        Returns
        -------
        turns:
            List of :class:`~app.models.DiarTurn` in timeline order.
        embeddings_by_label:
            Mapping from local speaker label (e.g. ``"SPEAKER_00"``) to its
            centroid embedding as a plain Python list of floats.  Speakers
            whose embedding row is all-NaN are omitted (they still appear in
            *turns*).
        """
        wav_path = str(wav_path)

        try:
            output = self._pipeline(wav_path)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning(
                    "CUDA OOM during diarization — retrying on CPU."
                )
                self._pipeline.to(torch.device("cpu"))
                self.device = "cpu"
                output = self._pipeline(wav_path)
            else:
                raise

        # pyannote 4.x returns a DiarizeOutput dataclass.
        # Older builds (legacy=True) may return a bare Annotation.
        if hasattr(output, "speaker_diarization"):
            diarization = output.speaker_diarization
            raw_embeddings: np.ndarray | None = output.speaker_embeddings
        else:
            # Fallback: legacy Annotation-only return (should not happen with
            # the 3.1 checkpoint and pyannote 4.x, but be defensive).
            diarization = output
            raw_embeddings = None

        # ---- turns -------------------------------------------------------
        turns: list[DiarTurn] = [
            DiarTurn(t.start, t.end, label)
            for t, _, label in diarization.itertracks(yield_label=True)
        ]

        # ---- embeddings --------------------------------------------------
        # Labels are sorted; the i-th row of speaker_embeddings corresponds to
        # the i-th element of diarization.labels().
        embeddings_by_label: dict[str, list[float]] = {}

        if raw_embeddings is not None and raw_embeddings.ndim == 2:
            labels = diarization.labels()  # sorted list
            for idx, label in enumerate(labels):
                if idx >= raw_embeddings.shape[0]:
                    break
                row = raw_embeddings[idx]
                if np.all(np.isnan(row)):
                    # No valid embedding for this speaker — skip the row but
                    # keep the speaker's turns in the turns list.
                    logger.debug("Skipping all-NaN embedding for %s", label)
                    continue
                embeddings_by_label[label] = row.tolist()

        return turns, embeddings_by_label
