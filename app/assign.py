"""Assign speaker labels from diarization turns to transcribed words or segments."""

from __future__ import annotations

from app.models import DiarTurn, Segment, Word


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Return the temporal overlap between [a0, a1] and [b0, b1]."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _assign_speaker(start: float, end: float, turns: list[DiarTurn]) -> str | None:
    """Pick the best speaker label for a time span against the given turns.

    First tries the turn with the largest overlap; if no turn overlaps, falls
    back to the nearest turn by midpoint distance.  Returns None when *turns*
    is empty.
    """
    if not turns:
        return None

    midpoint = (start + end) / 2.0

    best_turn: DiarTurn | None = None
    best_overlap = 0.0

    for turn in turns:
        ov = overlap(start, end, turn.start, turn.end)
        if ov > best_overlap:
            best_overlap = ov
            best_turn = turn

    if best_turn is not None:
        return best_turn.speaker

    # No overlap: snap to nearest turn by midpoint distance
    best_turn = min(turns, key=lambda t: abs((t.start + t.end) / 2.0 - midpoint))
    return best_turn.speaker


def words_to_speaker_segments(words: list[Word], turns: list[DiarTurn]) -> list[Segment]:
    """Combine word timings with diarization turns to produce speaker-labeled segments.

    Each word is assigned to the turn with the largest temporal overlap.  If a
    word overlaps no turn it is assigned to the nearest turn by midpoint distance
    (or None when *turns* is empty).  Consecutive words that share the same
    speaker label are then grouped into a single :class:`~app.models.Segment`.
    """
    if not words:
        return []

    # Assign a speaker label to every word
    labeled: list[tuple[Word, str | None]] = [
        (w, _assign_speaker(w.start, w.end, turns)) for w in words
    ]

    # Group consecutive words with the same speaker into segments
    segments: list[Segment] = []
    group_words: list[Word] = [labeled[0][0]]
    group_speaker: str | None = labeled[0][1]

    for word, speaker in labeled[1:]:
        if speaker == group_speaker:
            group_words.append(word)
        else:
            segments.append(
                Segment(
                    start=group_words[0].start,
                    end=group_words[-1].end,
                    text=" ".join(w.word for w in group_words).strip(),
                    speaker=group_speaker,
                    person_id=None,
                )
            )
            group_words = [word]
            group_speaker = speaker

    # Flush the last group
    segments.append(
        Segment(
            start=group_words[0].start,
            end=group_words[-1].end,
            text=" ".join(w.word for w in group_words).strip(),
            speaker=group_speaker,
            person_id=None,
        )
    )

    return segments


def segments_to_speaker_segments(
    segments: list[Segment], turns: list[DiarTurn]
) -> list[Segment]:
    """Assign speaker labels to existing segments using diarization turns.

    Fallback for when per-word timings are unavailable.  For each segment the
    turn with the largest overlap over ``[segment.start, segment.end]`` is
    chosen.  Returns new :class:`~app.models.Segment` objects with the same
    ``start``, ``end``, and ``text`` but with ``.speaker`` set and
    ``.person_id`` set to None.  Order is preserved.
    """
    result: list[Segment] = []
    for seg in segments:
        speaker = _assign_speaker(seg.start, seg.end, turns)
        result.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=speaker,
                person_id=None,
            )
        )
    return result
