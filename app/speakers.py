"""Global voice gallery: persons persist across all files.

Matching a voice means matching against all known persons by centroid cosine
similarity. A new person is created when no existing person is close enough.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import numpy as np

from app import db

# Auto-assigned gallery names like "Person 3" carry no signal — exclude them so
# they don't pollute the hotword bias.
_PLACEHOLDER_NAME = re.compile(r"^Person \d+$")


def build_hotwords(persons: list[dict]) -> str | None:
    """Union of all persons' names + keyword lists, for faster-whisper ``hotwords``.

    Each person contributes their display name (unless it's a ``Person N``
    placeholder) plus every term in their ``keywords`` field (split on commas and
    newlines). Terms are de-duplicated case-insensitively, preserving first-seen
    order and casing, and joined with ``", "``. Returns ``None`` when nothing
    qualifies, so callers pass no bias rather than an empty string.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        t = term.strip()
        if not t:
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        terms.append(t)

    for p in persons:
        name = (p.get("name") or "").strip()
        if name and not _PLACEHOLDER_NAME.match(name):
            add(name)
        for raw in re.split(r"[,\n]", p.get("keywords") or ""):
            add(raw)

    return ", ".join(terms) if terms else None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D numpy vectors. Returns 0 if a is a zero vector."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(a))
    if norm_a == 0.0:
        return 0.0
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _emb_to_bytes(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _bytes_to_emb(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


class Gallery:
    def __init__(self, db_path: Path, threshold: float = 0.45) -> None:
        self.db_path = db_path
        self.threshold = threshold

    # ── public API ────────────────────────────────────────────────────────────

    def identify(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Nearest person by centroid cosine.

        Returns (person_id, best_sim) or (None, 0.0) if the gallery is empty.
        """
        persons = db.list_persons(self.db_path)
        if not persons:
            return None, 0.0

        emb = np.asarray(embedding, dtype=np.float32)
        best_id: str | None = None
        best_sim: float = -1.0

        for p in persons:
            centroid_bytes = p.get("centroid")
            if not centroid_bytes:
                continue
            centroid = _bytes_to_emb(centroid_bytes)
            sim = cosine(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = p["id"]

        if best_id is None:
            return None, 0.0
        return best_id, best_sim

    def assign_or_create(
        self,
        embedding: np.ndarray,
        job_id: str | None = None,
        exclude_ids: set[str] = frozenset(),
    ) -> tuple[str, bool]:
        """Nearest centroid among persons NOT in exclude_ids.

        If best_sim >= self.threshold  -> assign to that person (add_sample).
        Else                           -> create a new person and add_sample.

        Returns (person_id, created).
        """
        persons = [
            p for p in db.list_persons(self.db_path) if p["id"] not in exclude_ids
        ]

        emb = np.asarray(embedding, dtype=np.float32)
        best_id: str | None = None
        best_sim: float = -1.0

        for p in persons:
            centroid_bytes = p.get("centroid")
            if not centroid_bytes:
                continue
            centroid = _bytes_to_emb(centroid_bytes)
            sim = cosine(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = p["id"]

        if best_id is not None and best_sim >= self.threshold:
            self.add_sample(best_id, emb, job_id=job_id)
            return best_id, False

        # Create new person
        all_persons = db.list_persons(self.db_path)
        name = f"Person {len(all_persons) + 1}"
        new_id = uuid.uuid4().hex
        db.create_person(
            self.db_path,
            {"id": new_id, "name": name, "n_samples": 0},
        )
        self.add_sample(new_id, emb, job_id=job_id)
        return new_id, True

    def add_sample(
        self,
        person_id: str,
        embedding: np.ndarray,
        job_id: str | None = None,
    ) -> None:
        """Store the embedding row, then recompute the person's centroid as the
        mean of ALL its embeddings. Updates centroid (bytes), n_samples, dim.

        A given job contributes at most one sample per person: re-adding for the
        same (person, job) replaces the previous sample rather than appending,
        so re-diarizing a file doesn't inflate the gallery."""
        emb = np.asarray(embedding, dtype=np.float32)
        if job_id is not None:
            db.delete_job_embeddings(self.db_path, person_id, job_id)
        db.add_person_embedding(
            self.db_path,
            {
                "id": uuid.uuid4().hex,
                "person_id": person_id,
                "job_id": job_id,
                "embedding": _emb_to_bytes(emb),
            },
        )
        self._recompute_centroid(person_id)

    def rename(self, person_id: str, name: str) -> None:
        db.update_person(self.db_path, person_id, name=name)

    def merge(self, src_id: str, dst_id: str) -> None:
        """Move src's embeddings to dst, recompute dst centroid + n_samples,
        then delete src person."""
        db.reassign_embeddings(self.db_path, src_id, dst_id)
        self._recompute_centroid(dst_id)
        db.delete_person(self.db_path, src_id)

    def delete(self, person_id: str) -> None:
        db.delete_person(self.db_path, person_id)

    def list(self) -> list[dict]:
        """Return [{id, name, n_samples, created_at}] — no raw blobs."""
        persons = db.list_persons(self.db_path)
        return [
            {
                "id": p["id"],
                "name": p["name"],
                "n_samples": p["n_samples"],
                "created_at": p["created_at"],
            }
            for p in persons
        ]

    # ── internal helpers ──────────────────────────────────────────────────────

    def _recompute_centroid(self, person_id: str) -> None:
        rows = db.list_person_embeddings(self.db_path, person_id)
        if not rows:
            db.update_person(
                self.db_path, person_id, centroid=None, n_samples=0, dim=None
            )
            return
        vectors = [_bytes_to_emb(r["embedding"]) for r in rows]
        centroid = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        db.update_person(
            self.db_path,
            person_id,
            centroid=_emb_to_bytes(centroid),
            n_samples=len(vectors),
            dim=len(centroid),
        )
