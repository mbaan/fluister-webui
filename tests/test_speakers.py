"""Tests for app/speakers.py – global voice gallery."""

from __future__ import annotations

import numpy as np
import pytest

from app import db
from app.speakers import Gallery, cosine


# ── helpers ───────────────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    """Return a unit-normalised copy of v."""
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_db(tmp_path):
    db_path = tmp_path / "t.db"
    db.init_db(db_path)
    return db_path


# ── cosine sanity ─────────────────────────────────────────────────────────────

def test_cosine_identical():
    v = unit(np.array([1.0, 2.0, 3.0]))
    assert abs(cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine(a, b)) < 1e-6


def test_cosine_zero_vector():
    a = np.zeros(4, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert cosine(a, b) == 0.0


# ── empty gallery ─────────────────────────────────────────────────────────────

def test_identify_empty_gallery(tmp_path):
    g = Gallery(make_db(tmp_path), threshold=0.45)
    pid, sim = g.identify(unit(np.array([1.0, 0.0, 0.0])))
    assert pid is None
    assert sim == 0.0


def test_assign_or_create_empty_gallery(tmp_path):
    g = Gallery(make_db(tmp_path), threshold=0.45)
    v = unit(np.array([1.0, 0.0, 0.0]))
    pid, created = g.assign_or_create(v)
    assert created is True
    persons = g.list()
    assert len(persons) == 1
    assert persons[0]["name"] == "Person 1"
    assert persons[0]["id"] == pid


# ── near-identical embedding → same person ────────────────────────────────────

def test_assign_near_identical_returns_same_person(tmp_path):
    g = Gallery(make_db(tmp_path), threshold=0.45)
    base = unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    pid1, created1 = g.assign_or_create(base)
    assert created1 is True

    # Add a tiny perturbation — cosine similarity will be very close to 1.0
    noisy = unit(base + np.array([0.0, 1e-4, 0.0], dtype=np.float32))
    pid2, created2 = g.assign_or_create(noisy)

    assert pid2 == pid1
    assert created2 is False
    persons = g.list()
    assert len(persons) == 1
    assert persons[0]["n_samples"] == 2


# ── clearly different embedding → new person ─────────────────────────────────

def test_assign_different_embedding_creates_new_person(tmp_path):
    g = Gallery(make_db(tmp_path), threshold=0.45)
    a = unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    b = unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    # cosine(a, b) == 0.0 which is below threshold 0.45

    pid1, created1 = g.assign_or_create(a)
    pid2, created2 = g.assign_or_create(b)

    assert created1 is True
    assert created2 is True
    assert pid1 != pid2
    assert len(g.list()) == 2


# ── exclude_ids forces new person ────────────────────────────────────────────

def test_exclude_ids_forces_new_person(tmp_path):
    g = Gallery(make_db(tmp_path), threshold=0.45)
    base = unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    pid1, _ = g.assign_or_create(base)

    # Same-ish embedding but the matching person is excluded
    noisy = unit(base + np.array([0.0, 1e-4, 0.0], dtype=np.float32))
    pid2, created = g.assign_or_create(noisy, exclude_ids={pid1})

    assert created is True
    assert pid2 != pid1
    assert len(g.list()) == 2


# ── merge ─────────────────────────────────────────────────────────────────────

def test_merge(tmp_path):
    g = Gallery(make_db(tmp_path), threshold=0.45)
    a = unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    b = unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    pid_a, _ = g.assign_or_create(a)  # 1 sample
    pid_b, _ = g.assign_or_create(b)  # 1 sample
    # Add a second sample to src
    g.add_sample(pid_a, unit(np.array([1.0, 0.1, 0.0], dtype=np.float32)))

    g.merge(pid_a, pid_b)

    persons = g.list()
    ids = {p["id"] for p in persons}
    assert pid_a not in ids
    assert pid_b in ids

    dst = next(p for p in persons if p["id"] == pid_b)
    # dst originally had 1 sample; src had 2 → total 3
    assert dst["n_samples"] == 3

    # Centroid must have changed (originally purely [0,1,0])
    raw = db.get_person(g.db_path, pid_b)
    centroid = np.frombuffer(raw["centroid"], dtype=np.float32)
    # The merged centroid includes x-axis components, so its x component > 0
    assert centroid[0] > 0.0


# ── rename ────────────────────────────────────────────────────────────────────

def test_rename(tmp_path):
    g = Gallery(make_db(tmp_path), threshold=0.45)
    v = unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    pid, _ = g.assign_or_create(v)

    g.rename(pid, "Alice")
    persons = g.list()
    assert persons[0]["name"] == "Alice"


# ── delete ────────────────────────────────────────────────────────────────────

def test_delete_removes_person_and_embeddings(tmp_path):
    g = Gallery(make_db(tmp_path), threshold=0.45)
    v = unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    pid, _ = g.assign_or_create(v)

    g.delete(pid)

    assert g.list() == []
    # Embeddings also gone
    assert db.list_person_embeddings(g.db_path, pid) == []


# ── list() has no blob keys ───────────────────────────────────────────────────

def test_list_has_no_blob_keys(tmp_path):
    g = Gallery(make_db(tmp_path), threshold=0.45)
    v = unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    g.assign_or_create(v)

    for p in g.list():
        assert "centroid" not in p
        assert "embedding" not in p
        assert "dim" not in p
        # Required keys are present
        assert "id" in p
        assert "name" in p
        assert "n_samples" in p
        assert "created_at" in p


def test_add_sample_idempotent_per_job(tmp_path):
    """Re-adding a sample for the same (person, job) replaces, not appends."""
    import numpy as np

    from app import db
    from app.speakers import Gallery

    dbp = tmp_path / "t.db"
    db.init_db(dbp)
    g = Gallery(dbp)

    pid, created = g.assign_or_create(np.array([1, 0, 0], dtype="float32"), job_id="A")
    assert created and g.list()[0]["n_samples"] == 1

    # same job again -> still one sample
    g.add_sample(pid, np.array([1, 0, 0], dtype="float32"), job_id="A")
    assert g.list()[0]["n_samples"] == 1

    # a different job -> two samples
    g.add_sample(pid, np.array([1, 0, 0], dtype="float32"), job_id="B")
    assert g.list()[0]["n_samples"] == 2
