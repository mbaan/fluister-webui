# LLM tidy → readable transcript view — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a best-effort local-LLM post-pass that produces a *readable* (punctuated, paragraphed, filler-free) version of each transcript, shown as a toggleable second view alongside the untouched timestamped/diarized transcript.

**Architecture:** A `LlamaServer` supervisor spawns/stops `llama-server` (Qwen3-30B-A3B, `--cpu-moe`) under fluister's lifecycle. After transcription+diarization the queue persists the job as `done`, then best-effort calls a pure `tidier` module over HTTP and stores the result in a new `tidied_json` column. The frontend gains a Raw/Readable toggle; raw stays the source of truth.

**Tech Stack:** Python 3.12, FastAPI, faster-whisper (unchanged), `llama-server` (llama.cpp), stdlib `urllib`/`subprocess`/`socket`/`ctypes`, pytest + httpx, vanilla JS frontend.

**Spec:** `docs/superpowers/specs/2026-06-09-llm-tidy-readable-transcript-design.md`

**Guardrails (do not violate):** keep faster-whisper `large-v3` and its compute_type untouched; the LLM tidies but never fixes/translates/rephrases; every new GPU/IO path is best-effort (failure must never fail a transcription job).

---

## File structure

- **Create** `app/tidier.py` — pure, HTTP-only tidier: `group_turns`, `tidy_turns`, `chat_completion`. No torch/whisper imports.
- **Create** `app/llm_server.py` — `LlamaServer` subprocess supervisor (spawn/health/stop/orphan-guard/stale-port). No torch/whisper imports.
- **Create** `tests/test_tidier.py`, `tests/test_llm_server.py`, `tests/test_tidy_pipeline.py`.
- **Modify** `app/config.py` — new LLM settings.
- **Modify** `app/db.py` — `tidied_json` column + migration.
- **Modify** `app/queue.py` — own `LlamaServer` lifecycle; best-effort tidy step in `_process`; `_maybe_tidy` helper.
- **Modify** `app/static/app.js`, `app/static/index.html` (none expected), `app/static/style.css` — Raw/Readable toggle + readable renderer.
- **Modify** `.env.example`, `README.md` — document the new env + launch command.

---

## Task 1: Config — LLM settings

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_config_llm.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_llm.py
"""LLM tidy settings parse from env with sane defaults."""
from app.config import load_settings


def test_llm_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    for k in ("TRANSCRIBE_TIDY", "TRANSCRIBE_LLM_MODEL", "TRANSCRIBE_LLM_PORT",
              "TRANSCRIBE_LLM_CTX", "TRANSCRIBE_LLM_HEALTH_TIMEOUT",
              "TRANSCRIBE_LLM_REQUEST_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)
    s = load_settings()
    assert s.tidy_enabled is True
    assert s.llm_model is None
    assert s.llm_port == 8080
    assert s.llm_ctx == 8192
    assert s.llm_health_timeout == 120
    assert s.llm_request_timeout == 120


def test_llm_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRANSCRIBE_TIDY", "false")
    monkeypatch.setenv("TRANSCRIBE_LLM_MODEL", "/models/qwen.gguf")
    monkeypatch.setenv("TRANSCRIBE_LLM_PORT", "9001")
    monkeypatch.setenv("TRANSCRIBE_LLM_CTX", "4096")
    monkeypatch.setenv("TRANSCRIBE_LLM_HEALTH_TIMEOUT", "30")
    monkeypatch.setenv("TRANSCRIBE_LLM_REQUEST_TIMEOUT", "45")
    s = load_settings()
    assert s.tidy_enabled is False
    assert s.llm_model == "/models/qwen.gguf"
    assert s.llm_port == 9001
    assert s.llm_ctx == 4096
    assert s.llm_health_timeout == 30
    assert s.llm_request_timeout == 45
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_llm.py -v`
Expected: FAIL — `Settings` has no `tidy_enabled` (AttributeError / TypeError).

- [ ] **Step 3: Add fields to `Settings` and `load_settings`**

In `app/config.py`, add to the `Settings` dataclass (after `hf_token`):

```python
    # LLM tidy / readability post-pass
    tidy_enabled: bool
    llm_model: str | None
    llm_port: int
    llm_ctx: int
    llm_health_timeout: int
    llm_request_timeout: int
```

In `load_settings()`, add to the `Settings(...)` call (after `hf_token=...`):

```python
        tidy_enabled=_env_bool("TRANSCRIBE_TIDY", True),
        llm_model=os.environ.get("TRANSCRIBE_LLM_MODEL") or None,
        llm_port=int(os.environ.get("TRANSCRIBE_LLM_PORT", "8080")),
        llm_ctx=int(os.environ.get("TRANSCRIBE_LLM_CTX", "8192")),
        llm_health_timeout=int(os.environ.get("TRANSCRIBE_LLM_HEALTH_TIMEOUT", "120")),
        llm_request_timeout=int(os.environ.get("TRANSCRIBE_LLM_REQUEST_TIMEOUT", "120")),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_llm.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Run the full suite (nothing else broke)**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/config.py tests/test_config_llm.py
git commit -m "tidy: add LLM post-pass settings"
```

---

## Task 2: DB — `tidied_json` column + migration

**Files:**
- Modify: `app/db.py` (`_SCHEMA`, `_migrate`, `create_job` column list is unaffected — tidied is set via `update_job`)
- Test: `tests/test_tidied_column.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tidied_column.py
"""tidied_json column exists, migrates onto an old DB, and round-trips."""
import sqlite3

from app import db


def test_fresh_db_has_tidied_json(tmp_path):
    p = tmp_path / "t.db"
    db.init_db(p)
    job = db.create_job(p, {
        "id": "j1", "original_filename": "a.m4a", "stored_path": "/x/a.m4a",
        "language": "auto", "status": db.STATUS_QUEUED, "model_name": "large-v3",
    })
    assert "tidied_json" in job
    assert job["tidied_json"] is None
    db.update_job(p, "j1", tidied_json='[{"speaker": null, "text": "Hi."}]')
    assert db.get_job(p, "j1")["tidied_json"] == '[{"speaker": null, "text": "Hi."}]'


def test_migration_adds_column_to_old_db(tmp_path):
    p = tmp_path / "old.db"
    # Minimal pre-tidied jobs table (no tidied_json).
    conn = sqlite3.connect(p)
    conn.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, original_filename TEXT NOT NULL, "
        "stored_path TEXT NOT NULL, language TEXT NOT NULL DEFAULT 'auto', "
        "status TEXT NOT NULL, model_name TEXT NOT NULL, created_at TEXT NOT NULL, "
        "progress REAL NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO jobs (id, original_filename, stored_path, status, model_name, created_at) "
        "VALUES ('old1', 'o.m4a', '/x/o.m4a', 'done', 'large-v3', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    db.init_db(p)  # runs _migrate
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    assert "tidied_json" in cols
    db.update_job(p, "old1", tidied_json="[]")
    assert db.get_job(p, "old1")["tidied_json"] == "[]"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tidied_column.py -v`
Expected: FAIL — `tidied_json` not in job dict / not in PRAGMA columns.

- [ ] **Step 3: Add column to schema + migration**

In `app/db.py`, add to `_SCHEMA`'s `jobs` table (after `segments_json TEXT`):

```python
    segments_json        TEXT,
    tidied_json          TEXT
```

(Note: remove the trailing comma issue — `tidied_json TEXT` becomes the new last column before the closing `)`.)

In `_migrate(conn)`, after the `segments_json` block:

```python
    if "tidied_json" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN tidied_json TEXT")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tidied_column.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/db.py tests/test_tidied_column.py
git commit -m "tidy: add tidied_json column + migration"
```

---

## Task 3: Tidier — `group_turns`

**Files:**
- Create: `app/tidier.py`
- Test: `tests/test_tidier.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tidier.py
from app.models import Segment
from app.tidier import Turn, group_turns


def _seg(text, speaker=None):
    return Segment(start=0.0, end=1.0, text=text, speaker=speaker)


def test_single_speaker_is_one_turn():
    segs = [_seg("hello"), _seg("there"), _seg("friend")]
    turns = group_turns(segs)
    assert turns == [Turn(speaker=None, text="hello there friend")]


def test_speaker_change_splits():
    segs = [_seg("hi", "Ann"), _seg("yo", "Ann"), _seg("hello", "Bob")]
    turns = group_turns(segs)
    assert turns == [Turn("Ann", "hi yo"), Turn("Bob", "hello")]


def test_blank_segments_skipped():
    segs = [_seg("  "), _seg("real"), _seg("")]
    assert group_turns(segs) == [Turn(None, "real")]


def test_long_turn_splits_at_segment_boundary():
    segs = [_seg("a" * 30, "Ann"), _seg("b" * 30, "Ann"), _seg("c" * 30, "Ann")]
    turns = group_turns(segs, max_chars=50)
    # Same speaker, but chunked so no chunk exceeds the budget materially.
    assert len(turns) == 3
    assert all(t.speaker == "Ann" for t in turns)
    assert [t.text for t in turns] == ["a" * 30, "b" * 30, "c" * 30]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tidier.py -v`
Expected: FAIL — `app.tidier` does not exist.

- [ ] **Step 3: Implement `Turn` + `group_turns`**

```python
# app/tidier.py
"""Readability tidier: turns a raw transcript into punctuated, paragraphed,
filler-free text via a local llama-server. Pure + HTTP-only — no torch/whisper
imports, so it stays cheap to import and easy to test.

Contract: the LLM *tidies, never fixes* — it must not change, add, translate,
reorder, or "correct" meaningful words; it only punctuates, paragraphs, and
drops fillers. See docs/superpowers/specs/2026-06-09-llm-tidy-readable-transcript-design.md.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    """A contiguous block of one speaker's text to tidy in isolation."""
    speaker: str | None
    text: str


def group_turns(segments: Iterable, max_chars: int = 4000) -> list[Turn]:
    """Merge consecutive same-speaker segments into turns, splitting a turn at a
    segment boundary when it would exceed ``max_chars`` (keeps prompts well inside
    the model context). Blank segments are skipped. ``segments`` are objects with
    ``.text`` and ``.speaker`` (e.g. ``app.models.Segment``)."""
    turns: list[Turn] = []
    cur_speaker = None
    cur_parts: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur_parts, cur_len
        if cur_parts:
            turns.append(Turn(speaker=cur_speaker, text=" ".join(cur_parts)))
        cur_parts = []
        cur_len = 0

    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        speaker = getattr(seg, "speaker", None)
        new_speaker = speaker != cur_speaker
        too_long = cur_parts and (cur_len + 1 + len(text)) > max_chars
        if new_speaker or too_long:
            flush()
            cur_speaker = speaker
        cur_parts.append(text)
        cur_len += (1 if cur_len else 0) + len(text)
    flush()
    return turns
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tidier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/tidier.py tests/test_tidier.py
git commit -m "tidy: group transcript segments into speaker turns"
```

---

## Task 4: Tidier — `tidy_turns` + `chat_completion`

**Files:**
- Modify: `app/tidier.py`
- Test: `tests/test_tidier.py` (extend)

- [ ] **Step 1: Write the failing test (append to `tests/test_tidier.py`)**

```python
import app.tidier as tidier_mod
from app.tidier import tidy_turns, SYSTEM_PROMPT


def test_tidy_turns_builds_prompt_and_parses(monkeypatch):
    captured = []

    def fake_chat(base_url, messages, *, model, temperature, timeout):
        captured.append((base_url, messages, temperature, timeout))
        return "Cleaned: " + messages[-1]["content"]

    monkeypatch.setattr(tidier_mod, "chat_completion", fake_chat)
    turns = [Turn("Ann", "um hello"), Turn(None, "like yeah")]
    out = tidy_turns(turns, "http://x:8080", timeout=42)

    assert out == [
        {"speaker": "Ann", "text": "Cleaned: um hello"},
        {"speaker": None, "text": "Cleaned: like yeah"},
    ]
    # System prompt is sent and is the tidier-not-fixer contract.
    assert captured[0][1][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert captured[0][1][1]["content"] == "um hello"
    assert captured[0][3] == 42  # request timeout threaded through


def test_tidy_turns_falls_back_to_raw_on_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("server down")

    monkeypatch.setattr(tidier_mod, "chat_completion", boom)
    out = tidy_turns([Turn(None, "raw text")], "http://x:8080", timeout=5)
    assert out == [{"speaker": None, "text": "raw text"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tidier.py -v`
Expected: FAIL — `tidy_turns` / `SYSTEM_PROMPT` / `chat_completion` not defined.

- [ ] **Step 3: Implement (append to `app/tidier.py`)**

```python
SYSTEM_PROMPT = (
    "You clean up speech-to-text transcripts for readability. Add punctuation, "
    "capitalization, and paragraph breaks. Remove filler words (uh, um, like, you "
    "know) and false starts / repeated restarts. Do NOT change, add, remove (other "
    "than fillers), reorder, translate, or 'correct' any meaningful word. Preserve "
    "the original language(s) exactly, including Dutch-English code-switching. "
    "Output only the cleaned text, nothing else."
)


def chat_completion(
    base_url: str,
    messages: list[dict],
    *,
    model: str = "local",
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    """POST an OpenAI-style chat completion to llama-server; return the content.
    Raises on transport/HTTP/parse errors (callers handle best-effort)."""
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def tidy_turns(
    turns: list[Turn],
    base_url: str,
    *,
    model: str = "local",
    temperature: float = 0.1,
    timeout: int = 120,
) -> list[dict]:
    """Tidy each turn independently. On a per-turn failure, fall back to that
    turn's raw text (degrade gracefully — never drop content). Returns a list of
    ``{"speaker": str | None, "text": str}`` paragraphs."""
    out: list[dict] = []
    for turn in turns:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": turn.text},
        ]
        try:
            text = chat_completion(
                base_url, messages, model=model, temperature=temperature, timeout=timeout
            )
        except Exception:  # noqa: BLE001
            logger.warning("Tidy failed for a turn; keeping raw text", exc_info=True)
            text = turn.text
        out.append({"speaker": turn.speaker, "text": text})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tidier.py -v`
Expected: PASS (all four/six tests).

- [ ] **Step 5: Commit**

```bash
git add app/tidier.py tests/test_tidier.py
git commit -m "tidy: call llama-server per turn, fall back to raw on error"
```

---

## Task 5: `LlamaServer` supervisor

**Files:**
- Create: `app/llm_server.py`
- Test: `tests/test_llm_server.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_server.py
import signal

import app.llm_server as mod
from app.llm_server import LlamaServer


def _srv(tmp_path, **kw):
    model = kw.pop("model", None)
    return LlamaServer(
        enabled=kw.pop("enabled", True),
        model_path=model,
        port=kw.pop("port", 8080),
        ctx=kw.pop("ctx", 8192),
        health_timeout=kw.pop("health_timeout", 1),
    )


def test_build_command_has_coresidence_flags(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cmd = _srv(tmp_path, model=str(gguf), port=9000, ctx=4096).build_command()
    assert cmd[0].endswith("llama-server")
    assert "-m" in cmd and str(gguf) in cmd
    assert "--cpu-moe" in cmd
    assert cmd[cmd.index("--port") + 1] == "9000"
    assert cmd[cmd.index("-c") + 1] == "4096"
    assert cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert cmd[cmd.index("-ctv") + 1] == "q8_0"


def test_disabled_never_spawns(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    s = _srv(tmp_path, enabled=False)
    s.start()
    assert s.available is False


def test_missing_model_never_spawns(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    s = _srv(tmp_path, model=str(tmp_path / "nope.gguf"))
    s.start()
    assert s.available is False


def test_stale_port_never_spawns(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"; gguf.write_bytes(b"x")
    monkeypatch.setattr(mod, "_port_in_use", lambda host, port: True)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    s = _srv(tmp_path, model=str(gguf))
    s.start()
    assert s.available is False


class _FakeProc:
    def __init__(self):
        self.pid = 4321
        self.signals = []
        self._alive = True
    def poll(self):
        return None if self._alive else 0
    def wait(self, timeout=None):
        self._alive = False
        return 0


def test_stop_signals_process_group(tmp_path, monkeypatch):
    s = _srv(tmp_path)
    s._proc = _FakeProc()
    s.available = True
    killed = []
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    s.stop()
    assert (4321, signal.SIGTERM) in killed
    assert s.available is False
    assert s._proc is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_server.py -v`
Expected: FAIL — `app.llm_server` does not exist.

- [ ] **Step 3: Implement `app/llm_server.py`**

```python
# app/llm_server.py
"""Supervise the local llama-server used for the readability tidy pass.

fluister owns the full lifecycle: it spawns llama-server on startup and stops it
on shutdown. A kernel-level orphan backstop (PR_SET_PDEATHSIG, Linux) guarantees
the multi-GB child is reaped even if fluister dies abnormally. Everything is
best-effort: any failure sets ``available=False`` and logs — transcription is
never affected.
"""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_PR_SET_PDEATHSIG = 1  # <sys/prctl.h>


def _set_pdeathsig() -> None:
    """preexec_fn: ask the kernel to SIGKILL this child if the parent dies."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:  # noqa: BLE001 — non-Linux or no libc; clean shutdown still covers it
        pass


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


class LlamaServer:
    def __init__(
        self,
        *,
        enabled: bool,
        model_path: str | None,
        port: int,
        ctx: int,
        health_timeout: int,
        host: str = "127.0.0.1",
        binary: str = "llama-server",
    ) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self.port = port
        self.ctx = ctx
        self.health_timeout = health_timeout
        self.host = host
        self.binary = binary
        self.available = False
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def build_command(self) -> list[str]:
        return [
            self.binary,
            "-m", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "--cpu-moe",
            "-ngl", "99",
            "-c", str(self.ctx),
            "-ctk", "q8_0",
            "-ctv", "q8_0",
        ]

    def start(self) -> None:
        """Spawn llama-server and wait for /health. Best-effort: sets
        ``available`` and never raises."""
        if not self.enabled:
            logger.info("Tidy disabled (TRANSCRIBE_TIDY) — no LLM started.")
            return
        if not self.model_path or not os.path.isfile(self.model_path):
            logger.warning(
                "Tidy LLM model not found (TRANSCRIBE_LLM_MODEL=%r) — readable view disabled.",
                self.model_path,
            )
            return
        if _port_in_use(self.host, self.port):
            logger.warning(
                "Port %s already in use — not starting a second llama-server. "
                "Resolve the stale process and restart.", self.port,
            )
            return
        try:
            self._proc = subprocess.Popen(
                self.build_command(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,   # own process group
                preexec_fn=_set_pdeathsig,  # orphan backstop (Linux)
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to spawn llama-server — readable view disabled.")
            self._proc = None
            return
        if self._wait_health():
            self.available = True
            logger.info("llama-server ready on %s — readable view enabled.", self.base_url)
        else:
            logger.warning("llama-server did not become healthy in %ss — disabling tidy.",
                           self.health_timeout)
            self.stop()

    def _wait_health(self) -> bool:
        deadline = time.monotonic() + self.health_timeout
        url = self.base_url + "/health"
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False  # process exited early
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except Exception:  # noqa: BLE001 — not up yet
                pass
            time.sleep(1.0)
        return False

    def stop(self) -> None:
        """SIGTERM the process group, grace, then SIGKILL. Always safe to call."""
        self.available = False
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:  # noqa: BLE001 — already gone
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 — still alive after grace
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("Error stopping llama-server")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_server.py -v`
Expected: PASS (all tests). The `_FakeProc.wait` flips `poll()` so no SIGKILL is needed; SIGTERM is asserted.

- [ ] **Step 5: Commit**

```bash
git add app/llm_server.py tests/test_llm_server.py
git commit -m "tidy: llama-server supervisor with orphan-safe teardown"
```

---

## Task 6: Queue integration — own lifecycle + best-effort tidy

**Files:**
- Modify: `app/queue.py`
- Test: `tests/test_tidy_pipeline.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tidy_pipeline.py
"""Best-effort tidy step: populated when LLM is up, skipped/safe when not."""
import os
import tempfile

os.environ.setdefault("TRANSCRIBE_DATA_DIR", tempfile.mkdtemp(prefix="fluister-tidy-"))

import pytest

import app.queue as qmod
from app import db
from app.config import load_settings
from app.models import Segment
from app.queue import JobQueue


class _FakeLLM:
    def __init__(self, available):
        self.available = available
        self.base_url = "http://x:8080"
        self.started = False
        self.stopped = False
    def start(self):
        self.started = True
    def stop(self):
        self.stopped = True


def _queue(settings, llm):
    q = JobQueue(settings)
    q.llm_server = llm
    return q


def test_maybe_tidy_populates_when_available(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    monkeypatch.setattr(
        qmod, "tidy_turns",
        lambda turns, base_url, timeout: [{"speaker": t.speaker, "text": t.text.upper()} for t in turns],
    )
    q = _queue(settings, _FakeLLM(available=True))
    segs = [Segment(0, 1, "hi", "Ann"), Segment(1, 2, "yo", "Ann")]
    out = q._maybe_tidy("job1", segs)
    assert out == [{"speaker": "Ann", "text": "HI YO"}]


def test_maybe_tidy_skips_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    called = []
    monkeypatch.setattr(qmod, "tidy_turns", lambda *a, **k: called.append(1) or [])
    q = _queue(settings, _FakeLLM(available=False))
    assert q._maybe_tidy("job1", [Segment(0, 1, "hi")]) is None
    assert called == []


def test_maybe_tidy_best_effort_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(qmod, "tidy_turns", boom)
    q = _queue(settings, _FakeLLM(available=True))
    assert q._maybe_tidy("job1", [Segment(0, 1, "hi")]) is None  # swallowed


@pytest.mark.anyio
async def test_stop_stops_llm_server(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))
    settings = load_settings()
    llm = _FakeLLM(available=True)
    q = _queue(settings, llm)
    await q.stop()
    assert llm.stopped is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tidy_pipeline.py -v`
Expected: FAIL — `JobQueue` has no `llm_server` / `_maybe_tidy`; `qmod.tidy_turns` missing.

- [ ] **Step 3: Wire `LlamaServer` + `_maybe_tidy` into `app/queue.py`**

Add imports near the top:

```python
from app.llm_server import LlamaServer
from app.tidier import group_turns, tidy_turns
```

In `JobQueue.__init__`, after `self.diarizer: Any | None = None`:

```python
        self.llm_server: Any = self._default_llm_server()
```

Add a factory method (next to `_default_factory`):

```python
    def _default_llm_server(self) -> Any:
        s = self.settings
        return LlamaServer(
            enabled=s.tidy_enabled,
            model_path=s.llm_model,
            port=s.llm_port,
            ctx=s.llm_ctx,
            health_timeout=s.llm_health_timeout,
        )
```

In `_load_model`, after the diarizer try/except block and before `self._model_ready.set()` (i.e. inside the `finally:` is wrong — put it just before the `finally:`), start the LLM best-effort:

```python
        # LLM tidier is optional — start it best-effort; failure just disables
        # the readable view.
        try:
            await asyncio.to_thread(self.llm_server.start)
        except Exception:  # noqa: BLE001
            logger.exception("llama-server start failed — readable view disabled.")
```

In `stop()`, stop the LLM (do it first so VRAM frees promptly):

```python
    async def stop(self) -> None:
        if self.llm_server is not None:
            self.llm_server.stop()
        for task in (self._worker_task, self._load_task):
            ...
```

Add the `_maybe_tidy` helper (next to `_diarize_and_identify`):

```python
    def _maybe_tidy(self, job_id: str, segments) -> list[dict] | None:
        """Best-effort readable tidy. Returns paragraphs or None (LLM down / error)."""
        if not (self.llm_server is not None and self.llm_server.available):
            return None
        try:
            turns = group_turns(segments)
            if not turns:
                return None
            return tidy_turns(
                turns, self.llm_server.base_url, timeout=self.settings.llm_request_timeout
            )
        except Exception:  # noqa: BLE001
            logger.exception("Tidy pass failed for job %s", job_id)
            return None
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `uv run pytest tests/test_tidy_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Integrate into `_process` (persist DONE first, then tidy, then publish done)**

In `_process`, replace the final persist+publish (the block that does
`db.update_job(... status=db.STATUS_DONE ...)` then `self.publish(job_id, "done", ...)`)
with:

```python
            # 3. Persist results (segments + speakers stored in the DB). Mark DONE
            # now so a restart can never strand the job mid-tidy.
            segments_payload = assign.attach_words_to_segments(segments, words)
            transcript_text = "\n".join(s.text for s in segments if s.text)
            db.update_job(
                db_path, job_id, status=db.STATUS_DONE,
                detected_language=info.language, duration=info.duration,
                progress=1.0, transcript_text=transcript_text,
                segments_json=json.dumps(segments_payload, ensure_ascii=False),
                diarized=1 if diarized else 0,
                speakers=json.dumps(speakers_map) if speakers_map else None,
                finished_at=db.now_iso(),
            )

            # 4. Best-effort readable tidy. The job is already DONE; we only delay
            # the *live* `done` event so a watching client keeps its stream open
            # until the readable view arrives.
            if self.llm_server is not None and self.llm_server.available:
                self.publish(job_id, "status", {
                    "status": "tidying", "progress": 1.0,
                    "detected_language": info.language,
                })
                tidied = await asyncio.to_thread(self._maybe_tidy, job_id, segments)
                if tidied is not None:
                    db.update_job(
                        db_path, job_id,
                        tidied_json=json.dumps(tidied, ensure_ascii=False),
                    )
                    self.publish(job_id, "tidied", {"tidied": tidied})

            self.publish(job_id, "done", db.get_job(db_path, job_id))
```

- [ ] **Step 6: Run the full suite (existing API flow must still pass — tidy is off in tests since `TRANSCRIBE_LLM_MODEL` is unset)**

Run: `uv run pytest -q`
Expected: all pass. (In `tests/test_api.py` the LLM is unavailable, so jobs reach `done` with `tidied_json` NULL — unchanged behavior.)

- [ ] **Step 7: Commit**

```bash
git add app/queue.py tests/test_tidy_pipeline.py
git commit -m "tidy: best-effort readable pass in the job pipeline; own llama-server lifecycle"
```

---

## Task 7: Frontend — Raw/Readable toggle (manual-verified; no JS test harness)

**Files:**
- Modify: `app/static/app.js`, `app/static/style.css`

> No automated JS tests exist in this repo; verify in the browser per the steps below.

- [ ] **Step 1: Add helpers to `app/static/app.js`**

Add near the other transcript builders (after `buildDiarizedTranscript`):

```javascript
  // Parse a job's stored readable paragraphs. Returns [{speaker, text}] or null.
  function parseTidied(job) {
    if (!job || !job.tidied_json) return null;
    try {
      const arr = JSON.parse(job.tidied_json);
      return Array.isArray(arr) && arr.length ? arr : null;
    } catch (e) {
      return null;
    }
  }

  // Render the LLM-tidied transcript: speaker-labeled blocks of paragraphs, no
  // per-word timestamps / click-to-play (raw view owns that). Consecutive
  // same-speaker entries share one name chip.
  function buildReadableTranscript(tidied) {
    const t = el("div", { class: "transcript transcript--readable" });
    let lastKey = null;
    let block = null;
    for (const item of tidied) {
      if (!item || typeof item.text !== "string") continue;
      const name = item.speaker || null;
      const key = name || "__none";
      if (key !== lastKey) {
        block = el("div", { class: "turn" });
        if (name) {
          const chip = el("span", { class: "chip", text: name });
          chip.style.setProperty("--chip-h", String(hueFor(key)));
          block.appendChild(chip);
        }
        block.appendChild(el("div", { class: "turn__text" }));
        t.appendChild(block);
        lastKey = key;
      }
      const textNode = block.lastChild;
      for (const para of item.text.split(/\n{2,}|\n/)) {
        const p = para.trim();
        if (p) textNode.appendChild(el("p", { class: "para", text: p }));
      }
    }
    if (!t.children.length) {
      t.appendChild(el("span", { class: "placeholder", text: "(empty transcript)" }));
    }
    return t;
  }

  // Two-button toggle that shows exactly one of [readableNode, rawGroup].
  function buildViewToggle(readableNode, rawGroup) {
    const bar = el("div", { class: "view-toggle" });
    const rBtn = el("button", { class: "view-toggle__btn is-active", text: "Readable" });
    const wBtn = el("button", { class: "view-toggle__btn", text: "Raw" });
    const show = (readable) => {
      readableNode.hidden = !readable;
      rawGroup.hidden = readable;
      rBtn.classList.toggle("is-active", readable);
      wBtn.classList.toggle("is-active", !readable);
    };
    rBtn.addEventListener("click", () => show(true));
    wBtn.addEventListener("click", () => show(false));
    bar.appendChild(rBtn);
    bar.appendChild(wBtn);
    return bar;
  }
```

- [ ] **Step 2: Restructure the `done` branch of `renderBody`**

Replace the `if (status === "done") { ... }` block (the one that builds `t`, caches/wires words, appends the player bar and actions) with:

```javascript
    if (status === "done") {
      const cached = jobJson.get(job.id);
      const data = cached ? cached.data : null;
      const hasSegments = data && Array.isArray(data.segments) && data.segments.length > 0;
      let rawNode;
      if (data && hasSpeakers(data)) {
        rawNode = buildDiarizedTranscript(data);
      } else if (hasSegments) {
        rawNode = buildWordTranscript(data);
      } else {
        const text = job.transcript_text || "";
        rawNode = el("div", { class: "transcript" });
        if (text.trim()) rawNode.textContent = text;
        else rawNode.appendChild(el("span", { class: "placeholder", text: "(empty transcript)" }));
      }
      cacheWordSpans(u, rawNode);
      wireWordClicks(card, job.id, rawNode);
      const player = buildPlayerBar(card, job.id, u);

      const tidied = parseTidied(job);
      if (tidied) {
        const readableNode = buildReadableTranscript(tidied);
        const rawGroup = el("div", { class: "raw-group" });
        rawGroup.appendChild(rawNode);
        rawGroup.appendChild(player);
        rawGroup.hidden = true;            // default to Readable
        body.appendChild(buildViewToggle(readableNode, rawGroup));
        body.appendChild(readableNode);
        body.appendChild(rawGroup);
      } else {
        body.appendChild(rawNode);
        body.appendChild(player);
      }
      body.appendChild(buildActions(job));
    } else if (u.streaming || ACTIVE.has(status)) {
```

(Everything after `else if (u.streaming || ACTIVE.has(status))` stays unchanged.)

- [ ] **Step 3: Handle the `tidied` event + `tidying` status in the SSE block**

In `startStream`, add a handler (after the `segment` listener):

```javascript
    es.addEventListener("tidied", (e) => {
      const data = parseEvent(e);
      if (!data || !Array.isArray(data.tidied)) return;
      const cur = jobs.get(id);
      if (cur) cur.tidied_json = JSON.stringify(data.tidied);
    });
```

The existing `status` listener already stores `u.liveStatus = "tidying"`; the existing `done` listener already replaces the job with the server payload (which now carries `tidied_json`) and re-renders. No further change needed there.

- [ ] **Step 4: Add a label for the transient `tidying` status**

Find the status→label map used by `buildStatLine` (search `app.js` for `"transcribing"`). Add a `tidying` entry, e.g. `tidying: "Polishing…"`, so the live statline reads sensibly during the tidy window. (If statuses are formatted ad-hoc rather than via a map, add a `case "tidying"` returning `"Polishing…"`.)

- [ ] **Step 5: Add CSS to `app/static/style.css`**

```css
/* Raw/Readable toggle */
.view-toggle { display: flex; gap: 4px; margin: 0 0 10px; }
.view-toggle__btn {
  font: inherit; padding: 3px 10px; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--border, #d0d0d0); background: transparent; color: inherit;
}
.view-toggle__btn.is-active {
  background: var(--accent, #2d6cdf); border-color: var(--accent, #2d6cdf); color: #fff;
}
.transcript--readable .turn { margin-bottom: 14px; }
.transcript--readable .para { margin: 0 0 8px; line-height: 1.55; }
```

(Use the project's existing accent/border variables if they differ — match `style.css`.)

- [ ] **Step 6: Manual verification in the browser**

With `llama-server` running and a real note transcribed:
1. Open the note → it defaults to **Readable** (punctuated, paragraphed, no "uhm").
2. Click **Raw** → the original timestamped/diarized transcript shows; click a word → audio seeks (click-to-play intact).
3. Click **Readable** again → toggles back.
4. With the LLM stopped, transcribe another note → it completes and shows only Raw (no toggle), no errors in console/server log.

- [ ] **Step 7: Commit**

```bash
git add app/static/app.js app/static/style.css
git commit -m "tidy: Raw/Readable transcript toggle in the web UI"
```

---

## Task 8: Docs — `.env.example` + README

**Files:**
- Modify: `.env.example`, `README.md`

- [ ] **Step 1: Document the env in `.env.example`**

Append:

```bash
# ── Readability tidy pass (optional local LLM) ───────────────────────────────
# fluister spawns/stops llama-server itself; point it at a GGUF to enable.
TRANSCRIBE_TIDY=true
TRANSCRIBE_LLM_MODEL=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf
TRANSCRIBE_LLM_PORT=8080
TRANSCRIBE_LLM_CTX=8192
TRANSCRIBE_LLM_HEALTH_TIMEOUT=120
TRANSCRIBE_LLM_REQUEST_TIMEOUT=120
```

- [ ] **Step 2: Add a README section**

Add a short "Readable transcripts" section explaining: it's a best-effort local-LLM tidy pass (punctuation/paragraphs/filler removal, not a fixer), fluister owns the llama-server process, set `TRANSCRIBE_LLM_MODEL` to enable, and note the VRAM co-residence flags it launches (`--cpu-moe -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0`).

- [ ] **Step 3: Commit**

```bash
git add .env.example README.md
git commit -m "tidy: document readability LLM env + behavior"
```

---

## Task 9: Manual verification (hard requirements — run once the model is present)

These cannot be unit-tested; run them and record results.

- [ ] **VRAM co-residence:** start fluister with `TRANSCRIBE_LLM_MODEL` set, transcribe a note, and run `nvidia-smi` during the tidy step. Confirm `llama-server` + whisper + diarizer stay resident together **under 10 GB**. If over budget, lower the quant or `-ngl` in `build_command` — **never** quantize whisper.
- [ ] **Orphan backstop:** while llama-server is running, `kill -9 <fluister-pid>`; confirm with `pgrep -af llama-server` (and `nvidia-smi`) that the child is gone within ~1s.
- [ ] **Clean shutdown:** stop fluister normally (Ctrl-C); confirm `pgrep llama-server` is empty.
- [ ] **Readability + audit:** confirm the readable view drops fillers/reads cleanly and Dutch⇄English is preserved; toggle to Raw and confirm the original is untouched with click-to-play working.
- [ ] **Best-effort degrade:** unset `TRANSCRIBE_LLM_MODEL`, restart, transcribe — job completes, only Raw view, no errors.

---

## Self-review notes

- **Spec coverage:** §1 LlamaServer → Task 5; §2 tidier → Tasks 3–4; §3 pipeline → Task 6; §4 db → Task 2; §5 config → Task 1; §6 frontend → Task 7; §7 testing → embedded per task; §8 manual verification → Task 9; docs → Task 8.
- **Best-effort everywhere:** `_maybe_tidy`, `LlamaServer.start`, and the `_process` tidy block all swallow errors and leave the job `done`.
- **No guardrail violations:** whisper model/compute_type untouched; tidier prompt forbids changing meaningful words.
- **Type consistency:** `Turn(speaker, text)` (Task 3) is consumed by `tidy_turns` (Task 4) and `group_turns`/`tidy_turns` are imported into `queue.py` (Task 6); `tidied_json` shape `[{speaker,text}]` is produced by `tidy_turns`, stored in Task 2's column, and parsed by `parseTidied` (Task 7).
