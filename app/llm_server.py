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
        port: int,
        ctx: int,
        health_timeout: int,
        repo: str | None = None,
        file: str | None = None,
        token: str | None = None,
        model_path: str | None = None,
        host: str = "127.0.0.1",
        binary: str = "llama-server",
    ) -> None:
        self.enabled = enabled
        self.repo = repo
        self.file = file
        self.token = token
        # Explicit local override, or the path resolved from the HF cache at start().
        self.model_path = model_path
        self.port = port
        self.ctx = ctx
        self.health_timeout = health_timeout
        self.host = host
        self.binary = binary
        self.available = False
        self._proc: subprocess.Popen | None = None

    def _resolve_model(self) -> str | None:
        """Resolve the GGUF path the same way whisper/pyannote resolve theirs.

        An explicit local override (``model_path``) wins; otherwise download/locate
        ``file`` from ``repo`` via the HF cache (``~/.cache/huggingface``), so it
        shows up in ``hf cache ls`` and is cleanable with ``hf cache rm``. Returns
        a path or ``None``; HF/network errors propagate to ``start()`` (best-effort)."""
        if self.model_path:
            if os.path.isfile(self.model_path):
                return self.model_path
            logger.warning(
                "TRANSCRIBE_LLM_MODEL set but file not found: %s", self.model_path
            )
            return None
        if self.repo and self.file:
            from huggingface_hub import hf_hub_download  # same machinery as faster-whisper

            return hf_hub_download(self.repo, self.file, token=self.token)
        return None

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
        try:
            path = self._resolve_model()
        except Exception:  # noqa: BLE001
            logger.exception("Could not resolve tidy LLM model — readable view disabled.")
            return
        if not path:
            logger.warning(
                "No tidy LLM model configured — set TRANSCRIBE_LLM_REPO + TRANSCRIBE_LLM_FILE "
                "(or TRANSCRIBE_LLM_MODEL for a local file). Readable view disabled."
            )
            return
        self.model_path = path
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
                start_new_session=True,      # own process group
                preexec_fn=_set_pdeathsig,   # orphan backstop (Linux)
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to spawn llama-server — readable view disabled.")
            self._proc = None
            return
        if self._wait_health():
            self.available = True
            logger.info("llama-server ready on %s — readable view enabled.", self.base_url)
        else:
            logger.warning(
                "llama-server did not become healthy in %ss — disabling tidy.",
                self.health_timeout,
            )
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
