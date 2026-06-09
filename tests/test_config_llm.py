"""LLM tidy settings parse from env with sane defaults.

These tests are hermetic: they neutralise the project's real ``.env`` so the
developer's machine config can't change the outcome.
"""
import pytest

import app.config as config_mod
from app.config import DEFAULT_LLM_FILE, DEFAULT_LLM_REPO, load_settings


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "_load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path))


def test_llm_defaults(monkeypatch):
    for k in ("TRANSCRIBE_TIDY", "TRANSCRIBE_LLM_REPO", "TRANSCRIBE_LLM_FILE",
              "TRANSCRIBE_LLM_MODEL", "TRANSCRIBE_LLM_PORT", "TRANSCRIBE_LLM_CTX",
              "TRANSCRIBE_LLM_HEALTH_TIMEOUT", "TRANSCRIBE_LLM_REQUEST_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)
    s = load_settings()
    assert s.tidy_enabled is True
    # Defaults to the recommended model, resolved into the HF cache on first use.
    assert s.llm_repo == DEFAULT_LLM_REPO
    assert s.llm_file == DEFAULT_LLM_FILE
    assert s.llm_model is None  # no local override
    assert s.llm_port == 8080
    assert s.llm_ctx == 8192
    assert s.llm_health_timeout == 120
    assert s.llm_request_timeout == 120


def test_llm_repo_file_from_env(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_LLM_REPO", "someone/Other-GGUF")
    monkeypatch.setenv("TRANSCRIBE_LLM_FILE", "other-Q4_K_M.gguf")
    monkeypatch.delenv("TRANSCRIBE_LLM_MODEL", raising=False)
    s = load_settings()
    assert s.llm_repo == "someone/Other-GGUF"
    assert s.llm_file == "other-Q4_K_M.gguf"
    assert s.llm_model is None


def test_llm_local_override_passes_through_verbatim(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_LLM_MODEL", "/home/me/Downloads/model.gguf")
    s = load_settings()
    # No project-root rooting: an explicit override is used exactly as given.
    assert s.llm_model == "/home/me/Downloads/model.gguf"


def test_llm_other_overrides_from_env(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_TIDY", "false")
    monkeypatch.setenv("TRANSCRIBE_LLM_PORT", "9001")
    monkeypatch.setenv("TRANSCRIBE_LLM_CTX", "4096")
    monkeypatch.setenv("TRANSCRIBE_LLM_HEALTH_TIMEOUT", "30")
    monkeypatch.setenv("TRANSCRIBE_LLM_REQUEST_TIMEOUT", "45")
    s = load_settings()
    assert s.tidy_enabled is False
    assert s.llm_port == 9001
    assert s.llm_ctx == 4096
    assert s.llm_health_timeout == 30
    assert s.llm_request_timeout == 45
