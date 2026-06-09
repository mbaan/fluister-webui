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
