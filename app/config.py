"""Runtime configuration, overridable via environment variables.

All settings are read once via :func:`load_settings`. Paths default to a
``data/`` directory next to the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default tidy LLM — resolved into the HF cache on first use (like large-v3).
# Non-thinking Instruct variant so it never emits <think> blocks into the output.
DEFAULT_LLM_REPO = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
DEFAULT_LLM_FILE = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: ``KEY=VALUE`` lines into ``os.environ`` without
    overriding existing real env vars. Ignores blanks and ``#`` comments and
    strips optional surrounding quotes."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    uploads_dir: Path
    db_path: Path
    model_name: str
    device: str  # "auto" | "cuda" | "cpu"
    compute_type: str  # "auto" | "float16" | "int8" | "int8_float16"
    batch_size: int
    use_vad: bool
    default_language: str  # "auto" | "nl" | "en"
    max_upload_mb: int
    # Diarization / speaker recognition
    diarize: bool
    diarize_model: str
    speaker_threshold: float
    min_speaker_seconds: float
    hf_token: str | None
    # LLM tidy / readability post-pass
    tidy_enabled: bool
    llm_repo: str | None     # HF repo id, resolved into the HF cache (like whisper/pyannote)
    llm_file: str | None     # GGUF filename within that repo
    llm_model: str | None    # optional explicit local-path override (skips HF resolution)
    llm_port: int
    llm_ctx: int
    llm_health_timeout: int
    llm_request_timeout: int


def load_settings() -> Settings:
    _load_dotenv(PROJECT_ROOT / ".env")
    data_dir = Path(os.environ.get("TRANSCRIBE_DATA_DIR", PROJECT_ROOT / "data"))
    settings = Settings(
        host=os.environ.get("TRANSCRIBE_HOST", "127.0.0.1"),
        port=int(os.environ.get("TRANSCRIBE_PORT", "8000")),
        data_dir=data_dir,
        uploads_dir=data_dir / "uploads",
        db_path=data_dir / "transcribe.db",
        model_name=os.environ.get("TRANSCRIBE_MODEL", "large-v3"),
        device=os.environ.get("TRANSCRIBE_DEVICE", "auto"),
        compute_type=os.environ.get("TRANSCRIBE_COMPUTE_TYPE", "auto"),
        batch_size=int(os.environ.get("TRANSCRIBE_BATCH_SIZE", "8")),
        use_vad=_env_bool("TRANSCRIBE_VAD", True),
        default_language=os.environ.get("TRANSCRIBE_LANGUAGE", "auto"),
        max_upload_mb=int(os.environ.get("TRANSCRIBE_MAX_UPLOAD_MB", "2048")),
        diarize=_env_bool("TRANSCRIBE_DIARIZE", True),
        diarize_model=os.environ.get(
            "TRANSCRIBE_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1"
        ),
        speaker_threshold=float(os.environ.get("TRANSCRIBE_SPEAKER_THRESHOLD", "0.45")),
        min_speaker_seconds=float(os.environ.get("TRANSCRIBE_MIN_SPEAKER_SECONDS", "2.0")),
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        tidy_enabled=_env_bool("TRANSCRIBE_TIDY", True),
        llm_repo=os.environ.get("TRANSCRIBE_LLM_REPO", DEFAULT_LLM_REPO),
        llm_file=os.environ.get("TRANSCRIBE_LLM_FILE", DEFAULT_LLM_FILE),
        llm_model=os.environ.get("TRANSCRIBE_LLM_MODEL") or None,
        llm_port=int(os.environ.get("TRANSCRIBE_LLM_PORT", "8080")),
        llm_ctx=int(os.environ.get("TRANSCRIBE_LLM_CTX", "8192")),
        llm_health_timeout=int(os.environ.get("TRANSCRIBE_LLM_HEALTH_TIMEOUT", "120")),
        llm_request_timeout=int(os.environ.get("TRANSCRIBE_LLM_REQUEST_TIMEOUT", "120")),
    )
    return settings


def ensure_dirs(settings: Settings) -> None:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
