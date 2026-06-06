"""Runtime configuration, overridable via environment variables.

All settings are read once via :func:`load_settings`. Paths default to a
``data/`` directory next to the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    uploads_dir: Path
    outputs_dir: Path
    db_path: Path
    model_name: str
    device: str  # "auto" | "cuda" | "cpu"
    compute_type: str  # "auto" | "float16" | "int8" | "int8_float16"
    batch_size: int
    use_vad: bool
    default_language: str  # "auto" | "nl" | "en"
    max_upload_mb: int


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("TRANSCRIBE_DATA_DIR", PROJECT_ROOT / "data"))
    settings = Settings(
        host=os.environ.get("TRANSCRIBE_HOST", "127.0.0.1"),
        port=int(os.environ.get("TRANSCRIBE_PORT", "8000")),
        data_dir=data_dir,
        uploads_dir=data_dir / "uploads",
        outputs_dir=data_dir / "outputs",
        db_path=data_dir / "transcribe.db",
        model_name=os.environ.get("TRANSCRIBE_MODEL", "large-v3"),
        device=os.environ.get("TRANSCRIBE_DEVICE", "auto"),
        compute_type=os.environ.get("TRANSCRIBE_COMPUTE_TYPE", "auto"),
        batch_size=int(os.environ.get("TRANSCRIBE_BATCH_SIZE", "8")),
        use_vad=_env_bool("TRANSCRIBE_VAD", True),
        default_language=os.environ.get("TRANSCRIBE_LANGUAGE", "auto"),
        max_upload_mb=int(os.environ.get("TRANSCRIBE_MAX_UPLOAD_MB", "2048")),
    )
    return settings


def ensure_dirs(settings: Settings) -> None:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
