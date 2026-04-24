"""Runtime configuration from environment (Docker Compose friendly)."""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    """Service settings loaded once at import (restart to apply changes)."""

    host: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    port: int = _env_int("SERVICE_PORT", 8080)

    # Audio files must live under this directory (path traversal protection).
    data_dir: Path = Path(os.getenv("SERVICE_DATA_DIR", "/data")).resolve()

    # Transcript outputs written here.
    output_dir: Path = Path(os.getenv("SERVICE_OUTPUT_DIR", "/exports")).resolve()

    # SQLite job database.
    db_path: Path = Path(
        os.getenv("SERVICE_DB_PATH", "/var/lib/transcription/jobs.db")
    ).resolve()

    # Hugging Face + model caches (transcribe_simple also respects XDG_CACHE_HOME / HF_HOME).
    cache_root: Path = Path(os.getenv("SERVICE_CACHE_ROOT", "/cache")).resolve()

    max_queued_jobs: int = _env_int("SERVICE_MAX_QUEUED_JOBS", 16)
    job_timeout_sec: float = _env_float("SERVICE_JOB_TIMEOUT_SEC", 14_400.0)  # 4h default

    # Optional Bearer token; if set, all /jobs* and /admin* require Authorization.
    api_token: str | None = os.getenv("SERVICE_API_TOKEN") or None

    # Preflight: if true, /health returns 503 when CUDA is not available.
    require_cuda: bool = _env_bool("SERVICE_REQUIRE_CUDA", True)

    repo_root: Path = Path(__file__).resolve().parent.parent


settings = Settings()
