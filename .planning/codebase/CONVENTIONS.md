---
generated: 2026-07-30
---
# Code Conventions

**Analysis Date:** 2026-07-30

## Code Style

**Formatting:**
- No automated formatter or linter is configured (no `.flake8`, `.pylintrc`, `ruff.toml`, `pyproject.toml`, or `.editorconfig` exist in the repo root).
- Indentation: 4 spaces (standard Python).
- Line length: not enforced; code wraps naturally at roughly 88–100 characters.
- String quotes: double quotes (`"`) are preferred in `service/` modules (e.g., `service/api.py`, `service/storage.py`); `transcribe_simple.py` uses double quotes for most strings but f-strings appear throughout.
- No trailing whitespace observed; files end with a single newline.

**Docstrings:**
- Module-level docstrings are present at the top of every source file, written as triple-quoted strings.
- Function and class docstrings use the Google-style "Args:" / "Returns:" / "Raises:" format (seen in `transcribe_simple.py`, `gui/audio_converter.py`, `gui/transcription_worker.py`).
- Example from `transcribe_simple.py`:
  ```python
  def transcribe_audio(
      audio_file: str,
      num_speakers: Optional[int] = None,
      ...
  ) -> List[Dict[str, Any]]:
      """
      Transcribe audio with speaker diarization.

      PURPOSE: Main transcription function that processes audio files with speaker
      diarization and ASR. Supports streaming output via callbacks for GUI integration.

      CONTEXT: Designed for both CLI and GUI usage. Callbacks enable live updates
      during processing, especially useful for long audio files.

      Args:
          audio_file: Path to audio file (supports various formats via ffmpeg)
          num_speakers: Number of speakers (None = auto-detect)
          ...

      Returns:
          List of all transcription segments with timestamps and speakers.

      Raises:
          FileNotFoundError: If audio file doesn't exist
          ...
      """
  ```

**Future annotations:**
- `from __future__ import annotations` is used in all `service/` modules (`service/api.py`, `service/config.py`, `service/jobs.py`, `service/storage.py`, `service/preflight.py`, `service/schemas.py`) to enable PEP 604 union syntax (`str | None`) on older Python.

## Naming Conventions

**Files:**
- `snake_case` for all Python modules (e.g., `transcribe_simple.py`, `markdown_export.py`, `audio_converter.py`, `youtube_download.py`).
- Test files: `test_*.py` (e.g., `tests/test_service_paths.py`, `tests/test_markdown_export.py`).
- Entry points: `gui_main.py`, `transcribe_simple.py` (underscore-separated, no package prefix).

**Functions:**
- `snake_case` for all functions (e.g., `validate_audio_path`, `transcribe_audio`, `load_diarization_pipeline`, `get_device`).
- Private/internal functions use a leading underscore: `_write_transcript`, `_run_transcription_job`, `_load_whisper_model`, `_select_whisper_backend`, `_iter_audio_chunks`, `_write_temp_wav`, `_segment_speaker`, `_append_segment`, `_transcribe_chunk_with_faster_whisper`, `_transcribe_chunk_with_openai_whisper`, `_transcribe_with_parakeet`, `_transcribe_with_whisper`, `_faster_whisper_compute_type`, `_load_faster_whisper_model`, `_configure_runtime_env`, `_row_to_status`.

**Classes:**
- `PascalCase` for all classes (e.g., `Settings`, `TranscriptionWorker`, `CustomProgressHook`, `JobStatus`).
- Private/internal classes use a leading underscore: `_SpeakerTurnMerger`.

**Variables:**
- `snake_case` for local and module-level variables.
- Module-level constants use `UPPER_CASE` (e.g., `DEFAULT_DIARIZATION_MODEL_ID`, `FALLBACK_DIARIZATION_MODEL_ID`, `PARAKEET_MODEL_ID`, `DEFAULT_WHISPER_MODEL_NAME`, `DEFAULT_WHISPER_BACKEND`, `SUPPORTED_MANUAL_LANGUAGES`, `SAMPLE_RATE`, `CHUNK_DURATION`, `CHUNK_OVERLAP`, `_WHISPER_MODEL_CACHE`, `_FASTER_WHISPER_MODEL_CACHE`, `SUPPORTED_FORMATS`).

**Type aliases:**
- No custom type aliases are defined; `Optional[T]`, `List[T]`, `Dict[str, Any]`, `Callable`, `Any` from `typing` are used directly.

## Patterns

**Import organization:**
- Standard library imports first, then third-party, then local/project imports.
- `from __future__ import annotations` is the very first import (when present).
- Example from `service/api.py`:
  ```python
  from __future__ import annotations

  import json
  import logging
  import os
  import shutil
  from contextlib import asynccontextmanager
  from typing import Annotated, Optional

  from dotenv import load_dotenv
  from fastapi import Depends, FastAPI, HTTPException, Response, status

  from service.config import settings
  from service import jobs, storage
  from service.preflight import cuda_info, run_preflight
  from service.schemas import (...)
  ```

**Callback-based streaming:**
- Long-running functions accept optional `progress_callback`, `segment_callback`, and `check_interrupt` callables to support both CLI and GUI usage (see `transcribe_audio()` in `transcribe_simple.py`).
- Progress callbacks receive `(message: str, progress: float)` where progress is 0.0–1.0.
- Segment callbacks receive a dict with keys `start`, `end`, `speaker`, `text`.

**Model caching:**
- In-memory model caches use module-level dicts keyed by tuples (e.g., `_WHISPER_MODEL_CACHE: Dict[tuple, Any]`, `_FASTER_WHISPER_MODEL_CACHE: Dict[tuple, Any]` in `transcribe_simple.py`).
- Cache keys include device name and compute type to avoid loading duplicates.

**Environment configuration:**
- Settings loaded once at import via `os.getenv()` with helper functions (`_env_int`, `_env_float`, `_env_bool`) in `service/config.py`.
- `.env` files loaded with `python-dotenv` (`load_dotenv()`) at the top of entry-point scripts.
- Cache directories (`XDG_CACHE_HOME`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`) are set to repo-local `.cache/` in `transcribe_simple.py` and `gui_main.py` to avoid home-directory permission issues.

**torch.load compatibility patch:**
- `transcribe_simple.py` and `gui_main.py` both patch `torch.load` so `weights_only=None` becomes `False`, matching Lightning/pyannote expectations under PyTorch 2.6+ (see lines 51–62 of `transcribe_simple.py`).
- The patch is guarded by `getattr(torch, "_speaker_sep_torch_load_patched", False)` to avoid double-patching.

**SQLite job store:**
- `service/storage.py` uses a module-level `sqlite3.Connection` with `check_same_thread=False` and a `threading.Lock` (`_db_lock`) for thread safety.
- WAL mode is enabled via `PRAGMA journal_mode=WAL`.
- All write operations are wrapped in `with _db_lock:` blocks.

**Path validation:**
- `service/jobs.py:validate_audio_path()` resolves paths and enforces that they are under `SERVICE_DATA_DIR` to prevent path traversal.

## Error Handling

**Strategy:**
- Validation errors raise `ValueError` (e.g., `validate_audio_path` in `service/jobs.py`).
- Processing failures raise `RuntimeError` (e.g., `transcribe_audio` in `transcribe_simple.py`).
- API errors use `fastapi.HTTPException` with appropriate status codes (400, 401, 403, 404, 409, 429, 500, 503).
- Unhandled exceptions in background workers are caught and logged via `logger.exception()`.

**Patterns:**
- `try/except` blocks wrap external calls (ffmpeg subprocess, model loading, transcription).
- `logger.exception()` is used for unexpected errors to capture full tracebacks (e.g., `service/jobs.py:111`).
- `logger.error()` with `%s`-style format strings is used in `service/` modules (e.g., `logger.error("Job %s failed", job_id)`).
- `logger.warning()` with `%s`-style format strings is used for non-fatal issues (e.g., `service/jobs.py:107`).
- `f-strings` are used for logging in `transcribe_simple.py` (e.g., `logger.info(f"Loading Whisper model {model_name} on {device_name}")`).
- `finally` blocks ensure cleanup of temporary files (e.g., `os.unlink(tmp_audio_path)` in transcription loops).
- `raise ... from e` is used to preserve exception chains (e.g., `service/api.py:129`).

## Logging

**Framework:**
- Python standard library `logging` module.

**Patterns:**
- Module-level loggers: `logger = logging.getLogger(__name__)` (used in `service/api.py`, `service/jobs.py`, `service/preflight.py`, `transcribe_simple.py`, `gui/transcription_worker.py`).
- `service/api.py` configures root logging at module import:
  ```python
  logging.basicConfig(
      level=os.getenv("SERVICE_LOG_LEVEL", "INFO"),
      format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
  ```
- `transcribe_simple.py` configures a named logger `transcribe_simple` with a custom formatter:
  ```python
  handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter(
      '%(asctime)s [%(levelname)s] [BACKEND] %(message)s',
      datefmt='%H:%M:%S'
  ))
  logger.addHandler(handler)
  logger.setLevel(logging.WARNING)
  ```
- NeMo loggers are explicitly set to `WARNING` level to suppress noisy output.
- Log level is controlled via environment variables: `SERVICE_LOG_LEVEL` (service), `UVICORN_LOG_LEVEL` (uvicorn).

## Comments

**When to comment:**
- Module-level docstrings describe purpose and context.
- Function docstrings use Google-style with Args/Returns/Raises.
- Inline comments explain non-obvious logic (e.g., "MPS does not support float64", "PyTorch 2.6+ defaults torch.load(..., weights_only=True)").
- Section separators with `# ===` and `# ---` are used to mark major code sections (e.g., `# === MPS FIX ===`, `# === STEP 1: LOAD DIARIZATION PIPELINE ===`).

**JSDoc/TSDoc:**
- Not applicable (Python project). Docstrings serve the same role.

## Function Design

**Size:**
- Functions are kept focused; large functions like `transcribe_audio()` (lines 800–1014) are broken into logical steps with clear section comments.
- Helper functions are extracted for reusable logic (e.g., `_write_temp_wav`, `_segment_speaker`, `_append_segment`, `_iter_audio_chunks`).

**Parameters:**
- Optional parameters default to `None` and are typed with `Optional[T]`.
- Callbacks use `Optional[Callable[...]]` with `None` defaults.
- Keyword arguments are used for optional parameters.

**Return values:**
- Functions return typed values: `List[Dict[str, Any]]`, `str`, `bool`, `Optional[str]`, `tuple[bool, int]`.
- Empty collections are returned as `[]` or `{}` rather than `None` when the caller expects a collection (e.g., `merge_consecutive_same_speaker` returns `[]`).

## Module Design

**Exports:**
- No `__all__` is defined in any module.
- `service/__init__.py` contains only a module docstring.
- `gui/__init__.py` contains only a module docstring.

**Barrel files:**
- Not used. Imports are direct from specific modules (e.g., `from service.config import settings`, `from service import jobs, storage`).

---

*Convention analysis: 2026-07-30*
