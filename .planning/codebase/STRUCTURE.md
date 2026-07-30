---
generated: 2026-07-30
---
# Project Structure

**Analysis Date:** 2026-07-30

## Directory Layout

```
SpeakerSeparation/
├── transcribe_simple.py          # Core transcription pipeline (CLI + library)
├── gui_main.py                   # PySide6 GUI launcher entry point
├── service/                      # Headless FastAPI microservice
│   ├── __init__.py               # Package docstring
│   ├── __main__.py               # Uvicorn runner (python -m service)
│   ├── api.py                    # FastAPI app, endpoints, auth, lifespan
│   ├── config.py                 # Settings singleton (env-based)
│   ├── jobs.py                   # Background worker (ThreadPool + scheduler)
│   ├── schemas.py                # Pydantic models for API
│   ├── storage.py                # SQLite job store (thread-safe)
│   └── preflight.py              # CUDA/ffmpeg/HF token startup checks
├── gui/                          # PySide6 desktop GUI module
│   ├── __init__.py
│   ├── main_window.py            # MainWindow + DragDropWidget
│   ├── transcription_worker.py   # TranscriptionWorker (QThread)
│   ├── audio_converter.py        # ffmpeg WAV conversion
│   ├── markdown_export.py        # Markdown export + speaker merging
│   └── youtube_download.py       # yt-dlp YouTube audio download
├── tests/                        # Test suite
│   ├── test_service_paths.py     # Path validation (unittest)
│   ├── test_markdown_export.py   # Markdown export (pytest)
│   ├── test_youtube_download.py  # YouTube download (pytest, mocked)
│   ├── test_steps.py             # Step-by-step diagnostic script
│   ├── test_device_detection.py
│   ├── test_diarization_only.py
│   ├── test_gpu_integration.py
│   ├── test_minimal_pipeline.py
│   ├── test_parakeet_basic.py
│   ├── test_parakeet_integration.py
│   └── benchmark_*.py
├── scripts/                      # Setup and utility scripts
│   ├── requirements_ml.txt       # ML stack (pyannote, whisper, nemo)
│   ├── requirements_service.txt  # Service stack (fastapi, uvicorn)
│   ├── requirements_transcription.txt  # Local dev requirements
│   ├── docker_constraints.txt    # CUDA wheel pins for Docker
│   ├── download_models.py        # Pre-download HF models
│   ├── run_transcription.sh      # Conda activation + run
│   ├── setup_and_run.sh          # Full setup + run
│   └── fix_models.sh
├── docs/                         # Documentation
│   ├── MICROSERVICE_API.md       # API reference
│   ├── README_IMPROVED.md
│   ├── SETUP_COMPLETE.md
│   ├── SPEED_OPTIMIZATIONS.md
│   ├── TRANSCRIPTION_GUIDE.md
│   └── TROUBLESHOOTING.md
├── exports/                      # Generated transcript outputs
├── Dockerfile                    # CUDA 12.4 microservice image
├── docker-compose.yml            # Docker Compose (transcription service)
├── .env.example                  # Template for environment config
├── .env                          # Actual config (gitignored)
├── requirements_gui.txt          # PySide6 + yt-dlp + dotenv
├── AGENTS.md                     # Workspace preferences/facts
├── README.md                     # Project overview
├── SOLUTION_SUMMARY.md           # Technical documentation
├── GUI_README.md                 # GUI-specific docs
├── .gitignore
├── .dockerignore
└── .planning/                    # GSD planning artifacts
    └── codebase/                 # Codebase mapping documents
```

## Directory Purposes

**`service/`** — Headless transcription microservice (FastAPI + SQLite + background worker). Designed for Docker/CUDA deployment and LLM agent integration. All files are part of the `service` Python package.
- Key files: `service/api.py`, `service/jobs.py`, `service/storage.py`, `service/config.py`, `service/schemas.py`, `service/preflight.py`, `service/__main__.py`

**`gui/`** — PySide6 desktop GUI module. Provides drag-and-drop audio input, YouTube download, live streaming transcript display, and auto-save.
- Key files: `gui/main_window.py`, `gui/transcription_worker.py`, `gui/audio_converter.py`, `gui/markdown_export.py`, `gui/youtube_download.py`

**`tests/`** — Test suite with mixed frameworks: `unittest` for service path validation, `pytest` for GUI utilities. Also includes diagnostic and benchmark scripts.
- Key files: `tests/test_service_paths.py`, `tests/test_markdown_export.py`, `tests/test_youtube_download.py`

**`scripts/`** — Setup, requirements, and utility scripts. Split into ML stack, service stack, and transcription requirements.
- Key files: `scripts/requirements_ml.txt`, `scripts/requirements_service.txt`, `scripts/requirements_transcription.txt`, `scripts/docker_constraints.txt`, `scripts/download_models.py`

**`docs/`** — Project documentation including API reference, setup guides, and troubleshooting.
- Key files: `docs/MICROSERVICE_API.md`, `docs/TROUBLESHOOTING.md`, `docs/TRANSCRIPTION_GUIDE.md`

**`exports/`** — Runtime output directory for generated transcript files (txt and markdown). Populated by GUI and service.

## Key File Locations

**Entry Points:**
- `gui_main.py` — PySide6 GUI launcher (`python gui_main.py`)
- `transcribe_simple.py` — CLI transcription (`python transcribe_simple.py [audio_file]`)
- `service/__main__.py` — Microservice runner (`python -m service` or `uvicorn service.api:app`)
- `Dockerfile` — Docker image build (`docker compose up --build`)

**Configuration:**
- `service/config.py` — Service settings singleton (env-based, `Settings` class)
- `.env.example` — Template for environment variables
- `docker-compose.yml` — Docker Compose service definition
- `scripts/docker_constraints.txt` — CUDA wheel version pins

**Core Logic:**
- `transcribe_simple.py` — Transcription pipeline: `transcribe_audio()`, `load_diarization_pipeline()`, `_transcribe_with_parakeet()`, `_transcribe_with_whisper()`, `_SpeakerTurnMerger`, `CustomProgressHook`
- `service/jobs.py` — Background worker: `start_worker()`, `stop_worker()`, `_worker_loop()`, `_run_transcription_job()`, `validate_audio_path()`
- `service/storage.py` — SQLite job store: `init_db()`, `create_job()`, `try_claim_next_job()`, `mark_succeeded()`, `mark_failed()`, `recover_stale_running_jobs()`

**GUI:**
- `gui/main_window.py` — `MainWindow`, `DragDropWidget`, pipeline loading, UI state
- `gui/transcription_worker.py` — `TranscriptionWorker(QThread)`, signal-based streaming
- `gui/markdown_export.py` — `segments_to_markdown()`, `merge_consecutive_same_speaker()`
- `gui/audio_converter.py` — `convert_to_wav()`, `is_supported_format()`
- `gui/youtube_download.py` — `download_youtube_audio()`, `is_youtube_url()`

**API:**
- `service/api.py` — FastAPI app, endpoints, `require_agent_auth`, lifespan handler
- `service/schemas.py` — Pydantic models: `JobStatus`, `CreateJobRequest`, `JobStatusResponse`, `TranscriptSegment`, `JobResultResponse`, `HealthResponse`

**Tests:**
- `tests/test_service_paths.py` — Path traversal validation (unittest)
- `tests/test_markdown_export.py` — Markdown export and speaker merging (pytest)
- `tests/test_youtube_download.py` — YouTube download with mocked yt-dlp (pytest)

## Naming Conventions

**Files:**
- `snake_case.py` for all Python modules (e.g., `transcription_worker.py`, `audio_converter.py`, `markdown_export.py`)
- `__init__.py`, `__main__.py`, `__pycache__/` for Python package conventions
- `test_*.py` for pytest test files (e.g., `test_markdown_export.py`)
- `test_*.py` also used for unittest files (e.g., `test_service_paths.py`)
- `benchmark_*.py` for benchmark scripts (e.g., `benchmark_gpu_performance.py`)
- `*.txt` for requirements files (e.g., `requirements_ml.txt`, `requirements_service.txt`)
- `*.md` for documentation (e.g., `MICROSERVICE_API.md`, `TROUBLESHOOTING.md`)

**Directories:**
- `service/` — FastAPI microservice package
- `gui/` — PySide6 GUI package
- `tests/` — test suite
- `scripts/` — setup and utility scripts
- `docs/` — documentation
- `exports/` — generated transcript outputs
- `.planning/` — GSD planning artifacts
- `.cache/` — HuggingFace/torch model cache (gitignored)

**Classes:**
- `PascalCase` (e.g., `MainWindow`, `TranscriptionWorker`, `DragDropWidget`, `Settings`, `JobStatus`)
- `CustomProgressHook` — callable class with `__enter__`/`__exit__`/`__call__`

**Functions:**
- `snake_case` (e.g., `transcribe_audio`, `validate_audio_path`, `convert_to_wav`, `merge_consecutive_same_speaker`)
- Private functions prefixed with `_` (e.g., `_run_transcription_job`, `_worker_loop`, `_write_transcript`, `_load_whisper_model`)

**Constants:**
- `UPPER_SNAKE_CASE` (e.g., `DEFAULT_DIARIZATION_MODEL_ID`, `DEFAULT_WHISPER_MODEL_NAME`, `SAMPLE_RATE`, `CHUNK_DURATION`, `CHUNK_OVERLAP`, `SUPPORTED_FORMATS`, `SUPPORTED_MANUAL_LANGUAGES`)

**Environment Variables:**
- `SERVICE_*` — service configuration (e.g., `SERVICE_HOST`, `SERVICE_PORT`, `SERVICE_DATA_DIR`)
- `HUGGINGFACE_TOKEN` — HuggingFace API token
- `WHISPER_MODEL` — Whisper model name (default: `turbo`)
- `WHISPER_BACKEND` — `auto`, `faster-whisper`, `openai-whisper`
- `NUM_SPEAKERS`, `TEST_MODE`, `TRANSCRIPTION_LANGUAGE` — CLI/transcription config
- `UVICORN_LOG_LEVEL`, `SERVICE_LOG_LEVEL` — logging

## Where to Add New Code

**New API Endpoint:**
- Implementation: `service/api.py` — add route function with `@app.get/post` decorator
- Schema: `service/schemas.py` — add Pydantic model if new request/response type
- Storage: `service/storage.py` — add DB function if new persistence needed

**New Service Module:**
- Implementation: `service/<new_module>.py` — new file in the `service/` package
- Register: import in `service/api.py` if it needs to be wired into the API

**New GUI Component:**
- Implementation: `gui/<new_component>.py` — new file in the `gui/` package
- Register: import in `gui/main_window.py` and connect to `MainWindow`

**New Transcription Feature:**
- Implementation: `transcribe_simple.py` — add function near related logic (e.g., new ASR backend near `_transcribe_with_parakeet`)
- Entry points: `transcribe_simple.py:transcribe_audio()` for CLI, `service/jobs.py:_run_transcription_job()` for service, `gui/transcription_worker.py` for GUI

**New Test:**
- Implementation: `tests/test_<feature>.py` — use pytest for GUI utilities, unittest for service logic
- Pattern: follow existing test structure (e.g., `tests/test_markdown_export.py` for pytest, `tests/test_service_paths.py` for unittest)

**New Script:**
- Implementation: `scripts/<name>.py` or `scripts/<name>.sh`

**New Documentation:**
- Implementation: `docs/<name>.md`

## Special Directories

**`.cache/`**
- Purpose: HuggingFace and torch model cache. Redirected from `~/.cache` to repo-local for permission safety.
- Generated: Yes (by `transcribe_simple.py`, `gui_main.py`, `service/api.py`)
- Committed: No (gitignored)

**`.planning/codebase/`**
- Purpose: GSD codebase mapping documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (by `/gsd-map-codebase`)
- Committed: Yes (planning artifacts)

**`exports/`**
- Purpose: Generated transcript output files (txt and markdown).
- Generated: Yes (by GUI auto-save, service job completion)
- Committed: No (gitignored)

**`archive/`**
- Purpose: Old versions and logs (not actively used).
- Generated: No
- Committed: Yes

**`.uv-cache/`**
- Purpose: uv package cache for virtual environment management.
- Generated: Yes (by `uv`)
- Committed: No (gitignored)

---

*Structure analysis: 2026-07-30*
