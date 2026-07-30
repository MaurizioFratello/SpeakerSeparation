---
generated: 2026-07-30
---
# External Integrations

## APIs and Services

### Hugging Face Hub

- **Purpose**: Hosts all ML models (diarization, ASR, Whisper)
- **SDK/Client**: `huggingface-hub>=0.20.0` (model download), `pyannote.audio.Pipeline.from_pretrained()` (diarization), `nemo_asr.models.ASRModel.from_pretrained()` (Parakeet)
- **Auth**: `HUGGINGFACE_TOKEN` environment variable (set in `.env`, loaded via `python-dotenv`)
- **Models**:
  - `pyannote/speaker-diarization-3.1` — Primary diarization pipeline (`transcribe_simple.py:74`, `gui/main_window.py:51`)
  - `pyannote/speaker-diarization-community-1` — Fallback diarization model (`transcribe_simple.py:75`, `gui/main_window.py:52`)
  - `nvidia/parakeet-tdt-0.6b-v3` — NeMo Parakeet ASR model for automatic transcription (`transcribe_simple.py:76`)
- **Cache**: Repo-local `./.cache/huggingface` (local/GUI) or `SERVICE_CACHE_ROOT` (Docker service). Set via `XDG_CACHE_HOME`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`
- **Notes**: All gated checkpoints require the same HF account as the token. Terms must be accepted on huggingface.co for each model.

### Whisper Models

- **Purpose**: Speaker-attributed transcription (language-specific: English, German)
- **SDK/Client**: `openai-whisper` (MPS/CPU) or `faster-whisper` (CUDA/CPU via CTranslate2)
- **Auth**: None (public models)
- **Models**: `turbo` by default (configurable via `WHISPER_MODEL` env var)
- **Backend selection**: `_select_whisper_backend()` in `transcribe_simple.py:535` — uses `faster-whisper` for CUDA/CPU, `openai-whisper` for MPS (faster-whisper does not support MPS)
- **Cache**: Repo-local `./.cache/whisper` (`transcribe_simple.py:24`)

### YouTube

- **Purpose**: Download audio from YouTube URLs for transcription
- **SDK/Client**: `yt-dlp` (Python package, `gui/youtube_download.py:11`)
- **Auth**: None
- **Usage**: `download_youtube_audio()` extracts best audio, saves to temp directory, returns metadata (`gui/youtube_download.py:29`)
- **URL validation**: Regex pattern for `youtube.com` and `youtu.be` (`gui/youtube_download.py:16`)

## Databases and Storage

### SQLite

- **Type**: Embedded relational database (stdlib `sqlite3`)
- **Connection**: `service/storage.py:19` — `sqlite3.connect()` with `check_same_thread=False`
- **DB path**: `SERVICE_DB_PATH` env var (default `/var/lib/transcription/jobs.db`)
- **Schema**: Two tables:
  - `jobs` — Job metadata (id, status, audio_path, num_speakers, test_mode, progress, error, output_path, result_segments_json, cancel_requested, timestamps)
  - `job_events` — Audit trail (id, job_id, ts, event, detail)
- **WAL mode**: Enabled (`service/storage.py:36`) for concurrent read/write
- **Thread safety**: `_db_lock` threading.Lock around all operations (`service/storage.py:16`)
- **Recovery**: `recover_stale_running_jobs()` marks interrupted jobs as failed after container restart (`service/storage.py:322`)

### File Storage

- **Input audio**: `SERVICE_DATA_DIR` (default `/data`) — Docker volume `./data:/data:ro` (read-only mount)
- **Transcript output**: `SERVICE_OUTPUT_DIR` (default `/exports`) — Docker volume `./exports:/exports`
- **Model cache**: `SERVICE_CACHE_ROOT` (default `/cache`) — Docker volume `transcription_cache:/cache`
- **SQLite DB**: `SERVICE_DB_PATH` (default `/var/lib/transcription/jobs.db`) — Docker volume `transcription_db:/var/lib/transcription`
- **GUI local cache**: Repo-local `./.cache/` tree (`.cache/huggingface`, `.cache/whisper`)
- **GUI transcript export**: `exports/` directory at repo root for YouTube-sourced transcripts (`gui/main_window.py:894`)

### Caching

- **Hugging Face Hub cache**: `HUGGINGFACE_HUB_CACHE` env var
- **Whisper cache**: `WHISPER_MODEL` download root (`.cache/whisper`)
- **In-memory caches**: `_WHISPER_MODEL_CACHE` and `_FASTER_WHISPER_MODEL_CACHE` dicts in `transcribe_simple.py:86-87` for model reuse within a process

## Authentication

### Hugging Face Token

- **Provider**: Hugging Face
- **Implementation**: `HUGGINGFACE_TOKEN` environment variable, loaded from `.env` via `python-dotenv`
- **Used by**: `transcribe_simple.py:843` (raises `RuntimeError` if missing), `gui_main.py:64` (GUI warning dialog if missing), `service/preflight.py:16` (health check)
- **Scope**: Required for all gated pyannote models and dependent checkpoints

### Service API Token

- **Provider**: Custom (bearer token)
- **Implementation**: `SERVICE_API_TOKEN` environment variable (`service/config.py:54`)
- **Used by**: `service/api.py:54` — `require_agent_auth()` dependency on all `/jobs*` and `/admin/warmup` endpoints
- **Format**: `Authorization: Bearer <SERVICE_API_TOKEN>` header
- **Behavior**: If unset, all endpoints are open (no auth). If set, `GET /health` remains unauthenticated

## Webhooks and Callbacks

### Incoming Webhooks

- **None detected** — The service does not accept incoming webhooks. It uses a polling-based job API (`GET /jobs/{job_id}`) instead.

### Outgoing Webhooks/Callbacks

- **None detected** — The service does not make outgoing webhook calls.
- **Callback mechanism**: Instead of webhooks, the service uses:
  - **Progress callbacks**: `progress_callback(message, progress)` — forwarded to DB via `storage.update_progress()` (`service/jobs.py:81`)
  - **Segment callbacks**: `segment_callback(segment)` — used by GUI for streaming transcript display (`gui/transcription_worker.py:135`)
  - **Interrupt checks**: `check_interrupt()` — cooperative cancellation (`service/jobs.py:84`)

## Monitoring and Observability

### Health Check

- **Endpoint**: `GET /health` (`service/api.py:92`)
- **Auth**: None (unauthenticated for orchestrators)
- **Checks**: CUDA availability, ffmpeg presence, HuggingFace token configured
- **Response**: `HealthResponse` with `ok`, `cuda_available`, `cuda_device_count`, `ffmpeg_ok`, `huggingface_token_configured`, paths, and `message`
- **HTTP status**: 200 when healthy, 503 when unhealthy
- **Docker healthcheck**: `curl -fsS http://127.0.0.1:8080/health` (interval 30s, timeout 10s, start-period 120s, retries 3)

### Logging

- **Framework**: Python `logging` module
- **Service**: `logging.basicConfig()` with configurable level via `SERVICE_LOG_LEVEL` (`service/api.py:31`)
- **Format**: `%(asctime)s %(levelname)s %(name)s %(message)s`
- **Backend**: `transcribe_simple` logger at WARNING level (`transcribe_simple.py:339`)
- **GUI**: Separate loggers (`main_window`, `transcription_worker`) at WARNING level
- **NeMo**: Suppressed to WARNING level (`transcribe_simple.py:318-323`)
- **PyTorch distributed**: Suppressed to ERROR (`transcribe_simple.py:326-327`)

### Job Lifecycle Events

- **Audit trail**: `job_events` table stores all state transitions (`queued`, `running`, `succeeded`, `failed`, `cancelled`, `cancel_requested`)
- **Preflight**: `service/preflight.py` runs CUDA/ffmpeg/token checks at startup; failure raises `RuntimeError` and prevents service start

## CI/CD and Deployment

### Hosting

- **Platform**: Docker (single-container, single GPU host)
- **Base image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- **Orchestration**: Docker Compose (`docker-compose.yml`)
- **GPU**: `gpus: all` in compose, requires NVIDIA Container Toolkit on host

### CI Pipeline

- **Not detected** — No CI/CD pipeline configuration found (no `.github/`, `.gitlab-ci.yml`, `.circleci/`, etc.)
- **Build**: `docker compose up --build` (manual)
- **Restart policy**: `restart: unless-stopped`

### Environment Configuration

**Required env vars:**
- `HUGGINGFACE_TOKEN` — Hugging Face API token (gated model access)

**Optional env vars:**
- `SERVICE_API_TOKEN` — Bearer token for job/admin endpoints
- `SERVICE_HOST`, `SERVICE_PORT` — API bind address and port
- `SERVICE_DATA_DIR`, `SERVICE_OUTPUT_DIR` — Audio input and transcript output directories
- `SERVICE_DB_PATH` — SQLite database path
- `SERVICE_CACHE_ROOT` — Model cache root
- `SERVICE_MAX_QUEUED_JOBS` — Max queued jobs (default 16)
- `SERVICE_JOB_TIMEOUT_SEC` — Per-job timeout (default 14400)
- `SERVICE_REQUIRE_CUDA` — Fail startup if CUDA unavailable (default true)
- `SERVICE_LOG_LEVEL` — Python logging level (default INFO)
- `WHISPER_MODEL` — Whisper model size (default `turbo`)
- `WHISPER_BACKEND` — Whisper backend (default `auto`)
- `FASTER_WHISPER_COMPUTE_TYPE` — faster-whisper compute type override

**Secrets location:**
- `.env` file at repo root (gitignored, never committed)
- `.env.example` provides template

---

*Integration audit: 2026-07-30*
