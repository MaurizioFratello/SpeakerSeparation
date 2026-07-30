---
generated: 2026-07-30
---
# Architecture

**Analysis Date:** 2026-07-30

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                           │
│  ┌──────────────────┐   ┌──────────────────────┐   ┌─────────────┐ │
│  │  gui_main.py     │   │ transcribe_simple.py │   │ service/    │ │
│  │  (PySide6 GUI)   │   │  (CLI entry)         │   │ api.py      │ │
│  │  gui/main_window │   │                      │   │ (FastAPI)   │ │
│  │  gui/* workers   │   │                      │   │             │ │
│  └────────┬─────────┘   └──────────┬───────────┘   └──────┬──────┘ │
├──────────┴──────────────────────┴──────────────────────┴────────┤
│                        Application Layer                           │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  service/jobs.py                                             │ │
│  │  (Background worker: SQLite → transcribe_audio)              │ │
│  │  service/storage.py                                          │ │
│  │  (SQLite job store, thread-safe)                             │ │
│  └──────────────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│                        Core Layer                                 │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  transcribe_simple.py                                        │ │
│  │  (Diarization + ASR pipeline, callbacks)                     │ │
│  │  - load_diarization_pipeline()  (pyannote)                   │ │
│  │  - transcribe_audio()  (main orchestrator)                   │ │
│  │  - _transcribe_with_parakeet()  (NeMo)                       │ │
│  │  - _transcribe_with_whisper()  (faster-whisper/openai)       │ │
│  │  - _SpeakerTurnMerger, CustomProgressHook                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│                        Infrastructure Layer                        │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ service/config.py│  │ service/     │  │ service/     │         │
│  │ (Settings/env)   │  │ schemas.py   │  │ preflight.py │         │
│  └──────────────────┘  │ (Pydantic)   │  │ (CUDA/ffmpeg)│         │
│                        └──────────────┘  └──────────────┘         │
└────────────────────────────────────────────────────────────────────┘
```

## Pattern

**Three-tier application with dual frontends:**

1. **Desktop/GUI application** — PySide6 GUI (`gui_main.py` + `gui/`) that drives the transcription pipeline directly in-process, loading the pyannote pipeline in the main thread and running `transcribe_audio()` in a `QThread` worker.
2. **CLI application** — `transcribe_simple.py` run as a script, calling `transcribe_audio()` with console progress/segment callbacks.
3. **Headless microservice** — FastAPI app (`service/api.py`) that exposes an async job API. A background worker thread claims SQLite jobs and runs `transcribe_audio()` in a `ThreadPoolExecutor`. Designed for Docker/CUDA deployment and LLM agent integration.

All three frontends share the same core: `transcribe_simple.py`'s `transcribe_audio()` function, which orchestrates diarization → ASR → segment merging via injectable callbacks.

## Layers

### Presentation Layer

**GUI** (`gui_main.py`, `gui/`):
- `gui_main.py` — entry point, creates `QApplication`, loads pipeline, shows `MainWindow`.
- `gui/main_window.py` — `MainWindow` (QMainWindow), `DragDropWidget` (QLabel). Manages UI state, file/YouTube selection, progress display, transcript rendering, auto-save.
- `gui/transcription_worker.py` — `TranscriptionWorker(QThread)`. Runs `transcribe_audio()` in background, emits `progress`, `segment_ready`, `finished`, `error` signals.
- `gui/audio_converter.py` — `convert_to_wav()`, `is_supported_format()`. Uses ffmpeg to convert to 16kHz mono PCM.
- `gui/markdown_export.py` — `segments_to_markdown()`, `merge_consecutive_same_speaker()`.
- `gui/youtube_download.py` — `download_youtube_audio()`, `is_youtube_url()`. Uses yt-dlp.

**CLI** (`transcribe_simple.py` `main()`):
- Reads `sys.argv[1]` for audio file, `NUM_SPEAKERS`/`TEST_MODE`/`TRANSCRIPTION_LANGUAGE` env vars.
- Prints progress and segments to console.

**API** (`service/api.py`):
- FastAPI app with `/health`, `/jobs` (POST), `/jobs/{id}` (GET), `/jobs/{id}/result` (GET), `/jobs/{id}/cancel` (POST), `/admin/warmup` (POST).
- Bearer token auth via `SERVICE_API_TOKEN`.
- Lifespan handler: `storage.init_db()`, `storage.recover_stale_running_jobs()`, `run_preflight()`, `jobs.start_worker()`.

### Application Layer

**`service/jobs.py`** — Background worker:
- `start_worker()` / `stop_worker()` — manages a `ThreadPoolExecutor(max_workers=1)` and a daemon `threading.Thread` scheduler.
- `_worker_loop()` — polls `storage.try_claim_next_job()` every 0.25s, submits `_run_transcription_job()` to the executor, enforces `SERVICE_JOB_TIMEOUT_SEC` via `future.result(timeout=...)`.
- `_run_transcription_job()` — calls `transcribe_simple.transcribe_audio()` with progress/cancel callbacks, writes transcript via `_write_transcript()`, updates DB via `storage.mark_succeeded()` / `storage.mark_failed()`.
- `validate_audio_path()` — path traversal protection: resolves path and ensures it's under `SERVICE_DATA_DIR`.

**`service/storage.py`** — SQLite job store:
- Module-level `sqlite3.Connection` with `check_same_thread=False`, guarded by `threading.Lock`.
- `init_db()` — creates `jobs` and `job_events` tables, WAL mode.
- Job lifecycle: `create_job()` → `try_claim_next_job()` (atomic `BEGIN IMMEDIATE`) → `update_progress()` → `mark_succeeded()` / `mark_failed()` / `mark_cancelled()`.
- `recover_stale_running_jobs()` — marks interrupted `running` jobs as `failed` after restart.

### Core Layer

**`transcribe_simple.py`** — The shared transcription pipeline:

`transcribe_audio()` orchestrator:
1. **Load diarization pipeline** — `load_diarization_pipeline(token)` loads pyannote `Pipeline.from_pretrained()`, with fallback model and `token`/`use_auth_token` keyword compatibility.
2. **Load audio** — ffmpeg subprocess → `np.frombuffer` → `torch.from_numpy` → `torch.Tensor` (16kHz mono).
3. **Run diarization** — `pipeline({"waveform": ..., "sample_rate": 16000}, hook=ProgressHook(), num_speakers=...)`. Extracts `annotation` (speaker labels per time segment).
4. **Transcribe** — either:
   - `_transcribe_with_parakeet()` — NeMo `ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")`, chunks audio into 4-minute segments, transcribes each.
   - `_transcribe_with_whisper()` — faster-whisper or openai-whisper, same chunking.
5. **Merge** — `_SpeakerTurnMerger` (Whisper path) or `_append_segment` (Parakeet path) accumulates same-speaker turns.
6. **Sort** — segments sorted by start time.

Key abstractions:
- **`CustomProgressHook`** — intercepts pyannote `ProgressHook` step updates, maps them to weighted overall progress (diarization phase = 0.25–0.30 of total).
- **`_SpeakerTurnMerger`** — accumulates short ASR segments into continuous speaker turns before emitting.
- **`_iter_audio_chunks()`** — yields 4-minute chunks with 3-second overlap.
- **`_write_temp_wav()`** — writes chunk to temp file for ASR model input.

### Infrastructure Layer

- `service/config.py` — `Settings` class, loads all config from env vars (`SERVICE_HOST`, `SERVICE_PORT`, `SERVICE_DATA_DIR`, `SERVICE_OUTPUT_DIR`, `SERVICE_DB_PATH`, `SERVICE_CACHE_ROOT`, `SERVICE_MAX_QUEUED_JOBS`, `SERVICE_JOB_TIMEOUT_SEC`, `SERVICE_API_TOKEN`, `SERVICE_REQUIRE_CUDA`). `settings` singleton.
- `service/schemas.py` — Pydantic models: `JobStatus`, `CreateJobRequest`, `CreateJobResponse`, `JobStatusResponse`, `TranscriptSegment`, `JobResultResponse`, `HealthResponse`, `WarmupResponse`.
- `service/preflight.py` — `check_ffmpeg()`, `check_huggingface_token()`, `cuda_info()`, `run_preflight()`.

## Data Flow

### Primary Request Path (Microservice)

1. **`POST /jobs`** (`service/api.py:124`) — validates `audio_path` via `jobs.validate_audio_path()`, checks queue depth, calls `storage.create_job()` → returns `job_id`.
2. **Worker scheduler** (`service/jobs.py:116`) — `_worker_loop()` polls `storage.try_claim_next_job()` (atomic `BEGIN IMMEDIATE` → `UPDATE jobs SET status='running'`), gets `job_id`.
3. **Transcription** (`service/jobs.py:62`) — `_run_transcription_job()` calls `transcribe_simple.transcribe_audio()` with:
   - `progress_cb` → `storage.update_progress()`
   - `check_interrupt` → `storage.cancel_requested()`
   - `segment_callback=None` (service doesn't stream segments)
4. **Output** (`service/jobs.py:50`) — `_write_transcript()` writes merged segments to `SERVICE_OUTPUT_DIR/{stem}_transcript_{job_id}.txt`.
5. **DB update** (`service/jobs.py:106`) — `storage.mark_succeeded()` stores output path + segments JSON.
6. **`GET /jobs/{id}/result`** (`service/api.py:186`) — returns `JobResultResponse` with `TranscriptSegment` list.

### GUI Flow

1. **`gui_main.py`** — loads `.env`, patches `torch.load`, creates `QApplication`, shows `MainWindow`.
2. **`MainWindow._load_pipeline()`** (`gui/main_window.py:539`) — deferred via `QTimer.singleShot(100)`, loads pyannote pipeline in main thread (avoids PyTorch threading issues), moves to CUDA/MPS/CPU.
3. **`MainWindow._on_start_clicked()`** (`gui/main_window.py:646`) — optionally downloads YouTube audio via yt-dlp, creates `TranscriptionWorker` with pre-loaded pipeline.
4. **`TranscriptionWorker.run()`** (`gui/transcription_worker.py:82`) — converts non-WAV audio via `convert_to_wav()`, calls `transcribe_audio()` with signal-emitting callbacks.
5. **Streaming** — `segment_callback` → `segment_ready` signal → `MainWindow._on_segment_ready()` appends to transcript display with auto-scroll.
6. **Completion** — `finished` signal → `MainWindow._on_finished()` → `merge_consecutive_same_speaker()` → `_auto_save_transcript()` (txt or markdown).

### CLI Flow

1. **`transcribe_simple.py main()`** — reads env vars, calls `transcribe_audio()` with console `progress_callback` and `segment_callback`.
2. **Output** — `merge_consecutive_same_speaker()` → writes `{stem}_transcript.txt` next to source file.

**State Management:**
- GUI: `MainWindow` instance attributes (`_current_audio_file`, `_transcript_segments`, `_is_processing`, `_pipeline`, `_worker`).
- Service: SQLite `jobs` table (status, progress, error, output_path, result_segments_json, timestamps) + `job_events` table (audit trail). Module-level `sqlite3.Connection` with `threading.Lock`.
- CLI: local variables, no persistent state.

## Key Abstractions

**`transcribe_audio()`** (`transcribe_simple.py:800`):
- Purpose: Main transcription orchestrator. Accepts injectable `progress_callback`, `segment_callback`, `check_interrupt`, and `pipeline` (pre-loaded to avoid threading issues).
- Returns: `List[Dict]` of `{start, end, speaker, text}`.
- Pattern: Callback-based streaming — all three frontends inject different callbacks.

**`Pipeline` (pyannote)**:
- Purpose: Speaker diarization. Loaded once via `Pipeline.from_pretrained()`.
- Examples: `gui/main_window.py:559` (GUI), `transcribe_simple.py:871` (CLI/service).
- Pattern: Pre-loaded in main thread for GUI; loaded in worker for CLI/service.

**`_SpeakerTurnMerger`** (`transcribe_simple.py:473`):
- Purpose: Accumulate short ASR segments into continuous speaker turns.
- Pattern: Stateful accumulator with `add()` / `flush()`.

**`CustomProgressHook`** (`transcribe_simple.py:152`):
- Purpose: Intercept pyannote `ProgressHook` step updates, map to weighted overall progress.
- Pattern: Callable class with `__enter__`/`__exit__`/`__call__`.

**`TranscriptionWorker(QThread)`** (`gui/transcription_worker.py:39`):
- Purpose: Run transcription in background without blocking GUI.
- Pattern: Qt signal/slot — `progress`, `segment_ready`, `finished`, `error` signals.

## Entry Points

**`gui_main.py`** (`gui_main.py:1`):
- Triggers: `python gui_main.py`
- Responsibilities: QApplication setup, pipeline pre-loading, MainWindow lifecycle.

**`transcribe_simple.py`** (`transcribe_simple.py:1055`):
- Triggers: `python transcribe_simple.py [audio_file]`
- Responsibilities: CLI transcription with console output.

**`service/__main__.py`** (`service/__main__.py:1`):
- Triggers: `python -m service` or `uvicorn service.api:app`
- Responsibilities: Uvicorn server startup.

**`service/api.py`** (`service/api.py:85`):
- Triggers: HTTP requests to FastAPI app
- Responsibilities: API endpoints, auth, lifespan management.

**`Dockerfile`** (`Dockerfile:1`):
- Triggers: `docker build` / `docker compose up`
- Responsibilities: CUDA 12.4 base image, pip install with constraints, `CMD uvicorn service.api:app`.

## Architectural Constraints

- **Threading (GUI):** Pipeline loaded in main thread (PyTorch threading issues). Transcription runs in `QThread` worker. `TranscriptionWorker.requestInterruption()` for cooperative cancellation.
- **Threading (Service):** `ThreadPoolExecutor(max_workers=1)` — single job at a time. Daemon scheduler thread polls every 0.25s. SQLite guarded by `threading.Lock` with `check_same_thread=False`.
- **Global state:** `service/storage.py` — module-level `_conn` (SQLite connection) and `_db_lock` (threading.Lock). `transcribe_simple.py` — module-level `_WHISPER_MODEL_CACHE` and `_FASTER_WHISPER_MODEL_CACHE` dicts.
- **torch.load patch:** Both `gui_main.py` and `transcribe_simple.py` patch `torch.load` to set `weights_only=False` when `None` is passed (PyTorch 2.6+ compatibility with Lightning/pyannote checkpoints).
- **MPS float64 patch:** `transcribe_simple.py` patches `nemo_data_utils.move_data_to_device` to convert `float64` → `float32` before moving to MPS device.
- **Cache isolation:** `gui_main.py` and `transcribe_simple.py` redirect `XDG_CACHE_HOME`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE` to repo-local `./.cache`. `service/api.py` redirects to `SERVICE_CACHE_ROOT`.
- **Path traversal:** `service/jobs.py:validate_audio_path()` resolves path and ensures it's under `SERVICE_DATA_DIR` using `Path.relative_to()`.
- **CUDA constraints:** `scripts/docker_constraints.txt` pins `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`, `torchaudio==2.5.1+cu124` to match the `pytorch/pytorch:2.5.1-cuda12.4` base image.
- **Single worker:** Service runs one transcription job at a time (`max_workers=1`). No job parallelization.

## Anti-Patterns

### Duplicate `merge_consecutive_same_speaker`

**What happens:** `merge_consecutive_same_speaker()` is defined in both `transcribe_simple.py:1017` and `gui/markdown_export.py:15`. The GUI imports from `gui.markdown_export`, the service/CLI use the one in `transcribe_simple.py`.

**Why it's wrong:** Code duplication — changes to the merge logic must be applied in two places. The two implementations are nearly identical.

**Do this instead:** Extract to a single shared utility module (e.g., `gui/markdown_export.py` or a new `utils.py`) and import from one place.

### `torch.load` monkeypatch duplicated

**What happens:** Both `gui_main.py:34` and `transcribe_simple.py:51` contain the same `torch.load` patching logic with the same `_speaker_sep_torch_load_patched` guard.

**Why it's wrong:** Duplication of a compatibility shim. If the patch logic changes, both copies must be updated.

**Do this instead:** Extract to a shared module (e.g., `torch_compat.py`) imported by both entry points.

### `load_diarization_pipeline` / `_load_pyannote_pipeline` duplication

**What happens:** `transcribe_simple.py:90` (`load_diarization_pipeline`) and `gui/main_window.py:466` (`_load_pyannote_pipeline`) contain nearly identical pyannote model-loading logic with `token`/`use_auth_token` fallback and model candidate fallback.

**Why it's wrong:** Same model-loading logic in two places. The GUI version is a copy of the backend version.

**Do this instead:** Have the GUI call `load_diarization_pipeline()` from `transcribe_simple.py` instead of maintaining its own copy.

## Error Handling

**Strategy:** Fail-fast with exceptions for setup errors (missing token, pipeline load failure), cooperative cancellation for user-initiated stops, and graceful degradation for optional components.

**Patterns:**
- **Pipeline load failure** (`transcribe_simple.py:875`): raises `RuntimeError`, caught in `jobs.py:110` → `storage.mark_failed()`.
- **Chunk-level failure** (`transcribe_simple.py:684`): logs error, continues to next chunk. Job still succeeds with partial segments.
- **Cancellation** (`transcribe_simple.py:84`): `check_interrupt()` checked between chunks; `jobs.py:73` checks `storage.cancel_requested()`.
- **Timeout** (`jobs.py:131`): `FutureTimeoutError` → `storage.mark_failed()` with timeout message. GPU work may continue in background.
- **Stale recovery** (`storage.py:322`): `recover_stale_running_jobs()` marks interrupted jobs as failed on restart.
- **Auth** (`api.py:54`): `require_agent_auth` dependency — 401 if no bearer, 403 if wrong token.

## Cross-Cutting Concerns

**Logging:**
- `transcribe_simple.py` — module-level logger `transcribe_simple` at WARNING level.
- `gui_main.py` / `gui/*` — separate loggers (`main_window`, `transcription_worker`) at WARNING level.
- `service/*` — `logging.basicConfig(level=SERVICE_LOG_LEVEL)` with `%(asctime)s %(levelname)s %(name)s` format.
- NeMo loggers suppressed to WARNING (`nemo`, `nemo.collections.asr`, etc.).

**Validation:**
- `service/jobs.py:validate_audio_path()` — path traversal protection.
- `service/schemas.py` — Pydantic field validation (`num_speakers` ge=1, `progress` ge=0.0 le=1.0).
- `gui/audio_converter.py:is_supported_format()` — extension whitelist.

**Authentication:**
- Service: optional bearer token (`SERVICE_API_TOKEN`). `GET /health` unauthenticated. All `/jobs*` and `/admin/warmup` require auth.
- GUI/CLI: `HUGGINGFACE_TOKEN` from `.env` for model access.

---

*Architecture analysis: 2026-07-30*
