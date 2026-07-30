---
generated: 2026-07-30
---
# Technology Stack

## Language and Runtime

**Primary:**
- **Python 3.11** — Used throughout the project. Referenced in `scripts/setup_and_run.sh` (`conda create -n speaker_separation python=3.11`).

**Runtime:**
- **Local (GUI/CLI):** CPython interpreter with conda environment `speaker_separation`.
- **Service (Docker):** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` base image (`Dockerfile` line 4). CUDA 12.4, cuDNN 9.

**Package Manager:**
- **pip** — Primary installer. Lockfile: not present (no `requirements.lock` or `poetry.lock`).
- **uv** — Alternative preferred by project conventions (per `AGENTS.md`): `UV_CACHE_DIR=.uv-cache` for tight permissions.
- **conda** — Used in `scripts/setup_and_run.sh` and `scripts/run_transcription.sh` to activate the `speaker_separation` environment.

## Frameworks and Libraries

### Core Application

| Framework | Version | Purpose | Location |
|-----------|---------|---------|----------|
| **PySide6** | `>=6.5.0` | Desktop GUI (QML-free, QtWidgets) | `gui_main.py`, `gui/` |
| **FastAPI** | `>=0.115.0` | Async transcription microservice (agent-facing API) | `service/api.py` |
| **Uvicorn** | `>=0.30.0` | ASGI server for FastAPI | `service/__main__.py`, `Dockerfile` CMD |
| **Pydantic** | `>=2.0` | Request/response schema validation | `service/schemas.py` |
| **SQLite3** | stdlib | Durable job store for the microservice | `service/storage.py` |

### ML / AI Stack

| Library | Version | Purpose | Location |
|---------|---------|---------|----------|
| **pyannote.audio** | `>=3.1.1,<4.0` | Speaker diarization pipeline | `transcribe_simple.py`, `gui/main_window.py` |
| **openai-whisper** | `>=20231117` | Transcription (MPS/CPU fallback) | `transcribe_simple.py` |
| **faster-whisper** | `>=1.1.0` | Optimized Whisper (CUDA/CPU via CTranslate2) | `transcribe_simple.py` |
| **nemo_toolkit[asr]** | — | NeMo Parakeet ASR model runtime | `transcribe_simple.py` |
| **torch** | `>=2.0.0` (local) / `==2.5.1+cu124` (Docker) | PyTorch tensor ops, MPS/CUDA acceleration | `transcribe_simple.py`, `gui/main_window.py` |
| **torchaudio** | `>=2.0.0` (local) / `==0.20.1+cu124` (Docker) | Audio tensor processing | `transcribe_simple.py` |
| **soundfile** | `>=0.12.1` | WAV read/write (libsndfile backend) | `transcribe_simple.py` |
| **huggingface-hub** | `>=0.20.0` | Model download and cache management | `transcribe_simple.py`, `scripts/download_models.py` |
| **numpy** | `<2` | Numerical operations (pinned <2 for pyannote 3.1 compat) | `scripts/requirements_transcription.txt` |

### External CLI Tools

| Tool | Purpose | Invoked From |
|------|---------|-------------|
| **ffmpeg** | Audio format conversion to 16kHz mono PCM WAV | `gui/audio_converter.py`, `service/preflight.py` |
| **yt-dlp** | YouTube audio download | `gui/youtube_download.py` |
| **huggingface-cli** | Model download (script-based) | `scripts/fix_models.sh` |

### GUI Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **pyside6** | `>=6.5.0` | Qt-based desktop application |
| **yt-dlp** | — | YouTube audio download (GUI feature) |
| **python-dotenv** | — | `.env` file loading |

## Dependencies

### Requirements Files

| File | Scope | Key Packages |
|------|-------|-------------|
| `requirements_gui.txt` | GUI application | pyside6, yt-dlp, python-dotenv |
| `scripts/requirements_service.txt` | Microservice | fastapi, uvicorn[standard], python-dotenv, pydantic |
| `scripts/requirements_ml.txt` | ML stack (Docker) | pyannote.audio, openai-whisper, nemo_toolkit[asr], soundfile, huggingface-hub |
| `scripts/requirements_transcription.txt` | Local transcription | pyannote.audio, openai-whisper, faster-whisper, nemo_toolkit[asr], numpy<2, torch, torchaudio, soundfile, huggingface-hub |

### Key Version Constraints

- **pyannote.audio**: `>=3.1.1,<4.0` — 4.x pulls stacks that may upgrade PyTorch away from the CUDA base image.
- **numpy**: `<2` — pyannote 3.1.x references `np.NaN` (removed in NumPy 2).
- **torch/torchaudio/torchvision**: Pinned to `==2.5.1+cu124` / `==0.20.1+cu124` via `scripts/docker_constraints.txt` to match the CUDA 12.4 runtime base image.
- **PySide6**: `>=6.5.0`

## Configuration

### Environment Variables

**Required:**
- `HUGGINGFACE_TOKEN` — Hugging Face API token with access to pyannote models. Set in `.env` (gitignored). Referenced in `transcribe_simple.py:843`, `gui_main.py:64`, `service/preflight.py:16`.

**Optional:**
- `SERVICE_HOST` (default `0.0.0.0`) — API bind address (`service/config.py:33`).
- `SERVICE_PORT` (default `8080`) — API port (`service/config.py:34`).
- `SERVICE_DATA_DIR` (default `/data`) — Root for input audio, path traversal guard (`service/config.py:37`).
- `SERVICE_OUTPUT_DIR` (default `/exports`) — Transcript output directory (`service/config.py:40`).
- `SERVICE_DB_PATH` (default `/var/lib/transcription/jobs.db`) — SQLite jobs DB (`service/config.py:43`).
- `SERVICE_CACHE_ROOT` (default `/cache`) — HF/torch caches (`service/config.py:48`).
- `SERVICE_MAX_QUEUED_JOBS` (default `16`) — Max queued jobs (`service/config.py:50`).
- `SERVICE_JOB_TIMEOUT_SEC` (default `14400`) — Per-job future timeout (`service/config.py:51`).
- `SERVICE_API_TOKEN` — Optional bearer token for job/admin endpoints (`service/config.py:54`).
- `SERVICE_REQUIRE_CUDA` (default `true`) — Fail startup if CUDA unavailable (`service/config.py:57`).
- `SERVICE_LOG_LEVEL` (default `INFO`) — Python logging level (`service/api.py:32`).
- `WHISPER_MODEL` (default `turbo`) — Whisper model size (`transcribe_simple.py:77`).
- `WHISPER_BACKEND` (default `auto`) — `auto`, `faster-whisper`, or `openai-whisper` (`transcribe_simple.py:78`).
- `FASTER_WHISPER_COMPUTE_TYPE` — Override compute type for faster-whisper (`transcribe_simple.py:559`).
- `NUM_SPEAKERS` — Force specific speaker count (CLI).
- `TEST_MODE` — Process only first 60 seconds (CLI).
- `PYANNOTE_METRICS_ENABLED` — Set to `0` to disable telemetry (`transcribe_simple.py:32`).
- `PYTORCH_ENABLE_MPS_FALLBACK` — Set to `1` for NeMo MPS fallback (`transcribe_simple.py:33`).
- `NEMO_LOGGING_LEVEL` — Set to `WARNING` (`transcribe_simple.py:36`).

### Cache Configuration

Both `gui_main.py` and `transcribe_simple.py` redirect Hugging Face caches to a repo-local `./.cache` tree to avoid home-directory permission issues:

```python
# gui_main.py:26-28, transcribe_simple.py:28-30
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
os.environ.setdefault("HF_HOME", str(_hf_home))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_hf_hub_cache))
```

The service (`service/api.py:38-48`) redirects caches to `SERVICE_CACHE_ROOT` for Docker volumes.

### PyTorch Compatibility Patch

Both `gui_main.py` and `transcribe_simple.py` patch `torch.load` to set `weights_only=False` when `None` is passed, matching Lightning/pyannote expectations under PyTorch 2.6+:

```python
# gui_main.py:34-45, transcribe_simple.py:51-62
def _torch_load_compat(*args, **kwargs):
    if kwargs.get("weights_only", None) is None:
        kwargs = {**kwargs, "weights_only": False}
    return _orig_torch_load(*args, **kwargs)
```

### Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables (gitignored, never commit). Exists at repo root. |
| `.env.example` | Template for `.env` (`HUGGINGFACE_TOKEN`, service tuning vars) |
| `docker-compose.yml` | Docker service definition with volumes, GPU, env vars |
| `Dockerfile` | CUDA-enabled transcription microservice image |
| `scripts/docker_constraints.txt` | Pin torch/torchvision/torchaudio to CUDA 12.4 wheels |
| `AGENTS.md` | Project-specific preferences and workspace facts |

## Platform Requirements

### Development

- **Python**: 3.11 (via conda or uv)
- **ffmpeg**: Must be on PATH (audio conversion)
- **Hugging Face token**: Required for model access (gated pyannote models)
- **GPU**: Apple Silicon MPS (optimized), NVIDIA CUDA (service/Docker), or CPU (fallback)
- **Conda environment**: `speaker_separation` (created via `scripts/setup_and_run.sh`)

### Production (Docker)

- **Base image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- **NVIDIA Container Toolkit**: Required on host for GPU access
- **Volumes**: `transcription_cache` (HF/torch caches), `transcription_db` (SQLite jobs DB)
- **Ports**: `8080:8080` (FastAPI service)
- **Healthcheck**: `curl -fsS http://127.0.0.1:8080/health` (interval 30s, start-period 120s)

---

*Stack analysis: 2026-07-30*
