# Transcription microservice (CUDA + Docker)

Async job API for LLM agents. The worker runs `transcribe_simple.transcribe_audio` with pyannote diarization and NeMo Parakeet ASR.

## Run (Docker Compose)

The image pins `torch` / `torchaudio` / `torchvision` to CUDA 12.4 wheels via [scripts/docker_constraints.txt](../scripts/docker_constraints.txt) so `pip` does not replace the base image’s CUDA stack.

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.
2. Accept Hugging Face model terms for [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1).
3. Create `.env` from [.env.example](../.env.example) with `HUGGINGFACE_TOKEN`.
4. Create directories and place audio under `./data`:

```bash
mkdir -p data exports
docker compose up --build
```

5. Submit a job with a path **inside the container** (mapped from host `./data`):

- Host file: `./data/meeting.wav`
- Request `audio_path`: `/data/meeting.wav`

## Authentication

If `SERVICE_API_TOKEN` is set, all `/jobs` routes and `POST /admin/warmup` require:

```http
Authorization: Bearer <SERVICE_API_TOKEN>
```

`GET /health` is unauthenticated (for orchestrators and health checks).

## Endpoints

### `GET /health`

Returns JSON with `ok`, `cuda_available`, `ffmpeg_ok`, `huggingface_token_configured`, and paths. HTTP **503** when unhealthy (e.g. CUDA required but missing).

### `POST /jobs`

Create a transcription job.

**Body (JSON)**

| Field | Type | Required | Description |
|--------|------|----------|-------------|
| `audio_path` | string | yes | Absolute path under `SERVICE_DATA_DIR` (e.g. `/data/file.wav`) |
| `num_speakers` | int | no | Fix speaker count; omit for auto-detect |
| `test_mode` | bool | no | If true, only first **60 seconds** are processed |

**Response**

```json
{ "job_id": "uuid", "status": "queued" }
```

**Errors**

- `400` — path outside data dir, or file missing
- `429` — queue full (`SERVICE_MAX_QUEUED_JOBS`)
- `401` / `403` — auth when token is configured

### `GET /jobs/{job_id}`

Poll job status.

**Response fields**

- `status`: `queued` | `running` | `succeeded` | `failed` | `cancelled`
- `progress`: 0.0–1.0
- `message`, `error`, timestamps, `output_path` when done

**Polling**: agents should poll every 2–10s while `queued` or `running`.

### `GET /jobs/{job_id}/result`

Returns merged segments and metadata when `status === succeeded`.

**Response**

- `segments`: list of `{ start, end, speaker, text }`
- `output_path`: transcript text file on the service filesystem
- `409` if job is not yet succeeded

### `POST /jobs/{job_id}/cancel`

- If `queued`: job moves to `cancelled`
- If `running`: `cancel_requested` is set; worker stops cooperatively when possible

### `POST /admin/warmup`

Lightweight CUDA visibility check (no full model preload). Auth required if `SERVICE_API_TOKEN` is set.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | — | Required |
| `SERVICE_DATA_DIR` | `/data` | Root for input audio (path traversal guard) |
| `SERVICE_OUTPUT_DIR` | `/exports` | Transcript output directory |
| `SERVICE_DB_PATH` | `/var/lib/transcription/jobs.db` | SQLite jobs DB |
| `SERVICE_CACHE_ROOT` | `/cache` | HF / torch caches (use a volume) |
| `SERVICE_MAX_QUEUED_JOBS` | `16` | Max queued jobs |
| `SERVICE_JOB_TIMEOUT_SEC` | `14400` | Per-job future timeout (see note below) |
| `SERVICE_REQUIRE_CUDA` | `true` | Fail startup if CUDA unavailable |
| `SERVICE_API_TOKEN` | unset | Optional bearer token |

**Timeout note**: `SERVICE_JOB_TIMEOUT_SEC` uses `concurrent.futures` timeout. If it fires, the job is marked `failed` in the DB, but long GPU work may still run until completion in the background; avoid overlapping conflicting runs on a single GPU or scale timeouts.

## Agent integration checklist

1. `GET /health` until `ok: true`.
2. `POST /jobs` with container `audio_path`.
3. Poll `GET /jobs/{id}` until terminal state.
4. On `succeeded`, `GET /jobs/{id}/result` (or read `output_path` from the shared volume).

## Local run (without Docker)

From repo root, with CUDA/CPU stack installed:

```bash
export HUGGINGFACE_TOKEN=...
export SERVICE_REQUIRE_CUDA=false   # if no GPU
pip install -r scripts/requirements_ml.txt -r scripts/requirements_service.txt
uvicorn service.api:app --host 0.0.0.0 --port 8080
```

Point `SERVICE_DATA_DIR` at a directory containing your audio files.
