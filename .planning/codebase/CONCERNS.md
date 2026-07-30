---
generated: 2026-07-30
---
# Codebase Concerns

## Technical Debt

### Duplicated `merge_consecutive_same_speaker` function

**What happens:** The `merge_consecutive_same_speaker` function is defined identically in two locations: `transcribe_simple.py` (line 1017) and `gui/markdown_export.py` (line 15). Both implementations are byte-for-byte identical.

**Why it's wrong:** Any fix or enhancement to one copy is not reflected in the other. The `main_window.py` imports it from `gui/markdown_export.py` (line 29), while `transcribe_simple.py` uses its own copy. This creates a maintenance hazard where the two copies can silently diverge.

**Do this instead:** Extract the function into a shared utility module (e.g., `gui/markdown_export.py` or a new `utils.py`) and import it from both locations.

### Duplicated pyannote pipeline loading logic

**What happens:** The pyannote pipeline loading with API compatibility fallback (`token=` vs `use_auth_token=`, model candidate fallback) is duplicated in `transcribe_simple.py` (`load_diarization_pipeline`, line 90) and `gui/main_window.py` (`_load_pyannote_pipeline`, line 466). Both functions are ~70 lines and nearly identical.

**Why it's wrong:** Changes to the compatibility logic (e.g., adding a new fallback model or handling a new pyannote API change) must be made in two places. The two copies can diverge, leading to inconsistent behavior between CLI and GUI.

**Do this instead:** Extract the shared logic into a single function in a common module that both CLI and GUI import.

### Dead code: `CustomProgressHook` class

**What happens:** The `CustomProgressHook` class is defined in `transcribe_simple.py` (line 152) but never instantiated or used. The `transcribe_audio` function uses pyannote's built-in `ProgressHook` instead (line 964), with a comment at line 962-963 stating "Use original ProgressHook for console output (shows beautiful progress bars)" and "No GUI updates during diarization."

**Why it's wrong:** The class is ~130 lines of code that provides no value. It was likely intended to forward diarization step progress to the GUI callback, but the implementation was abandoned in favor of the simpler `ProgressHook`. The GUI receives no granular progress updates during the diarization phase (progress stays at 0.25–0.30).

**Do this instead:** Either implement the GUI progress forwarding using `CustomProgressHook` or remove the dead code.

### Stray empty file `=0.12.1`

**What happens:** An empty file named `=0.12.1` exists at the repository root. It is not tracked by git (not in `.gitignore` but appears to be untracked).

**Why it's wrong:** This file is a side effect of a botched `pip install` command (e.g., `pip install pyannote.audio=0.12.1` instead of `pyannote.audio==0.12.1`). The `=` character in the filename causes pip to interpret everything after `=` as a separate argument, creating a file named `=0.12.1`.

**Do this instead:** Delete the file. Add it to `.gitignore` if it might recur.

### Legacy archive script duplicates current functionality

**What happens:** `archive/transcribe_with_speakers.py` is an 810-line script that duplicates much of `transcribe_simple.py`'s functionality (audio loading, diarization, Whisper transcription, segment merging). The archive script uses an older approach (e.g., `torchaudio.load` for audio, different Whisper backend selection).

**Why it's wrong:** The archive script is not referenced by any current code path. It adds noise to the repository and could confuse developers about which script to use. The `.gitignore` has a commented-out `# archive/` entry, suggesting the archive directory was intended to be optional.

**Do this instead:** Either remove the archive directory entirely or ensure it is clearly marked as historical reference only.

### Incomplete `requirements_gui.txt`

**What happens:** `requirements_gui.txt` contains only 4 lines: `pyside6`, `yt-dlp`, `python-dotenv`. It does not include `pyannote.audio`, `openai-whisper`, `faster-whisper`, `nemo_toolkit`, `torch`, `numpy`, `soundfile`, or `huggingface-hub`.

**Why it's wrong:** The AGENTS.md (line 5) states "For local GUI and transcription runs, install from `requirements_gui.txt` and `scripts/requirements_transcription.txt`." However, `requirements_gui.txt` alone is insufficient to run the GUI — the GUI imports `transcribe_simple` which requires the full ML stack. A developer who only installs `requirements_gui.txt` will get `ImportError` when launching `gui_main.py`.

**Do this instead:** Either consolidate the requirements into a single file or add a comment in `requirements_gui.txt` explaining that `scripts/requirements_transcription.txt` must also be installed.

### Module-level `logger` referenced before definition in `load_diarization_pipeline`

**What happens:** In `transcribe_simple.py`, the `logger` variable is defined at line 331 (`logger = logging.getLogger('transcribe_simple')`), but `load_diarization_pipeline` (line 90) references `logger` at lines 122 and 134. Similarly, `_load_whisper_model` (line 520) references `logger` at line 526, and `_select_whisper_backend` (line 535) references `logger` at line 538.

**Why it's wrong:** This works in practice because Python resolves names at runtime, and `logger` is defined at module level before any function is called. However, if any code were to call `load_diarization_pipeline` during module import (before line 331 is reached), it would raise `NameError`. This is a latent bug and poor code organization.

**Do this instead:** Move the `logger` definition to the top of the module, before any function definitions.

### Hardcoded default audio filename in CLI

**What happens:** `transcribe_simple.py` line 1061: `audio_file = sys.argv[1] if len(sys.argv) > 1 else "Music Company Media Productions 10.m4a"`. The default filename is a specific audio file that may not exist on the user's system.

**Why it's wrong:** If no argument is provided and the default file doesn't exist, the script fails with a `FileNotFoundError` from `ffmpeg`. The default should be a usage message or a clear error.

**Do this instead:** If no argument is provided, print a usage message and exit with a non-zero status code.

### Missing `/jobs` list endpoint in service API

**What happens:** The service API (`service/api.py`) has endpoints for creating jobs (`POST /jobs`), getting a job (`GET /jobs/{job_id}`), getting results (`GET /jobs/{job_id}/result`), and canceling jobs (`POST /jobs/{job_id}/cancel`). However, there is no `GET /jobs` endpoint to list all jobs, and no `GET /jobs/{job_id}/events` endpoint to retrieve job event history.

**Why it's wrong:** The `storage.py` module has `list_events(job_id)` (line 350) and `count_queued_and_running()` (line 93) functions that are not exposed via the API. LLM agents using the service cannot enumerate jobs or view event history.

**Do this instead:** Add `GET /jobs` (list all jobs) and `GET /jobs/{job_id}/events` (get event history) endpoints to `service/api.py`.

## Bugs and Issues

### `torch.load` monkey-patch is process-wide and fragile

**What happens:** `transcribe_simple.py` (lines 51-62) and `gui_main.py` (lines 34-45) both monkey-patch `torch.load` to set `weights_only=False` when `weights_only` is `None`. This patch is applied at module import time and affects the entire Python process.

**Why it's wrong:** The patch overrides a core PyTorch function globally. If any other library in the same process relies on the default `weights_only=True` behavior (the PyTorch 2.6+ default for security), it will silently receive `weights_only=False` instead, potentially loading untrusted pickle data. The patch is also applied twice (once in `transcribe_simple.py` and once in `gui_main.py`), though the `_speaker_sep_torch_load_patched` guard prevents double-patching.

**Do this instead:** Use `torch.load(path, weights_only=False)` explicitly at each call site that needs it, rather than patching the global function. Alternatively, scope the patch to only the diarization pipeline loading.

### MPS float64→float32 patch modifies NeMo internals

**What happens:** `transcribe_simple.py` (lines 342-368) patches `nemo_data_utils.move_data_to_device` with a custom `_mps_safe_move_data_to_device` function that converts float64 tensors to float32 before moving to MPS.

**Why it's wrong:** The patch replaces a NeMo internal function with a custom implementation that may not handle all edge cases (e.g., custom dataclass types, nested structures not covered by the type checks). If NeMo changes the signature or behavior of `move_data_to_device` in a future version, the patch will silently break.

**Do this instead:** Consider using `torch.set_default_dtype(torch.float32)` (already done at line 64) more aggressively, or file an upstream fix with NeMo.

### Service job timeout leaves GPU work running

**What happens:** In `service/jobs.py` (line 130), `future.result(timeout=settings.job_timeout_sec)` raises `FutureTimeoutError` when a job exceeds the timeout. The code catches this and calls `storage.mark_failed(job_id, ...)` (line 133), but the underlying transcription work continues in the background because `ThreadPoolExecutor` cannot cancel a running future.

**Why it's wrong:** The job is marked as failed in the database, but the GPU is still processing the audio. If the service receives another job, it will be queued behind the still-running (but marked-failed) job, because the `ThreadPoolExecutor` has `max_workers=1` (line 149). This can lead to resource exhaustion and confusing behavior for clients.

**Do this instead:** Use a subprocess for each job so that timeout can kill the process, or use `multiprocessing` with a timeout that actually terminates the worker.

### `DiarizeOutput` vs `Annotation` handling is fragile

**What happens:** `transcribe_simple.py` (lines 971-974) checks `hasattr(diarization, 'speaker_diarization')` to handle the new pyannote `DiarizeOutput` format. The same check appears in `archive/transcribe_with_speakers.py` (line 101), `tests/test_diarization_only.py` (line 59), `tests/test_gpu_integration.py` (line 89), and `tests/test_steps.py` (line 100).

**Why it's wrong:** This attribute-based check is a workaround for pyannote API changes. If pyannote changes the attribute name or structure in a future version, all these checks will fail silently, returning the wrong type.

**Do this instead:** Use `isinstance` checks or a version-aware adapter pattern.

### Pipeline loading failure in GUI is not surfaced to user

**What happens:** In `gui/main_window.py`, `_load_pipeline` (line 539) loads the pyannote pipeline asynchronously after the GUI is shown. If loading fails, the error is logged and shown in the status bar (line 578), but no dialog is shown. The user only discovers the failure when they click "Start" and the pipeline is `None` and `_pipeline_loading` is `False` (line 724).

**Why it's wrong:** The user may wait for the pipeline to load, then click "Start" and see a critical error dialog. This is a poor UX. The pipeline loading failure should be surfaced immediately.

**Do this instead:** Show a `QMessageBox.critical` when pipeline loading fails in `_load_pipeline`, similar to the error handling in `_on_start_clicked`.

### `fix_models.sh` uses unsafe `export $(grep ... | xargs)`

**What happens:** `scripts/fix_models.sh` (line 37) uses `export $(grep -v '^#' .env | xargs)` to load environment variables from `.env`.

**Why it's wrong:** This is a shell injection vulnerability. If `.env` contains a line like `FOO=bar; rm -rf /`, the `xargs` command will execute it. While `.env` files are typically trusted, this pattern is unsafe and could be exploited if an attacker can modify the `.env` file.

**Do this instead:** Use `set -a; source .env; set +a` or use `dotenv` CLI tool.

### Service marks all running jobs as failed on startup

**What happens:** `service/storage.py` `recover_stale_running_jobs` (line 322) marks all jobs with `status = 'running'` as `failed` on service startup, with the error message "Service restarted while job was running."

**Why it's wrong:** If the service is restarted (e.g., for a configuration change or rolling update) while a job is running, the job is marked as failed even if it would have completed successfully. This is a data loss issue — the user loses the transcription result.

**Do this instead:** Use a heartbeat mechanism where the worker updates a `last_heartbeat` timestamp. On startup, only mark jobs as failed if the last heartbeat is older than a threshold (e.g., 2× the job timeout).

## Security

### Service API has no authentication by default

**What happens:** `service/api.py` `require_agent_auth` (line 54) returns early (no auth required) if `settings.api_token` is `None` (line 57). The `docker-compose.yml` does not set `SERVICE_API_TOKEN` by default (line 29 has it commented out).

**Why it's wrong:** The service API is exposed on port 8080 with no authentication by default. Any client that can reach the service can create, query, and cancel transcription jobs. This could be exploited for resource exhaustion (creating many jobs) or information disclosure (querying other users' job results).

**Do this instead:** Require `SERVICE_API_TOKEN` to be set. If it is not set, log a warning and either refuse to start or generate a random token.

### `.env` file is not encrypted and may be committed

**What happens:** The `.env` file (which contains `HUGGINGFACE_TOKEN`) is in `.gitignore` (line 38). However, `.env.example` shows the expected format. The `docker-compose.yml` uses `env_file: - .env` (line 18).

**Why it's wrong:** While `.env` is gitignored, there is no mechanism to prevent accidental commits (e.g., `git add .env` before the gitignore takes effect, or `git add -f .env`). The Hugging Face token has access to gated models and could be abused.

**Do this instead:** Add a pre-commit hook that checks for `.env` commits. Consider using a secrets manager for production deployments.

### No input validation on API `audio_path` field

**What happens:** `service/schemas.py` `CreateJobRequest` (line 19) validates that `audio_path` is a string, but does not validate its length or format. The `jobs.validate_audio_path` function (line 28) does validate that the path resolves under `SERVICE_DATA_DIR`, but this validation happens after the job is created in some code paths.

**Why it's wrong:** While `validate_audio_path` is called in `create_job` (line 127) before `storage.create_job`, the error handling returns a 400 status code, which is correct. However, there is no rate limiting on job creation, which could be exploited for denial of service.

**Do this instead:** Add input length validation to `CreateJobRequest.audio_path` and implement rate limiting on the `/jobs` endpoint.

### No HTTPS/TLS configuration

**What happens:** The service runs on `http://0.0.0.0:8080` with no TLS configuration. The `docker-compose.yml` maps port 8080 directly.

**Why it's wrong:** All API traffic, including bearer tokens and job results, is transmitted in plaintext. If the service is accessible over a network, this is a security risk.

**Do this instead:** Add a reverse proxy (e.g., nginx, traefik) with TLS termination, or add TLS support directly to the FastAPI app.

## Performance

### Whisper model cache grows unbounded

**What happens:** `transcribe_simple.py` uses two module-level dicts for caching: `_WHISPER_MODEL_CACHE` (line 86) and `_FASTER_WHISPER_MODEL_CACHE` (line 87). These caches use `(model_name, device_name)` and `(model_name, device_name, compute_type)` as keys, respectively.

**Why it's wrong:** The caches never evict entries. If the application is long-running and processes audio with different language/device combinations, the caches will accumulate multiple model instances, each consuming significant memory (Whisper models are 100MB–1GB+).

**Do this instead:** Use an LRU cache with a maximum size (e.g., `functools.lru_cache` or `cachetools.LRUCache`).

### Audio data is fully loaded into memory

**What happens:** `transcribe_simple.py` (lines 920-925) loads the entire audio file into memory via `ffmpeg` subprocess and converts it to a `torch.Tensor`. For a 45-minute audio file at 16kHz, this is ~1.4GB of float32 data.

**Why it's wrong:** Large audio files consume significant memory. If multiple jobs are processed concurrently (not currently possible with `max_workers=1`, but could be in the future), memory usage could exceed system limits.

**Do this instead:** Process audio in streaming chunks, loading only the current chunk from disk.

### Diarization pipeline is reloaded for each CLI invocation

**What happens:** `transcribe_simple.py` `transcribe_audio` (line 800) loads the diarization pipeline if `pipeline` is `None` (line 856). The CLI `main()` function (line 1055) does not pre-load the pipeline, so each CLI invocation loads the pipeline from scratch.

**Why it's wrong:** Pipeline loading involves downloading/loading model weights and can take 10–60 seconds. The GUI pre-loads the pipeline (line 176), but the CLI does not.

**Do this instead:** Add a `--preload` option to the CLI that loads the pipeline and exits, or cache the pipeline in a file-based cache.

### `CHUNK_DURATION` of 240 seconds may be too long for some use cases

**What happens:** `transcribe_simple.py` (line 84) sets `CHUNK_DURATION = 240.0` (4 minutes) with `CHUNK_OVERLAP = 3.0` (3 seconds). For a 45-minute audio file, this means ~11 chunks are processed sequentially.

**Why it's wrong:** Each chunk is processed independently by the ASR model, and the overlap is only 3 seconds. Short utterances that span chunk boundaries may be missed or truncated. Additionally, 4-minute chunks may exceed GPU memory for some models.

**Do this instead:** Make `CHUNK_DURATION` configurable via an environment variable or CLI argument, with a default of 120 seconds (2 minutes).

### `test_steps.py` and test scripts hardcode audio filename

**What happens:** `tests/test_steps.py` (line 71), `tests/test_diarization_only.py` (line 37), `tests/test_gpu_integration.py` (line 32), `scripts/setup_and_run.sh` (line 71), and `scripts/run_transcription.sh` (line 25) all hardcode the filename `Music Company Media Productions 10.m4a`.

**Why it's wrong:** These scripts will fail if the audio file doesn't exist or has a different name. The filename is also not in `.gitignore` (it's an `.m4a` file, so it IS gitignored, but the test scripts reference it without checking).

**Do this instead:** Accept the audio file as a command-line argument with a clear error message if it's not provided.

## Fragile Areas

### Pyannote API compatibility layer depends on error message strings

**What happens:** Both `transcribe_simple.py` `load_diarization_pipeline` (line 90) and `gui/main_window.py` `_load_pyannote_pipeline` (line 466) implement a compatibility layer that tries `Pipeline.from_pretrained(candidate, token=token)` first, then falls back to `use_auth_token=token` if the error message contains "unexpected keyword argument 'token'". They also check for "unexpected keyword argument 'plda'" to detect model incompatibility.

**Why it's fragile:** This approach depends on specific error message strings from the pyannote library. If pyannote changes its error messages in a future version, the compatibility layer will fail to detect the correct fallback path.

**Do this instead:** Use `inspect.signature` to check the function signature of `Pipeline.from_pretrained` before calling it, rather than relying on error messages.

### Module-level SQLite connection singleton

**What happens:** `service/storage.py` creates a module-level SQLite connection at import time (line 29: `_conn = _connect()`). The connection uses `check_same_thread=False` (line 23) and is protected by a module-level lock (`_db_lock`, line 16).

**Why it's fragile:** The connection is created when the module is first imported, before `init_db()` is called. If the database path is on a volume that isn't mounted yet (e.g., in a Docker container with delayed volume mounting), the connection will fail silently or create an empty database in the wrong location. Additionally, the single shared connection means that all database operations are serialized, which could be a bottleneck under high load.

**Do this instead:** Use a connection pool or create connections per-request. Ensure `init_db()` is called before any database operations.

### `torch.load` patch guard uses a non-standard attribute

**What happens:** Both `transcribe_simple.py` (line 62) and `gui_main.py` (line 45) use `torch._speaker_sep_torch_load_patched` as a guard to prevent double-patching. This is a non-standard attribute on the `torch` module.

**Why it's fragile:** If the `torch` module is reloaded or if another library uses the same attribute name, the guard will fail. Additionally, setting arbitrary attributes on the `torch` module is a side effect that could interfere with other libraries.

**Do this instead:** Use a module-level flag in the application's own namespace, not on the `torch` module.

### `DiarizeOutput` attribute check is a workaround

**What happens:** `transcribe_simple.py` (line 971) checks `hasattr(diarization, 'speaker_diarization')` to handle the new pyannote `DiarizeOutput` format. This same check is duplicated in 5 other files.

**Why it's fragile:** If pyannote renames the `speaker_diarization` attribute or changes the `DiarizeOutput` structure, all these checks will fail. The check is also not documented — a developer reading the code would not understand why it's there without knowledge of the pyannote API change.

**Do this instead:** Create a helper function `extract_annotation(diarization)` that handles both formats, and use it everywhere.

### `cancel_requested` polling in service worker

**What happens:** `service/jobs.py` `_run_transcription_job` (line 73) checks `storage.cancel_requested(job_id)` before starting transcription, and the `check_interrupt` callback (line 84-85) also checks it. The transcription function (`transcribe_audio`) checks `check_interrupt` between chunks.

**Why it's fragile:** Cancellation is cooperative — the transcription must check `check_interrupt` between chunks. If a chunk takes a long time (e.g., 4 minutes of audio on CPU), the cancellation will not take effect until the chunk completes. Additionally, the `cancel_requested` flag is checked via a database query, which adds latency.

**Do this instead:** Use a threading event or signal for faster cancellation, and reduce `CHUNK_DURATION` to make cancellation more responsive.

### `ProgressHook` usage suppresses GUI progress during diarization

**What happens:** `transcribe_simple.py` (line 964) uses `with ProgressHook() as hook:` during diarization. The `ProgressHook` from pyannote uses the `rich` library for console output, but does not call the `progress_callback` that was passed to `transcribe_audio`.

**Why it's fragile:** The GUI receives no progress updates during the diarization phase (progress stays at 0.25–0.30). For long audio files, the user sees no feedback for minutes during diarization. The `CustomProgressHook` class (line 152) was designed to solve this but is never used.

**Do this instead:** Use `CustomProgressHook` (or a similar custom hook) that forwards progress to the `progress_callback`, or modify the `ProgressHook` usage to also call the callback.

## Scaling Limits

### Single-worker transcription service

**What happens:** `service/jobs.py` `start_worker` (line 149) creates a `ThreadPoolExecutor` with `max_workers=1`. This means only one transcription job can run at a time.

**Why it's a limit:** The service cannot process multiple audio files concurrently. If multiple jobs are queued, they are processed sequentially. For a service that is supposed to be used by LLM agents, this is a significant throughput limitation.

**Scaling path:** Increase `max_workers` to match the number of available GPUs. However, this requires careful memory management, as each worker loads its own copy of the diarization and ASR models.

### SQLite database as job store

**What happens:** `service/storage.py` uses SQLite as the job store. The database is on a named volume (`transcription_db` in `docker-compose.yml`).

**Why it's a limit:** SQLite does not support concurrent writes. While the `_db_lock` serializes access, this means that all database operations (job creation, progress updates, status queries) are serialized. Under high load, this could become a bottleneck.

**Scaling path:** Migrate to PostgreSQL or MySQL for concurrent access. Alternatively, use a separate Redis instance for job state and SQLite for persistence.

### No horizontal scaling support

**What happens:** The service is designed as a single instance with a local SQLite database and local model cache. There is no support for running multiple instances behind a load balancer.

**Why it's a limit:** If the service needs to handle more jobs than a single instance can process, there is no way to scale horizontally. The local model cache means each instance must download its own copy of the models.

**Scaling path:** Use a shared model cache (e.g., NFS or S3) and a shared database. Add a service discovery mechanism so instances can register and deregister.

## Dependencies at Risk

### `numpy<2` constraint

**What happens:** `scripts/requirements_transcription.txt` (line 15) and `scripts/requirements_ml.txt` (no explicit constraint) both depend on `numpy<2`. The comment at line 14 says "pyannote 3.1.x still references np.NaN (removed in NumPy 2)."

**Why it's at risk:** NumPy 2.0 was released in June 2024. Many libraries have migrated to NumPy 2.0, but `pyannote.audio` 3.1.x has not. This means the project is stuck on NumPy 1.x, which may have security vulnerabilities or miss performance improvements.

**Migration plan:** Upgrade to `pyannote.audio>=4.0` when it is released and supports NumPy 2.0. Alternatively, patch the `np.NaN` references in pyannote.

### `nemo_toolkit[asr]` is a heavy, version-sensitive dependency

**What happens:** `scripts/requirements_transcription.txt` (line 12) and `scripts/requirements_ml.txt` (line 7) both include `nemo_toolkit[asr]` without a version constraint.

**Why it's at risk:** NeMo is a large package with many dependencies. It is sensitive to PyTorch version changes and may break with newer PyTorch releases. The `nemo_toolkit[asr]` package also includes many components that are not used by this project (e.g., NLP, vision, speech synthesis).

**Migration plan:** Consider using a lighter-weight ASR library (e.g., `transformers` with a T5/Whisper model) or pinning NeMo to a specific version.

### `openai-whisper` is unmaintained

**What happens:** `scripts/requirements_transcription.txt` (line 8) includes `openai-whisper>=20231117` without an upper bound.

**Why it's at risk:** The `openai-whisper` package is not actively maintained by OpenAI. It may not be compatible with future Python versions or PyTorch releases. The project relies on it as a fallback when `faster-whisper` is not available (e.g., on MPS devices).

**Migration plan:** Ensure `faster-whisper` is always available (it supports CUDA and CPU, just not MPS). For MPS, consider using `transformers` with a Whisper model.

### `pyannote.audio` upper bound prevents upgrades

**What happens:** `scripts/requirements_transcription.txt` (line 5) and `scripts/requirements_ml.txt` (line 5) both pin `pyannote.audio>=3.1.1,<4.0` and `pyannote.audio>=3.1.0,<4.0` respectively.

**Why it's at risk:** The `<4.0` upper bound prevents upgrading to pyannote 4.0, which may fix the NumPy 2.0 compatibility issue and other bugs. However, pyannote 4.0 may also introduce breaking API changes.

**Migration plan:** Monitor pyannote 4.0 release notes. When released, test the compatibility layer and update the upper bound.

## Missing Critical Features

### No `/jobs` list endpoint

**What happens:** The service API has no `GET /jobs` endpoint to list all jobs. The `storage.py` module has `count_queued()` and `count_queued_and_running()` functions, but these are not exposed via the API.

**Why it's missing:** LLM agents using the service have no way to enumerate jobs or check the queue status. They must know the job ID to query a job.

**Blocks:** LLM agents cannot monitor the job queue or discover existing jobs.

### No `/jobs/{job_id}/events` endpoint

**What happens:** The `storage.py` module has `list_events(job_id)` (line 350), but the API does not expose it.

**Why it's missing:** LLM agents cannot retrieve the event history for a job (e.g., when it was queued, started, succeeded/failed).

**Blocks:** LLM agents cannot debug job failures or understand the job lifecycle.

### No transcript export in multiple formats

**What happens:** The GUI exports transcripts as `.txt` (for local files) and `.md` (for YouTube downloads). The service writes transcripts as `.txt` only (`service/jobs.py` `_write_transcript`, line 50).

**Why it's missing:** Users may want to export transcripts in other formats (e.g., `.srt`, `.json`, `.docx`).

**Blocks:** Users cannot use the transcripts in video editing software or other tools that require specific formats.

### No audio file management in service

**What happens:** The service requires audio files to be pre-placed under `SERVICE_DATA_DIR` (default `/data`). There is no upload endpoint or file management API.

**Why it's missing:** LLM agents must use an external mechanism (e.g., Docker volume mount, file copy) to place audio files in the service's data directory.

**Blocks:** LLM agents cannot upload audio files to the service programmatically.

## Test Coverage Gaps

### `transcribe_simple.py` has minimal test coverage

**What's not tested:** The following functions in `transcribe_simple.py` have no unit tests:
- `load_diarization_pipeline` (line 90) — the entire pyannote pipeline loading with API compatibility
- `_transcribe_with_parakeet` (line 625) — the entire Parakeet transcription path
- `_transcribe_with_whisper` (line 694) — the Whisper transcription path (only partially tested via `test_device_detection.py` with mocks)
- `transcribe_audio` (line 800) — the main transcription function
- `merge_consecutive_same_speaker` (line 1017) — the segment merging function
- `_iter_audio_chunks` (line 418) — the audio chunking logic
- `_write_temp_wav` (line 434) — the temp WAV file creation
- `_segment_speaker` (line 443) — the speaker attribution logic
- `_append_segment` (line 449) — the segment appending logic
- `_load_whisper_model` (line 520) — the Whisper model loading
- `_load_faster_whisper_model` (line 567) — the faster-whisper model loading
- `_transcribe_chunk_with_faster_whisper` (line 586) — the faster-whisper chunk transcription
- `_transcribe_chunk_with_openai_whisper` (line 607) — the openai-whisper chunk transcription
- `CustomProgressHook` (line 152) — the custom progress hook class
- `_faster_whisper_compute_type` (line 558) — the compute type selection
- `describe_transcription_language` (line 411) — the language description function
- `_load_whisper_model` (line 520) — the Whisper model loading

**Files:** `transcribe_simple.py`

**Risk:** Changes to the transcription pipeline can introduce bugs that are not caught by tests. The `test_device_detection.py` file only tests `get_device`, `normalize_transcription_language`, `_select_whisper_backend`, `_SpeakerTurnMerger`, and `_transcribe_with_whisper` (with mocks).

**Priority:** High

### `service/` modules have no test coverage

**What's not tested:** The following service modules have no unit tests:
- `service/api.py` — all API endpoints (health, create job, get job, get result, cancel, warmup)
- `service/storage.py` — all database functions (create, get, claim, update progress, mark succeeded/failed/cancelled, recover stale jobs, list events)
- `service/config.py` — the settings loading logic
- `service/preflight.py` — the preflight checks (CUDA, ffmpeg, token)
- `service/schemas.py` — the Pydantic models
- `service/jobs.py` — the worker loop, job execution, and audio path validation (only `validate_audio_path` is tested in `test_service_paths.py`)

**Files:** `service/api.py`, `service/storage.py`, `service/config.py`, `service/preflight.py`, `service/schemas.py`, `service/jobs.py`

**Risk:** Changes to the service API or database layer can introduce bugs that are not caught by tests. The `test_service_paths.py` file only tests `validate_audio_path`.

**Priority:** High

### `gui/` modules have minimal test coverage

**What's not tested:** The following GUI modules have no unit tests:
- `gui/main_window.py` — all GUI logic (file selection, pipeline loading, transcription start/cancel, transcript display, auto-save)
- `gui/transcription_worker.py` — the QThread worker (run, cancel, cleanup)
- `gui/audio_converter.py` — the audio format conversion and cleanup

**Files:** `gui/main_window.py`, `gui/transcription_worker.py`, `gui/audio_converter.py`

**Risk:** Changes to the GUI can introduce bugs that are not caught by tests. The `test_markdown_export.py` and `test_youtube_download.py` files only test the markdown export and YouTube download modules.

**Priority:** Medium

### Integration tests require external resources

**What's not tested:** The following test files require external resources that are not available in CI:
- `tests/test_diarization_only.py` — requires `HUGGINGFACE_TOKEN` and `Music Company Media Productions 10.m4a`
- `tests/test_gpu_integration.py` — requires `HUGGINGFACE_TOKEN` and `test_chunk_2min.m4a`
- `tests/test_parakeet_basic.py` — requires `HUGGINGFACE_TOKEN` and `tests/test_chunk_2min.m4a`
- `tests/test_parakeet_integration.py` — requires `HUGGINGFACE_TOKEN` and `tests/test_chunk_2min.m4a`
- `tests/test_steps.py` — requires `HUGGINGFACE_TOKEN` and `Music Company Media Productions 10.m4a`
- `tests/benchmark_gpu_performance.py` — requires `HUGGINGFACE_TOKEN` and `test_chunk_2min.m4a`
- `tests/benchmark_whisper_vs_parakeet.py` — requires `HUGGINGFACE_TOKEN` and `tests/test_chunk_2min.m4a`
- `tests/quick_benchmark.py` — requires `HUGGINGFACE_TOKEN` and `test_chunk_2min.m4a`

**Files:** `tests/test_diarization_only.py`, `tests/test_gpu_integration.py`, `tests/test_parakeet_basic.py`, `tests/test_parakeet_integration.py`, `tests/test_steps.py`, `tests/benchmark_gpu_performance.py`, `tests/benchmark_whisper_vs_parakeet.py`, `tests/quick_benchmark.py`

**Risk:** These tests cannot run in CI without setting up Hugging Face tokens and test audio files. The `test_chunk_2min.m4a` file is not committed (gitignored as `.m4a`) and must be generated by `create_test_chunk.py`.

**Priority:** Medium

### No test configuration or test runner

**What's not tested:** There is no `pytest.ini`, `pyproject.toml`, or `setup.cfg` with test configuration. There is no `tox.ini` or CI workflow file.

**Files:** No test configuration files exist.

**Risk:** Tests are not consistently discovered or run. Different test files use different frameworks (some use `unittest`, some use `pytest`-style functions). There is no way to run all tests with a single command.

**Priority:** Medium

---

*Concerns audit: 2026-07-30*
