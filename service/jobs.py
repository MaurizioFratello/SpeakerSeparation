"""Background worker: claim SQLite jobs and run transcribe_simple."""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Optional

from service.config import settings

logger = logging.getLogger(__name__)

_executor: Optional[ThreadPoolExecutor] = None
_worker_thread: Optional[threading.Thread] = None
_stop = threading.Event()


def _ensure_repo_on_path() -> None:
    root = str(settings.repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def validate_audio_path(raw: str) -> str:
    """
    Resolve audio_path and ensure it is under SERVICE_DATA_DIR.
    Returns absolute path string.
    """
    base = settings.data_dir.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"audio_path must resolve under SERVICE_DATA_DIR ({base})"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"audio file not found: {candidate}")
    return str(candidate)


def _write_transcript(output_path: Path, segments: list) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        speaker = seg["speaker"]
        text = seg["text"]
        lines.append(f"[{start:6.1f}s - {end:6.1f}s] {speaker}: {text}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _run_transcription_job(job_id: str) -> None:
    _ensure_repo_on_path()
    from transcribe_simple import merge_consecutive_same_speaker, transcribe_audio

    from service import storage

    row = storage.get_job(job_id)
    if not row:
        logger.error("Job %s missing from DB", job_id)
        return

    if storage.cancel_requested(job_id):
        storage.mark_cancelled(job_id)
        return

    audio_path = row["audio_path"]
    num_speakers = row["num_speakers"]
    test_mode = bool(row["test_mode"])

    def progress_cb(message: str, progress: float) -> None:
        storage.update_progress(job_id, message, progress)

    def check_interrupt() -> bool:
        return storage.cancel_requested(job_id)

    try:
        segments = transcribe_audio(
            audio_file=audio_path,
            num_speakers=num_speakers,
            progress_callback=progress_cb,
            segment_callback=None,
            check_interrupt=check_interrupt,
            test_mode=test_mode,
        )
        if check_interrupt():
            storage.mark_cancelled(job_id)
            return

        merged = merge_consecutive_same_speaker(segments)
        stem = Path(audio_path).stem
        out_name = f"{stem}_transcript_{job_id}.txt"
        out_path = (settings.output_dir / out_name).resolve()
        _write_transcript(out_path, merged)

        if not storage.mark_succeeded(job_id, str(out_path), merged):
            logger.warning(
                "Job %s did not transition to succeeded (likely superseded)", job_id
            )
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        if not storage.mark_failed(job_id, str(e)):
            logger.debug("Job %s failure not recorded (state already terminal)", job_id)


def _worker_loop() -> None:
    global _executor
    assert _executor is not None
    from service import storage

    ex = _executor
    while not _stop.is_set():
        job_id = storage.try_claim_next_job()
        if not job_id:
            time.sleep(0.25)
            continue

        future = ex.submit(_run_transcription_job, job_id)
        try:
            future.result(timeout=settings.job_timeout_sec)
        except FutureTimeoutError:
            logger.error("Job %s exceeded timeout %.0fs", job_id, settings.job_timeout_sec)
            storage.mark_failed(
                job_id,
                f"Exceeded SERVICE_JOB_TIMEOUT_SEC ({settings.job_timeout_sec}s); "
                "GPU work may still be running in the background.",
            )
        except Exception as e:
            logger.exception("Worker error for job %s: %s", job_id, e)
            storage.mark_failed(job_id, str(e))


def start_worker() -> None:
    global _executor, _worker_thread
    if _worker_thread is not None:
        return
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _stop.clear()
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe")
    _worker_thread = threading.Thread(target=_worker_loop, name="job-scheduler", daemon=True)
    _worker_thread.start()
    logger.info("Background transcription worker started")


def stop_worker() -> None:
    global _worker_thread, _executor
    _stop.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=30.0)
        _worker_thread = None
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
    logger.info("Background transcription worker stopped")
