"""SQLite-backed durable job store (thread-safe)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from service.config import settings
from service.schemas import JobStatus


_db_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(settings.db_path),
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


_conn = _connect()


def init_db() -> None:
    with _db_lock:
        _conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              audio_path TEXT NOT NULL,
              num_speakers INTEGER,
              test_mode INTEGER NOT NULL DEFAULT 0,
              progress REAL NOT NULL DEFAULT 0,
              progress_message TEXT NOT NULL DEFAULT '',
              error TEXT,
              output_path TEXT,
              result_segments_json TEXT,
              cancel_requested INTEGER NOT NULL DEFAULT 0,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              started_at REAL,
              completed_at REAL
            );

            CREATE TABLE IF NOT EXISTS job_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_id TEXT NOT NULL,
              ts REAL NOT NULL,
              event TEXT NOT NULL,
              detail TEXT,
              FOREIGN KEY(job_id) REFERENCES jobs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status_created
              ON jobs(status, created_at);
            """
        )
        _conn.commit()


def _now() -> float:
    return time.time()


def _append_event(job_id: str, event: str, detail: Optional[str] = None) -> None:
    _conn.execute(
        "INSERT INTO job_events (job_id, ts, event, detail) VALUES (?, ?, ?, ?)",
        (job_id, _now(), event, detail),
    )


def count_queued() -> int:
    with _db_lock:
        row = _conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE status = ?",
            (JobStatus.queued.value,),
        ).fetchone()
    return int(row["c"]) if row else 0


def count_queued_and_running() -> int:
    with _db_lock:
        row = _conn.execute(
            """
            SELECT COUNT(*) AS c FROM jobs
            WHERE status IN (?, ?)
            """,
            (JobStatus.queued.value, JobStatus.running.value),
        ).fetchone()
    return int(row["c"]) if row else 0


def create_job(
    audio_path: str,
    num_speakers: Optional[int],
    test_mode: bool,
) -> str:
    job_id = str(uuid.uuid4())
    ts = _now()
    with _db_lock:
        _conn.execute(
            """
            INSERT INTO jobs (
              id, status, audio_path, num_speakers, test_mode,
              progress, progress_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                job_id,
                JobStatus.queued.value,
                audio_path,
                num_speakers,
                1 if test_mode else 0,
                "Queued",
                ts,
                ts,
            ),
        )
        _append_event(job_id, "queued", audio_path)
        _conn.commit()
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _db_lock:
        row = _conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def try_claim_next_job() -> Optional[str]:
    """Atomically move one queued job to running. Returns job_id or None."""
    with _db_lock:
        _conn.execute("BEGIN IMMEDIATE")
        try:
            row = _conn.execute(
                """
                SELECT id FROM jobs
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (JobStatus.queued.value,),
            ).fetchone()
            if row is None:
                _conn.execute("ROLLBACK")
                return None
            job_id = row["id"]
            ts = _now()
            cur = _conn.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = ?, updated_at = ?, progress = ?, progress_message = ?
                WHERE id = ? AND status = ?
                """,
                (
                    JobStatus.running.value,
                    ts,
                    ts,
                    0.0,
                    "Starting",
                    job_id,
                    JobStatus.queued.value,
                ),
            )
            if cur.rowcount != 1:
                _conn.execute("ROLLBACK")
                return None
            _conn.execute(
                "INSERT INTO job_events (job_id, ts, event, detail) VALUES (?, ?, ?, ?)",
                (job_id, _now(), "running", None),
            )
            _conn.commit()
            return job_id
        except Exception:
            _conn.execute("ROLLBACK")
            raise


def update_progress(job_id: str, message: str, progress: float) -> None:
    ts = _now()
    with _db_lock:
        _conn.execute(
            """
            UPDATE jobs
            SET progress_message = ?, progress = ?, updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (message, max(0.0, min(1.0, progress)), ts, job_id, JobStatus.running.value),
        )
        _conn.commit()


def set_cancel_requested(job_id: str) -> bool:
    ts = _now()
    with _db_lock:
        cur = _conn.execute(
            """
            UPDATE jobs
            SET cancel_requested = 1, updated_at = ?
            WHERE id = ? AND status IN (?, ?)
            """,
            (ts, job_id, JobStatus.queued.value, JobStatus.running.value),
        )
        if cur.rowcount:
            _append_event(job_id, "cancel_requested", None)
            _conn.commit()
        return cur.rowcount > 0


def cancel_requested(job_id: str) -> bool:
    with _db_lock:
        row = _conn.execute(
            "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return bool(row and row["cancel_requested"])


def mark_succeeded(
    job_id: str,
    output_path: str,
    segments: List[Dict[str, Any]],
) -> bool:
    ts = _now()
    payload = json.dumps(segments, ensure_ascii=False)
    with _db_lock:
        cur = _conn.execute(
            """
            UPDATE jobs
            SET status = ?, output_path = ?, result_segments_json = ?,
                progress = 1.0, progress_message = ?, completed_at = ?, updated_at = ?,
                error = NULL
            WHERE id = ? AND status = ?
            """,
            (
                JobStatus.succeeded.value,
                output_path,
                payload,
                "Complete",
                ts,
                ts,
                job_id,
                JobStatus.running.value,
            ),
        )
        if cur.rowcount:
            _append_event(job_id, "succeeded", output_path)
            _conn.commit()
            return True
        _conn.commit()
        return False


def mark_failed(job_id: str, error: str) -> bool:
    ts = _now()
    with _db_lock:
        cur = _conn.execute(
            """
            UPDATE jobs
            SET status = ?, error = ?, progress_message = ?, completed_at = ?, updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (
                JobStatus.failed.value,
                error,
                "Failed",
                ts,
                ts,
                job_id,
                JobStatus.running.value,
            ),
        )
        if cur.rowcount:
            _append_event(job_id, "failed", error[:2000])
            _conn.commit()
            return True
        _conn.commit()
        return False


def mark_cancelled(job_id: str) -> bool:
    ts = _now()
    with _db_lock:
        cur = _conn.execute(
            """
            UPDATE jobs
            SET status = ?, error = ?, progress_message = ?, completed_at = ?, updated_at = ?
            WHERE id = ? AND status IN (?, ?)
            """,
            (
                JobStatus.cancelled.value,
                "Cancelled by client",
                "Cancelled",
                ts,
                ts,
                job_id,
                JobStatus.queued.value,
                JobStatus.running.value,
            ),
        )
        if cur.rowcount:
            _append_event(job_id, "cancelled", None)
            _conn.commit()
            return True
        _conn.commit()
        return False


def recover_stale_running_jobs() -> None:
    """Mark interrupted jobs as failed after container restart."""
    ts = _now()
    with _db_lock:
        rows = _conn.execute(
            "SELECT id FROM jobs WHERE status = ?", (JobStatus.running.value,)
        ).fetchall()
        for row in rows:
            jid = row["id"]
            _conn.execute(
                """
                UPDATE jobs
                SET status = ?, error = ?, progress_message = ?, completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    JobStatus.failed.value,
                    "Service restarted while job was running",
                    "Failed",
                    ts,
                    ts,
                    jid,
                ),
            )
            _append_event(jid, "failed", "stale_running_recovered")
        _conn.commit()


def list_events(job_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with _db_lock:
        rows = _conn.execute(
            """
            SELECT ts, event, detail FROM job_events
            WHERE job_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (job_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]
