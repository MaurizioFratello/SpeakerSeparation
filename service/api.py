"""FastAPI application: async jobs API for LLM agents."""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

from service.config import settings
from service import jobs, storage
from service.preflight import cuda_info, run_preflight
from service.schemas import (
    CreateJobRequest,
    CreateJobResponse,
    HealthResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
    TranscriptSegment,
    WarmupResponse,
)

load_dotenv(settings.repo_root / ".env")

logging.basicConfig(
    level=os.getenv("SERVICE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _configure_runtime_env() -> None:
    """Point HF/torch caches at SERVICE_CACHE_ROOT for Docker volumes."""
    cache = settings.cache_root.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    hf_home = cache / "huggingface"
    hub = hf_home / "hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub))


_configure_runtime_env()


def require_agent_auth(
    authorization: Annotated[Optional[str], Header()] = None,
) -> None:
    if not settings.api_token:
        return
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization: Bearer <SERVICE_API_TOKEN> required",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid bearer token"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    storage.recover_stale_running_jobs()
    ok, msg = run_preflight()
    if not ok:
        logger.error("Preflight failed: %s", msg)
        raise RuntimeError(msg)
    logger.info("%s", msg)
    jobs.start_worker()
    yield
    jobs.stop_worker()


app = FastAPI(
    title="SpeakerSeparation Transcription Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    ok_cuda, n = cuda_info()
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    token_ok = bool(os.getenv("HUGGINGFACE_TOKEN", "").strip())
    msg_parts = []
    if settings.require_cuda and not ok_cuda:
        msg_parts.append("CUDA unavailable while SERVICE_REQUIRE_CUDA=true")
    if not ffmpeg_ok:
        msg_parts.append("ffmpeg missing")
    if not token_ok:
        msg_parts.append("HUGGINGFACE_TOKEN missing")
    ok = not msg_parts
    if ok:
        msg = "healthy"
    else:
        msg = "; ".join(msg_parts)
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        ok=ok,
        cuda_available=ok_cuda,
        cuda_device_count=n,
        ffmpeg_ok=ffmpeg_ok,
        huggingface_token_configured=token_ok,
        data_dir=str(settings.data_dir),
        output_dir=str(settings.output_dir),
        db_path=str(settings.db_path),
        message=msg,
    )


@app.post("/jobs", response_model=CreateJobResponse, dependencies=[Depends(require_agent_auth)])
def create_job(body: CreateJobRequest) -> CreateJobResponse:
    try:
        resolved = jobs.validate_audio_path(body.audio_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if storage.count_queued() >= settings.max_queued_jobs:
        raise HTTPException(
            status_code=429,
            detail=f"Too many queued jobs (max {settings.max_queued_jobs})",
        )

    job_id = storage.create_job(
        audio_path=resolved,
        num_speakers=body.num_speakers,
        test_mode=body.test_mode,
    )
    logger.info(
        "job_created job_id=%s audio_path=%s test_mode=%s",
        job_id,
        resolved,
        body.test_mode,
    )
    return CreateJobResponse(job_id=job_id, status=JobStatus.queued)


def _row_to_status(row: dict) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=row["id"],
        status=JobStatus(row["status"]),
        progress=float(row["progress"]),
        message=row["progress_message"] or "",
        error=row["error"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        started_at=float(row["started_at"]) if row["started_at"] else None,
        completed_at=float(row["completed_at"]) if row["completed_at"] else None,
        output_path=row["output_path"],
        audio_path=row["audio_path"],
        test_mode=bool(row["test_mode"]),
        num_speakers=row["num_speakers"],
    )


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(require_agent_auth)],
)
def get_job(job_id: str) -> JobStatusResponse:
    row = storage.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return _row_to_status(row)


@app.get(
    "/jobs/{job_id}/result",
    response_model=JobResultResponse,
    dependencies=[Depends(require_agent_auth)],
)
def get_job_result(job_id: str) -> JobResultResponse:
    row = storage.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    st = JobStatus(row["status"])
    if st != JobStatus.succeeded:
        raise HTTPException(
            status_code=409,
            detail=f"Job not complete (status={st.value})",
        )
    raw = row["result_segments_json"]
    segments: list = json.loads(raw) if raw else []
    return JobResultResponse(
        job_id=job_id,
        status=st,
        output_path=row["output_path"],
        segments=[TranscriptSegment(**s) for s in segments],
        segment_count=len(segments),
    )


@app.post(
    "/jobs/{job_id}/cancel",
    dependencies=[Depends(require_agent_auth)],
)
def cancel_job(job_id: str) -> dict:
    row = storage.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    st = JobStatus(row["status"])
    if st in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
        return {"job_id": job_id, "status": st.value, "detail": "already terminal"}
    if st == JobStatus.queued:
        if storage.mark_cancelled(job_id):
            logger.info("job_cancelled job_id=%s (was queued)", job_id)
        return {"job_id": job_id, "status": "cancelled"}
    if st == JobStatus.running:
        storage.set_cancel_requested(job_id)
        logger.info("job_cancel_requested job_id=%s", job_id)
        return {"job_id": job_id, "status": "running", "detail": "cancel_requested"}
    raise HTTPException(status_code=500, detail="unexpected job state")


@app.post(
    "/admin/warmup",
    response_model=WarmupResponse,
    dependencies=[Depends(require_agent_auth)],
)
def admin_warmup() -> WarmupResponse:
    """Lightweight CUDA visibility check; heavy model download happens on first job."""
    ok_cuda, n = cuda_info()
    return WarmupResponse(
        accepted=True,
        message=f"CUDA available={ok_cuda}, device_count={n}",
        cuda_available=ok_cuda,
    )
