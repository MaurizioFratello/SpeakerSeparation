"""Pydantic models for agent-facing API."""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class CreateJobRequest(BaseModel):
    """Path to audio inside the container (must be under SERVICE_DATA_DIR)."""

    audio_path: str = Field(..., description="Absolute path, e.g. /data/recording.wav")
    num_speakers: Optional[int] = Field(
        None, ge=1, description="Fixed speaker count; omit for auto-detect"
    )
    test_mode: bool = Field(False, description="Process only first 60 seconds")


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    message: str
    error: Optional[str] = None
    created_at: float
    updated_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    output_path: Optional[str] = None
    audio_path: str
    test_mode: bool
    num_speakers: Optional[int] = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    output_path: Optional[str] = None
    segments: List[TranscriptSegment] = Field(default_factory=list)
    segment_count: int = 0


class HealthResponse(BaseModel):
    ok: bool
    cuda_available: bool
    cuda_device_count: int
    ffmpeg_ok: bool
    huggingface_token_configured: bool
    data_dir: str
    output_dir: str
    db_path: str
    message: str


class JobEventOut(BaseModel):
    ts: float
    event: str
    detail: Optional[str] = None


class WarmupResponse(BaseModel):
    accepted: bool
    message: str
    cuda_available: bool
