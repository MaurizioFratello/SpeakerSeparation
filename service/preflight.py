"""Startup checks: CUDA, ffmpeg, Hugging Face token."""

from __future__ import annotations

import logging
import shutil
from service.config import settings

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def check_huggingface_token() -> bool:
    import os

    return bool(os.getenv("HUGGINGFACE_TOKEN", "").strip())


def cuda_info() -> tuple[bool, int]:
    try:
        import torch

        ok = torch.cuda.is_available()
        n = torch.cuda.device_count() if ok else 0
        return ok, n
    except Exception as e:
        logger.warning("CUDA probe failed: %s", e)
        return False, 0


def run_preflight() -> tuple[bool, str]:
    """
    Returns (ok, message). When SERVICE_REQUIRE_CUDA=true, CUDA must be available.
    """
    if not check_ffmpeg():
        return False, "ffmpeg not found on PATH"
    if not check_huggingface_token():
        return False, "HUGGINGFACE_TOKEN is not set"
    cuda_ok, count = cuda_info()
    if settings.require_cuda and not cuda_ok:
        return False, "CUDA not available (set SERVICE_REQUIRE_CUDA=false to allow CPU-only)"
    if cuda_ok:
        return True, f"Preflight OK (CUDA devices: {count})"
    return True, "Preflight OK (CPU mode; SERVICE_REQUIRE_CUDA=false)"
