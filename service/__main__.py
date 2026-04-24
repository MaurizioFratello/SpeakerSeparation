"""Run API: python -m service (or uvicorn service.api:app)."""

import os

import uvicorn

from service.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "service.api:app",
        host=settings.host,
        port=settings.port,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )
