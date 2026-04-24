# CUDA-enabled transcription microservice.
# Host needs NVIDIA Container Toolkit and a compatible driver.
# PyTorch image tag: adjust if you need a different CUDA minor version.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY scripts/requirements_ml.txt scripts/requirements_service.txt \
    scripts/docker_constraints.txt /app/scripts/
# Constrain torch* to match the CUDA 12.4 runtime base image (avoid pip upgrading to CPU/other CUDA builds).
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -c /app/scripts/docker_constraints.txt \
    -r /app/scripts/requirements_ml.txt \
    -r /app/scripts/requirements_service.txt

COPY transcribe_simple.py /app/
COPY service /app/service/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8080"]
