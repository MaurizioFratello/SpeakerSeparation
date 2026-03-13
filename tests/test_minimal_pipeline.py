#!/usr/bin/env python3
"""Minimal test of pipeline loading based on official pyannote-audio README."""

import os
from dotenv import load_dotenv

# Disable telemetry
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

print("Loading pipeline...")
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token
)

print("✓ Pipeline loaded successfully!")
print(f"Pipeline type: {type(pipeline)}")
