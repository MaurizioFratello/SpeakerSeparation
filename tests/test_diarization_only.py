#!/usr/bin/env python3
"""Simple test of diarization pipeline - minimal version."""

import os
import sys

# Load .env first
from dotenv import load_dotenv
load_dotenv()

# Disable telemetry
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'

print(f"Telemetry disabled: PYANNOTE_METRICS_ENABLED={os.environ.get('PYANNOTE_METRICS_ENABLED')}")

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    print("ERROR: HUGGINGFACE_TOKEN not found")
    sys.exit(1)

print("\n1. Loading pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token
)
print("✓ Pipeline loaded!")

print("\n2. Loading audio...")
import subprocess
import numpy as np

# Use ffmpeg to load audio
cmd = ['ffmpeg', '-i', 'Music Company Media Productions 10.m4a',
       '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-']
result = subprocess.run(cmd, capture_output=True, check=True)
audio_data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
waveform = torch.from_numpy(audio_data).unsqueeze(0)

# Limit to first 60 seconds
max_samples = 60 * 16000
if waveform.shape[1] > max_samples:
    waveform = waveform[:, :max_samples]

print(f"✓ Audio loaded: {waveform.shape[1]/16000:.1f}s")

print("\n3. Running diarization...")
audio_input = {"waveform": waveform, "sample_rate": 16000}

with ProgressHook() as hook:
    diarization = pipeline(audio_input, hook=hook, num_speakers=2)

print("✓ Diarization complete!")

print("\n4. Results:")
if hasattr(diarization, 'speaker_diarization'):
    annotation = diarization.speaker_diarization
else:
    annotation = diarization

for turn, _, speaker in annotation.itertracks(yield_label=True):
    print(f"  {turn.start:.1f}s - {turn.end:.1f}s: SPEAKER_{speaker}")

print(f"\nTotal speakers found: {len(annotation.labels())}")
