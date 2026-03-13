#!/usr/bin/env python3
"""Quick CPU vs MPS comparison for 2-minute audio."""
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'

import torch
import numpy as np
import subprocess
from pyannote.audio import Pipeline

def load_audio(audio_file):
    cmd = ['ffmpeg', '-i', audio_file, '-f', 's16le', '-acodec', 'pcm_s16le',
           '-ar', '16000', '-ac', '1', '-']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    audio_data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio_data).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": 16000}

token = os.getenv("HUGGINGFACE_TOKEN")
audio_input = load_audio("test_chunk_2min.m4a")

print("="*60)
print("Quick CPU vs MPS Benchmark (2-minute audio)")
print("="*60)

# Test CPU
print("\n[CPU] Loading pipeline...")
pipeline_cpu = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
pipeline_cpu.to(torch.device("cpu"))
print("[CPU] Running diarization...")
start = time.time()
diarization_cpu = pipeline_cpu(audio_input, num_speakers=2)
cpu_time = time.time() - start
print(f"[CPU] Time: {cpu_time:.1f} seconds")

# Test MPS
print("\n[MPS] Loading pipeline...")
pipeline_mps = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
pipeline_mps.to(torch.device("mps"))
print("[MPS] Running diarization...")
start = time.time()
diarization_mps = pipeline_mps(audio_input, num_speakers=2)
mps_time = time.time() - start
print(f"[MPS] Time: {mps_time:.1f} seconds")

# Results
speedup = cpu_time / mps_time
print("\n" + "="*60)
print(f"Speedup: {speedup:.2f}x faster with MPS")
print(f"Time saved: {cpu_time - mps_time:.1f} seconds")
print("\nExtrapolation to full 44.7-minute file:")
print(f"  CPU estimate: {cpu_time * 22.35:.0f} seconds ({cpu_time * 22.35 / 60:.1f} minutes)")
print(f"  MPS estimate: {mps_time * 22.35:.0f} seconds ({mps_time * 22.35 / 60:.1f} minutes)")
print(f"  Time saved: {(cpu_time - mps_time) * 22.35 / 60:.1f} minutes")
print("="*60)
