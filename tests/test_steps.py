#!/usr/bin/env python3
"""
Step-by-step testing script to identify bottlenecks.
Run each test separately to see where it's slow.
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()

print("="*60)
print("STEP-BY-STEP TESTING")
print("="*60)

# Test 1: Check models
print("\n[TEST 1] Checking downloaded models...")
from pathlib import Path
hf_cache = Path.home() / '.cache' / 'huggingface'
whisper_cache = Path.home() / '.cache' / 'whisper'

print(f"HuggingFace cache: {hf_cache.exists()}")
print(f"Whisper cache: {whisper_cache.exists()}")
if whisper_cache.exists():
    models = list(whisper_cache.glob('*.pt'))
    print(f"Whisper models: {len(models)} ({', '.join([m.name for m in models])})")

# Test 2: Load pipeline (this will download models if needed)
print("\n[TEST 2] Loading pyannote pipeline (may download models)...")
print("This may take a while on first run as it downloads models...")
start = time.time()
try:
    from pyannote.audio import Pipeline
    import torch
    
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    pipeline = Pipeline.from_pretrained(
        'pyannote/speaker-diarization-community-1',
        token=hf_token
    )
    device = 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
    pipeline.to(torch.device(device))
    elapsed = time.time() - start
    print(f"✓ Pipeline loaded in {elapsed:.1f} seconds on {device}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Load Whisper
print("\n[TEST 3] Loading Whisper model...")
start = time.time()
try:
    import whisper
    model = whisper.load_model('base', device='cpu')
    elapsed = time.time() - start
    print(f"✓ Whisper loaded in {elapsed:.1f} seconds")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Quick diarization (10 seconds only)
print("\n[TEST 4] Testing diarization on 10 seconds of audio...")
print("Creating test audio file...")
import subprocess
import tempfile

temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
temp_wav.close()

cmd = [
    'ffmpeg', '-i', 'Music Company Media Productions 10.m4a',
    '-t', '10',  # Only 10 seconds
    '-ar', '16000',
    '-ac', '1',
    '-y',
    temp_wav.name
]
try:
    subprocess.run(cmd, capture_output=True, check=True)
    print("✓ Test audio created")
    
    print("Running diarization (this is the slow part)...")
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    import torchaudio
    
    # Load audio in memory (workaround for torchcodec issue)
    waveform, sample_rate = torchaudio.load(temp_wav.name)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
    
    start = time.time()
    with ProgressHook() as hook:
        diarization = pipeline(audio_dict, hook=hook, num_speakers=2)
    elapsed = time.time() - start
    
    print(f"✓ Diarization completed in {elapsed:.1f} seconds")
    
    # Handle new DiarizeOutput format
    if hasattr(diarization, 'speaker_diarization'):
        # New format (DiarizeOutput)
        annotation = diarization.speaker_diarization
        speakers = set()
        segments = []
        for turn, track, speaker in annotation.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append((turn, speaker))
        
        print(f"  Found {len(speakers)} speaker(s): {sorted(speakers)}")
        for turn, speaker in segments[:3]:
            print(f"  {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
    else:
        # Old format (Annotation)
        speakers = diarization.labels()
        print(f"  Found {len(speakers)} speaker(s): {sorted(speakers)}")
        segments = list(diarization.itertracks(yield_label=True))[:3]
        for turn, track, speaker in segments:
            print(f"  {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
    
    # Cleanup
    os.unlink(temp_wav.name)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    if os.path.exists(temp_wav.name):
        os.unlink(temp_wav.name)

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
print("\nIf TEST 2 took a long time, models were downloading.")
print("If TEST 4 took a long time, that's the diarization bottleneck.")
print("For a 2-minute audio file, expect 1-3 minutes of diarization time.")

