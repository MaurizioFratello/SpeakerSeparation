#!/usr/bin/env python3
"""
Download all required models for speaker diarization and transcription.

This script downloads all models ahead of time to avoid hanging during processing.
Run this once before using the main transcription script.
"""

import os
import sys
from pathlib import Path

print("="*60)
print("Model Download Script")
print("="*60)
print("This will download all required models for:")
print("  - pyannote speaker diarization")
print("  - Whisper transcription")
print("="*60)

# Load HuggingFace token
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print("\nERROR: HUGGINGFACE_TOKEN not found in .env file")
    print("Please set it in .env: HUGGINGFACE_TOKEN=your_token")
    sys.exit(1)

print(f"\n✓ HuggingFace token loaded (length: {len(hf_token)})")

# Download pyannote models
print("\n" + "="*60)
print("Step 1: Downloading pyannote-audio models")
print("="*60)

try:
    from pyannote.audio import Pipeline
    from huggingface_hub import snapshot_download
    
    model_id = "pyannote/speaker-diarization-community-1"
    
    print(f"\nDownloading full model snapshot: {model_id}")
    print("This may take a few minutes (models are ~30MB)...")
    
    # Download entire model repository
    cache_dir = snapshot_download(
        repo_id=model_id,
        token=hf_token,
        resume_download=True,
        local_files_only=False
    )
    
    print(f"✓ Models downloaded to: {cache_dir}")
    
    # Now try to load the pipeline to verify
    print("\nVerifying pipeline loads correctly...")
    import torch
    pipeline = Pipeline.from_pretrained(
        model_id,
        token=hf_token
    )
    print("✓ Pipeline loaded successfully!")
    
except Exception as e:
    print(f"\n✗ ERROR downloading pyannote models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Download Whisper models
print("\n" + "="*60)
print("Step 2: Downloading Whisper models")
print("="*60)

# Check if faster-whisper is available
try:
    from faster_whisper import WhisperModel
    use_faster = True
    print("✓ faster-whisper is installed (recommended)")
except ImportError:
    use_faster = False
    try:
        import whisper
        print("Using openai-whisper (slower)")
    except ImportError:
        print("✗ ERROR: No Whisper implementation found")
        print("Install with: pip install faster-whisper")
        sys.exit(1)

whisper_model_sizes = ["base"]  # Can add "tiny", "small", "medium", "large"

for model_size in whisper_model_sizes:
    print(f"\nDownloading Whisper model: {model_size}")
    
    try:
        if use_faster:
            # faster-whisper downloads automatically on first use
            print(f"  Loading {model_size} model with faster-whisper...")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print(f"  ✓ {model_size} model ready")
        else:
            # openai-whisper
            print(f"  Loading {model_size} model with openai-whisper...")
            model = whisper.load_model(model_size)
            print(f"  ✓ {model_size} model ready")
    except Exception as e:
        print(f"  ✗ ERROR downloading {model_size}: {e}")
        continue

print("\n" + "="*60)
print("All models downloaded successfully!")
print("="*60)
print("\nYou can now run the transcription script:")
print("  python transcribe_with_speakers.py")
print("="*60)
