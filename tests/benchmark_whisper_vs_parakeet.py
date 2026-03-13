#!/usr/bin/env python3
"""
Benchmark Whisper vs Parakeet performance and quality on 2-minute test audio.
"""
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import time
import torch
from faster_whisper import WhisperModel
import nemo.collections.asr as nemo_asr


def benchmark_comparison(audio_file="tests/test_chunk_2min.m4a"):
    """Compare Whisper vs Parakeet on performance and output."""

    if not os.path.exists(audio_file):
        print(f"ERROR: Audio file not found: {audio_file}")
        print("Run: python tests/create_test_chunk.py")
        return

    print("="*60)
    print("Whisper vs Parakeet Benchmark")
    print("="*60)

    # Test Whisper
    print("\n[Whisper] Loading model...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

    print("[Whisper] Transcribing...")
    start = time.time()
    whisper_segments, whisper_info = whisper_model.transcribe(audio_file, beam_size=5)
    whisper_segments = list(whisper_segments)  # Materialize iterator
    whisper_time = time.time() - start

    print(f"[Whisper] Time: {whisper_time:.2f}s")
    print(f"[Whisper] Segments: {len(whisper_segments)}")
    print(f"[Whisper] Language: {whisper_info.language}")

    # Test Parakeet
    print("\n[Parakeet] Loading model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3"
    )

    # Move to MPS if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        asr_model = asr_model.to(torch.device("mps"))
        print("[Parakeet] Using MPS GPU")

    print("[Parakeet] Transcribing...")
    start = time.time()
    hypotheses = asr_model.transcribe([audio_file], timestamps=True)
    parakeet_time = time.time() - start

    parakeet_segments = hypotheses[0].timestamp['segment']
    print(f"[Parakeet] Time: {parakeet_time:.2f}s")
    print(f"[Parakeet] Segments: {len(parakeet_segments)}")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    speedup = whisper_time / parakeet_time if parakeet_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"Time difference: {abs(whisper_time - parakeet_time):.2f}s")

    # Extrapolate to full file (44.7 minutes = 2682 seconds)
    full_duration = 2682
    test_duration = 120  # 2 minutes
    scale_factor = full_duration / test_duration

    whisper_full_estimate = whisper_time * scale_factor
    parakeet_full_estimate = parakeet_time * scale_factor

    print(f"\nExtrapolation to 44.7-minute file:")
    print(f"  Whisper estimate: {whisper_full_estimate/60:.1f} minutes")
    print(f"  Parakeet estimate: {parakeet_full_estimate/60:.1f} minutes")
    print(f"  Time saved: {(whisper_full_estimate - parakeet_full_estimate)/60:.1f} minutes")

    # Show sample outputs
    print("\n--- Sample Whisper Output ---")
    for seg in whisper_segments[:3]:
        print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

    print("\n--- Sample Parakeet Output ---")
    for seg in parakeet_segments[:3]:
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['segment']}")

    print("\n" + "="*60)


if __name__ == "__main__":
    benchmark_comparison()
