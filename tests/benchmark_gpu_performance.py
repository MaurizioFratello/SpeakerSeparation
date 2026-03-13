#!/usr/bin/env python3
"""
Benchmark CPU vs MPS GPU performance for speaker diarization.
"""
import os
import sys
import time

# Load .env and disable telemetry FIRST
from dotenv import load_dotenv
load_dotenv()
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'

import torch
import numpy as np
import subprocess
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import TimingHook


def load_audio(audio_file):
    """Load audio file using ffmpeg."""
    cmd = ['ffmpeg', '-i', audio_file,
           '-f', 's16le', '-acodec', 'pcm_s16le',
           '-ar', '16000', '-ac', '1', '-']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    audio_data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio_data).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": 16000}


def run_benchmark(audio_input, device_name, num_speakers=2):
    """Run diarization benchmark on specified device."""
    print(f"\n{'='*60}")
    print(f"Benchmarking on {device_name.upper()}")
    print('='*60)

    # Load pipeline
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: HUGGINGFACE_TOKEN not found in .env")
        sys.exit(1)

    print("Loading pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token
    )

    # Move to device
    device = torch.device(device_name)
    pipeline.to(device)
    print(f"✓ Pipeline loaded and moved to {device_name}")

    # Run diarization with timing
    print(f"Running diarization on {device_name.upper()}...")
    start_time = time.time()

    with TimingHook() as hook:
        diarization = pipeline(audio_input, hook=hook, num_speakers=num_speakers)

    end_time = time.time()
    total_time = end_time - start_time

    # Extract annotation
    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization

    # Get detailed timings if available
    timings = {}
    if hasattr(diarization, 'timing'):
        timings = diarization.timing

    return {
        'device': device_name,
        'total_time': total_time,
        'speakers_found': len(annotation.labels()),
        'timings': timings,
        'annotation': annotation
    }


def format_time(seconds):
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs:02d}s"


def print_results(cpu_result, gpu_result, audio_duration_seconds):
    """Print comparison results."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    print(f"\nTest audio duration: {format_time(audio_duration_seconds)}")

    # CPU Results
    print(f"\n📊 CPU Performance:")
    print(f"  Total time: {format_time(cpu_result['total_time'])}")
    print(f"  Speakers found: {cpu_result['speakers_found']}")
    if cpu_result['timings']:
        print(f"  Detailed timings:")
        for step, duration in cpu_result['timings'].items():
            print(f"    - {step}: {format_time(duration)}")

    # GPU Results
    print(f"\n🚀 {gpu_result['device'].upper()} Performance:")
    print(f"  Total time: {format_time(gpu_result['total_time'])}")
    print(f"  Speakers found: {gpu_result['speakers_found']}")
    if gpu_result['timings']:
        print(f"  Detailed timings:")
        for step, duration in gpu_result['timings'].items():
            print(f"    - {step}: {format_time(duration)}")

    # Speedup calculations
    speedup = cpu_result['total_time'] / gpu_result['total_time']
    time_saved = cpu_result['total_time'] - gpu_result['total_time']

    print(f"\n⚡ Performance Improvement:")
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Time saved: {format_time(time_saved)} ({time_saved:.1f} seconds)")
    print(f"  Improvement: {((speedup - 1) * 100):.1f}% faster")

    # Extrapolate to full file (44.7 minutes)
    full_file_duration = 44.7 * 60  # seconds
    scaling_factor = full_file_duration / audio_duration_seconds

    cpu_full_estimate = cpu_result['total_time'] * scaling_factor
    gpu_full_estimate = gpu_result['total_time'] * scaling_factor
    full_time_saved = cpu_full_estimate - gpu_full_estimate

    print(f"\n📈 Extrapolation to Full File (44.7 minutes):")
    print(f"  Estimated CPU time: {format_time(cpu_full_estimate)}")
    print(f"  Estimated {gpu_result['device'].upper()} time: {format_time(gpu_full_estimate)}")
    print(f"  Estimated time saved: {format_time(full_time_saved)}")

    # Embeddings comparison if available
    if 'segmentation' in cpu_result['timings'] and 'segmentation' in gpu_result['timings']:
        seg_cpu = cpu_result['timings']['segmentation']
        seg_gpu = gpu_result['timings']['segmentation']
        seg_speedup = seg_cpu / seg_gpu if seg_gpu > 0 else 0

        print(f"\n🔍 Segmentation Stage:")
        print(f"  CPU: {format_time(seg_cpu)}")
        print(f"  {gpu_result['device'].upper()}: {format_time(seg_gpu)}")
        print(f"  Speedup: {seg_speedup:.2f}x")

    if 'embeddings' in cpu_result['timings'] and 'embeddings' in gpu_result['timings']:
        emb_cpu = cpu_result['timings']['embeddings']
        emb_gpu = gpu_result['timings']['embeddings']
        emb_speedup = emb_cpu / emb_gpu if emb_gpu > 0 else 0

        print(f"\n🎯 Embeddings Stage (Main Bottleneck):")
        print(f"  CPU: {format_time(emb_cpu)}")
        print(f"  {gpu_result['device'].upper()}: {format_time(emb_gpu)}")
        print(f"  Speedup: {emb_speedup:.2f}x")
        print(f"  This stage takes {(emb_cpu/cpu_result['total_time']*100):.0f}% of total time on CPU")

    print("\n" + "="*60)


def main():
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_chunk_2min.m4a"

    if not os.path.exists(audio_file):
        print(f"ERROR: Audio file not found: {audio_file}")
        print("Run: python create_test_chunk.py")
        sys.exit(1)

    print("="*60)
    print("GPU Performance Benchmark")
    print("="*60)
    print(f"Audio file: {audio_file}")

    # Load audio
    print("\nLoading audio...")
    audio_input = load_audio(audio_file)
    audio_duration = audio_input['waveform'].shape[1] / 16000
    print(f"✓ Audio loaded: {format_time(audio_duration)}")

    # Detect available GPU
    if torch.cuda.is_available():
        gpu_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_device = "mps"
    else:
        print("\nWARNING: No GPU detected. Comparing CPU vs CPU.")
        gpu_device = "cpu"

    print(f"\nWill compare CPU vs {gpu_device.upper()}")

    # Run CPU benchmark
    cpu_result = run_benchmark(audio_input, "cpu")

    # Run GPU benchmark
    gpu_result = run_benchmark(audio_input, gpu_device)

    # Print comparison
    print_results(cpu_result, gpu_result, audio_duration)


if __name__ == "__main__":
    main()
