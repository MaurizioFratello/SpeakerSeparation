#!/usr/bin/env python3
"""
Speaker-attributed transcription using pyannote-audio (diarization) + Whisper (transcription).

PURPOSE: Transcribe a long audio file (45 min) with two speakers, producing
         speaker-attributed transcription output.

CONTEXT: Combines pyannote-audio for speaker diarization (who spoke when) with
         Whisper for speech-to-text transcription, then merges results.
"""

import os
import sys

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not installed, skip .env loading

# Disable telemetry AFTER loading .env (so this takes precedence)
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'

import torch
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging
import time
from datetime import datetime

# Minimal logging setup - let default handlers work
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pyannote-audio imports (AFTER disabling telemetry)
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation, Segment

# Whisper imports - try faster-whisper first (much faster), fallback to openai-whisper
USE_FASTER_WHISPER = True
try:
    from faster_whisper import WhisperModel
    faster_whisper_available = True
except ImportError:
    faster_whisper_available = False
    try:
        import whisper
    except ImportError:
        print("ERROR: No Whisper implementation found.")
        print("Install one of: pip install faster-whisper  OR  pip install openai-whisper")
        sys.exit(1)

# Audio processing
try:
    import torchaudio
except ImportError:
    print("ERROR: torchaudio not installed. Install with: pip install torchaudio")
    sys.exit(1)


def get_device(for_whisper: bool = False) -> str:
    """
    Detect the best available device for PyTorch.
    
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU
    
    Note: Whisper doesn't fully support MPS, so it will use CPU on Mac.
    pyannote-audio can use MPS.
    
    Args:
        for_whisper: If True, skip MPS (Whisper doesn't support it)
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif for_whisper:
        # Whisper doesn't fully support MPS, use CPU
        return "cpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_audio_file(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and return waveform + sample rate.
    
    Handles .m4a format by using ffmpeg via subprocess (torchaudio doesn't support ffmpeg backend directly).
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (waveform tensor, sample_rate)
    """
    logger.debug(f"load_audio_file: Starting, path={audio_path}")
    start = time.time()
    
    # For m4a files, use ffmpeg via subprocess to convert to raw audio, then load
    if audio_path.lower().endswith('.m4a'):
        logger.debug("load_audio_file: Detected m4a file, using ffmpeg")
        import subprocess
        import numpy as np
        
        # Use ffmpeg to extract raw audio data
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-f', 's16le',  # 16-bit little-endian raw audio
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-'  # Output to stdout
        ]
        
        logger.debug(f"load_audio_file: Running ffmpeg command: {' '.join(cmd[:3])}...")
        try:
            ffmpeg_start = time.time()
            result = subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            logger.debug(f"load_audio_file: ffmpeg completed in {time.time() - ffmpeg_start:.1f}s, output size: {len(result.stdout)} bytes")
            
            # Convert bytes to numpy array
            logger.debug("load_audio_file: Converting bytes to numpy array")
            audio_data = np.frombuffer(result.stdout, dtype=np.int16)
            logger.debug(f"load_audio_file: Audio data shape: {audio_data.shape}")
            
            # Convert to float32 and normalize to [-1, 1]
            logger.debug("load_audio_file: Converting to float32 tensor")
            waveform = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
            # Add channel dimension: (1, samples)
            waveform = waveform.unsqueeze(0)
            sample_rate = 16000
            logger.info(f"load_audio_file: ✓ Loaded m4a in {time.time() - start:.1f}s, shape={waveform.shape}, sr={sample_rate}")
        except subprocess.TimeoutExpired:
            logger.error(f"load_audio_file: ✗ ffmpeg timed out after 300s")
            raise RuntimeError("ffmpeg timed out loading audio file") from None
        except Exception as e:
            logger.error(f"load_audio_file: ✗ Error: {e}")
            raise RuntimeError(f"Failed to load m4a file with ffmpeg: {e}") from e
    else:
        logger.debug("load_audio_file: Loading with torchaudio")
        # Load directly for other formats (WAV, etc.)
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.info(f"load_audio_file: ✓ Loaded in {time.time() - start:.1f}s")
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        logger.debug("load_audio_file: Converting stereo to mono")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    logger.debug(f"load_audio_file: Final shape={waveform.shape}, sample_rate={sample_rate}")
    return waveform, sample_rate


def load_pipeline(hf_token: str) -> Pipeline:
    """
    Load pyannote pipeline using official recommended approach.

    Based on official pyannote-audio documentation:
    https://github.com/pyannote/pyannote-audio

    Args:
        hf_token: HuggingFace access token

    Returns:
        Loaded Pipeline object

    Raises:
        Exception: If pipeline loading fails (e.g., authentication, network issues)
    """
    logger.info("Loading pyannote pipeline from HuggingFace...")
    load_start = time.time()

    try:
        # Simple, straightforward approach from official docs
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token
        )

        elapsed = time.time() - load_start
        logger.info(f"✓ Pipeline loaded successfully in {elapsed:.1f}s")
        print(f"✓ Pipeline loaded in {elapsed:.1f}s")
        return pipeline

    except Exception as e:
        # Provide helpful error messages for common issues
        if "403" in str(e) or "Forbidden" in str(e) or "gated" in str(e).lower():
            print("\n" + "="*60)
            print("ERROR: Access Denied (403 Forbidden)")
            print("="*60)
            print("This usually means you need to:")
            print("1. Accept the terms at: https://hf.co/pyannote/speaker-diarization-community-1")
            print("2. Create an access token at: https://hf.co/settings/tokens")
            print("3. Make sure your token has 'Read' access")
            print("="*60)
        logger.error(f"Failed to load pipeline: {e}")
        raise


def perform_diarization(
    audio_path: str,
    hf_token: str,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    device: str = None,
    max_duration: float = None
) -> Annotation:
    logger.info("="*60)
    logger.info("perform_diarization: STARTING")
    logger.info(f"  audio_path: {audio_path}")
    logger.info(f"  num_speakers: {num_speakers}")
    logger.info(f"  max_duration: {max_duration}")
    logger.info("="*60)
    diarization_start = time.time()
    """
    Perform speaker diarization using pyannote-audio.

    Identifies who spoke when in the audio file.

    Args:
        audio_path: Path to audio file
        hf_token: HuggingFace access token (required for gated models)
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        device: Device to use ('cuda', 'cpu', or None for auto)
        max_duration: Maximum duration in seconds (for testing, None = full file)

    Returns:
        Annotation object with speaker segments
    """
    logger.info("perform_diarization: Loading pipeline...")
    pipeline_start = time.time()

    # Load the pipeline using official approach (with telemetry disabled)
    try:
        pipeline = load_pipeline(hf_token)
        logger.info(f"perform_diarization: ✓ Pipeline loaded in {time.time() - pipeline_start:.1f}s")
    except Exception as e:
        logger.error(f"perform_diarization: Failed to load pipeline: {e}")
        raise
    
    # Set device
    logger.debug("perform_diarization: Setting device")
    if device is None:
        device = get_device()
    logger.info(f"perform_diarization: Moving pipeline to device: {device}")
    pipeline.to(torch.device(device))
    logger.info(f"perform_diarization: ✓ Pipeline on {device}")
    
    # Prepare pipeline call arguments
    pipeline_kwargs = {}
    if num_speakers is not None:
        pipeline_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        pipeline_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        pipeline_kwargs["max_speakers"] = max_speakers
    
    # If max_duration is set, create a temporary audio file with limited duration
    audio_to_process = audio_path
    temp_file = None
    
    if max_duration is not None:
        print(f"TEST MODE: Limiting audio to first {max_duration:.1f} seconds...")
        import tempfile
        import subprocess
        
        # Get audio duration first
        waveform, sample_rate = load_audio_file(audio_path)
        total_duration = waveform.shape[1] / sample_rate
        
        if max_duration < total_duration:
            # Create temporary file with limited duration
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Use ffmpeg to extract first portion
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-t', str(max_duration),
                '-ar', '16000',  # Resample to 16kHz
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                temp_file.name
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            audio_to_process = temp_file.name
            print(f"✓ Created test audio file: {max_duration:.1f}s / {total_duration:.1f}s")
        else:
            print(f"Note: max_duration ({max_duration:.1f}s) >= total duration ({total_duration:.1f}s), using full file")
    
    logger.info("perform_diarization: Loading audio into memory...")
    audio_load_start = time.time()
    waveform, sample_rate = load_audio_file(audio_to_process)
    logger.info(f"perform_diarization: ✓ Audio loaded in {time.time() - audio_load_start:.1f}s")
    
    # Limit to max_duration if in test mode
    if max_duration is not None:
        logger.info(f"perform_diarization: Limiting to {max_duration:.1f}s for testing")
        max_samples = int(max_duration * sample_rate)
        if max_samples < waveform.shape[1]:
            waveform = waveform[:, :max_samples]
            logger.debug(f"perform_diarization: Trimmed to {waveform.shape[1]/sample_rate:.1f}s")
    
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}
    logger.info(f"perform_diarization: Audio ready: {waveform.shape[1]/sample_rate:.1f}s at {sample_rate}Hz")
    logger.info(f"perform_diarization: Starting diarization with {len(pipeline_kwargs)} parameters")
    
    # Run diarization with progress hook and speaker count
    logger.info("perform_diarization: Calling pipeline() - THIS IS THE SLOW PART")
    diarization_start_time = time.time()
    try:
        with ProgressHook() as hook:
            logger.debug("perform_diarization: Inside ProgressHook context")
            diarization = pipeline(audio_input, hook=hook, **pipeline_kwargs)
        logger.info(f"perform_diarization: ✓ Diarization completed in {time.time() - diarization_start_time:.1f}s")
    except Exception as e:
        logger.error(f"perform_diarization: ✗ ERROR during pipeline call: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Clean up temp file if created
    if temp_file is not None:
        import os
        os.unlink(temp_file.name)
    
    # Handle new DiarizeOutput format
    logger.debug("perform_diarization: Processing diarization results")
    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
        speakers = set()
        for turn, track, speaker in annotation.itertracks(yield_label=True):
            speakers.add(speaker)
        logger.info(f"perform_diarization: ✓ Found {len(speakers)} speaker(s): {sorted(speakers)}")
    else:
        # Old format (Annotation)
        speakers = diarization.labels()
        logger.info(f"perform_diarization: ✓ Found {len(speakers)} speaker(s): {sorted(speakers)}")
    
    total_time = time.time() - diarization_start
    logger.info(f"perform_diarization: COMPLETE in {total_time:.1f}s")
    return diarization


def transcribe_segment(
    audio_path: str,
    segment: Segment,
    whisper_model,
    device: str = None,
    use_faster_whisper: bool = False,
    cached_audio: Tuple[torch.Tensor, int] = None
) -> str:
    logger.debug(f"transcribe_segment: {segment.start:.1f}s-{segment.end:.1f}s, use_faster={use_faster_whisper}")
    seg_start = time.time()
    """
    Transcribe a specific time segment of audio using Whisper.
    
    Args:
        audio_path: Path to full audio file
        segment: Segment object with start/end times
        whisper_model: Loaded Whisper model
        device: Device to use
        use_faster_whisper: Whether using faster-whisper
        cached_audio: Pre-loaded (waveform, sample_rate) tuple to avoid reloading
        
    Returns:
        Transcribed text for the segment
    """
    # Use cached audio if available, otherwise load
    if cached_audio is not None:
        logger.debug("transcribe_segment: Using cached audio")
        waveform, sample_rate = cached_audio
    else:
        logger.debug("transcribe_segment: Loading audio file")
        waveform, sample_rate = load_audio_file(audio_path)
    
    # Extract segment
    logger.debug(f"transcribe_segment: Extracting segment from audio")
    start_sample = int(segment.start * sample_rate)
    end_sample = int(segment.end * sample_rate)
    segment_waveform = waveform[:, start_sample:end_sample]
    logger.debug(f"transcribe_segment: Extracted {segment_waveform.shape[1]/sample_rate:.2f}s of audio")
    
    if use_faster_whisper:
        logger.debug("transcribe_segment: Using faster-whisper")
        # faster-whisper expects file path or numpy array
        # Convert to numpy array
        if sample_rate != 16000:
            logger.debug(f"transcribe_segment: Resampling from {sample_rate}Hz to 16000Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            segment_waveform = resampler(segment_waveform)
        
        audio_np = segment_waveform.squeeze().numpy().astype("float32")
        logger.debug(f"transcribe_segment: Calling faster-whisper.transcribe()")
        transcribe_start = time.time()
        
        # Transcribe with faster-whisper
        segments, info = whisper_model.transcribe(
            audio_np,
            language=None,  # Auto-detect
            beam_size=1,  # Faster decoding
            vad_filter=True,  # Voice activity detection
        )
        
        logger.debug(f"transcribe_segment: faster-whisper completed in {time.time() - transcribe_start:.2f}s")
        
        # Combine all segments
        text_parts = [segment.text for segment in segments]
        text = " ".join(text_parts).strip()
        logger.debug(f"transcribe_segment: ✓ Text length: {len(text)} chars")
        return text
    else:
        logger.debug("transcribe_segment: Using openai-whisper")
        # Original openai-whisper
        # Convert to numpy for Whisper (expects float32, 16kHz)
        if sample_rate != 16000:
            logger.debug(f"transcribe_segment: Resampling from {sample_rate}Hz to 16000Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            segment_waveform = resampler(segment_waveform)
        
        audio_np = segment_waveform.squeeze().numpy()
        logger.debug(f"transcribe_segment: Calling whisper.transcribe()")
        transcribe_start = time.time()
        
        # Transcribe
        result = whisper_model.transcribe(
            audio_np,
            language=None,  # Auto-detect
            task="transcribe",
            verbose=False
        )
        
        logger.debug(f"transcribe_segment: whisper completed in {time.time() - transcribe_start:.2f}s")
        text = result["text"].strip()
        logger.debug(f"transcribe_segment: ✓ Text length: {len(text)} chars")
        return text


def combine_diarization_and_transcription(
    audio_path: str,
    diarization: Annotation,
    whisper_model_name: str = "base",
    device: str = None,
    use_parallel: bool = True
) -> List[Dict[str, any]]:
    logger.info("="*60)
    logger.info("combine_diarization_and_transcription: STARTING")
    logger.info(f"  whisper_model: {whisper_model_name}")
    logger.info(f"  use_parallel: {use_parallel}")
    logger.info("="*60)
    transcription_start = time.time()
    """
    Combine diarization and transcription to produce speaker-attributed text.
    
    For each speaker segment, transcribe the audio and combine with speaker label.
    
    Args:
        audio_path: Path to audio file
        diarization: Annotation from pyannote diarization
        whisper_model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        device: Device to use
        use_parallel: Use parallel processing for transcription (faster)
        
    Returns:
        List of dictionaries with speaker, start, end, and text
    """
    # Determine which Whisper implementation to use
    logger.debug("combine_diarization_and_transcription: Checking Whisper implementation")
    use_faster = USE_FASTER_WHISPER and faster_whisper_available
    logger.info(f"combine_diarization_and_transcription: Using faster-whisper: {use_faster}")
    
    if use_faster:
        logger.info(f"combine_diarization_and_transcription: Loading faster-whisper model ({whisper_model_name})...")
        model_load_start = time.time()
        # faster-whisper uses device strings directly
        device_str = "cpu"  # faster-whisper works well on CPU
        logger.debug(f"combine_diarization_and_transcription: Creating WhisperModel with device={device_str}, compute_type=int8")
        whisper_model = WhisperModel(whisper_model_name, device=device_str, compute_type="int8")
        logger.info(f"combine_diarization_and_transcription: ✓ faster-whisper loaded in {time.time() - model_load_start:.1f}s")
    else:
        logger.info(f"combine_diarization_and_transcription: Loading openai-whisper model ({whisper_model_name})...")
        model_load_start = time.time()
        if device is None:
            device = get_device(for_whisper=True)
        logger.debug(f"combine_diarization_and_transcription: Loading with device={device}")
        whisper_model = whisper.load_model(whisper_model_name, device=device)
        logger.info(f"combine_diarization_and_transcription: ✓ Whisper loaded in {time.time() - model_load_start:.1f}s")
    
    # Load full audio once and cache it (avoid reloading for each segment)
    logger.info("combine_diarization_and_transcription: Loading audio file into memory...")
    audio_cache_start = time.time()
    cached_audio = load_audio_file(audio_path)
    logger.info(f"combine_diarization_and_transcription: ✓ Audio cached in {time.time() - audio_cache_start:.1f}s")
    
    # Get all segments sorted by time
    # Handle new DiarizeOutput format
    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization
    
    segments = []
    for turn, track, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "track": track
        })
    
    # Sort by start time
    segments.sort(key=lambda x: x["start"])
    
    logger.info(f"combine_diarization_and_transcription: Processing {len(segments)} segments")
    
    if use_parallel and len(segments) > 10:
        logger.info("combine_diarization_and_transcription: Using PARALLEL processing")
        # Use parallel processing for speed
        print("Using parallel processing for faster transcription...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        start_time = time.time()
        results = []
        
        def transcribe_one(seg_data):
            seg, idx = seg_data
            segment_obj = Segment(seg["start"], seg["end"])
            text = transcribe_segment(
                audio_path, segment_obj, whisper_model, device,
                use_faster_whisper=use_faster, cached_audio=cached_audio
            )
            return {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "index": idx
            }
        
        # Process in parallel (use number of CPU cores)
        import multiprocessing
        max_workers = min(multiprocessing.cpu_count(), len(segments), 8)
        print(f"Using {max_workers} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(transcribe_one, (seg, i)): i 
                      for i, seg in enumerate(segments)}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                results.append(result)
                if completed % 10 == 0 or completed == len(segments):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {completed}/{len(segments)} segments "
                          f"({rate:.1f} segments/sec)")
        
        # Sort by original index to maintain order
        results.sort(key=lambda x: x["index"])
        # Remove index before returning
        for r in results:
            del r["index"]
        
        elapsed = time.time() - start_time
        print(f"✓ Transcription completed in {elapsed:.1f} seconds")
    else:
        # Sequential processing (for small files or when parallel disabled)
        logger.info("combine_diarization_and_transcription: Using SEQUENTIAL processing")
        results = []
        for i, seg in enumerate(segments):
            logger.debug(f"combine_diarization_and_transcription: Segment {i+1}/{len(segments)}: {seg['speaker']} ({seg['start']:.1f}s-{seg['end']:.1f}s)")
            segment_start = time.time()
            segment_obj = Segment(seg["start"], seg["end"])
            
            text = transcribe_segment(
                audio_path, segment_obj, whisper_model, device,
                use_faster_whisper=use_faster, cached_audio=cached_audio
            )
            
            logger.debug(f"combine_diarization_and_transcription: Segment {i+1} completed in {time.time() - segment_start:.1f}s, text_length={len(text)}")
            
            results.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": text
            })
    
    total_time = time.time() - transcription_start
    logger.info(f"combine_diarization_and_transcription: COMPLETE in {total_time:.1f}s, {len(results)} segments")
    return results


def save_transcript(results: List[Dict], output_path: str, format: str = "txt"):
    """
    Save transcription results to file.
    
    Args:
        results: List of transcription dictionaries
        output_path: Output file path
        format: Output format ('txt', 'json', or 'srt')
    """
    if format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"[{result['start']:.1f}s - {result['end']:.1f}s] "
                       f"{result['speaker']}: {result['text']}\n\n")
    
    elif format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format == "srt":
        # SRT subtitle format
        with open(output_path, "w", encoding="utf-8") as f:
            for i, result in enumerate(results, 1):
                start_time = format_srt_time(result["start"])
                end_time = format_srt_time(result["end"])
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{result['speaker']}: {result['text']}\n\n")
    
    print(f"\nTranscript saved to: {output_path}")


def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    """Main function to orchestrate speaker-attributed transcription."""
    start_time = time.time()
    logger.info("="*60)
    logger.info("STARTING TRANSCRIPTION")
    logger.info("="*60)
    
    # Configuration
    audio_file = "Music Company Media Productions 10.m4a"
    audio_path = Path(audio_file)
    logger.info(f"Audio file: {audio_file}")
    
    logger.info(f"main: Checking audio file: {audio_path}")
    if not audio_path.exists():
        logger.error(f"main: ✗ Audio file not found: {audio_file}")
        logger.error(f"main: Current directory: {os.getcwd()}")
        print(f"ERROR: Audio file not found: {audio_file}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    logger.info(f"main: ✓ Audio file exists: {audio_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Get HuggingFace token from environment (.env file, env var, or prompt)
    logger.debug("main: Loading HuggingFace token")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("\n" + "="*60)
        print("HuggingFace Access Token Required")
        print("="*60)
        print("To use pyannote-audio models, you need to:")
        print("1. Accept terms at: https://hf.co/pyannote/speaker-diarization-community-1")
        print("2. Create token at: https://hf.co/settings/tokens")
        print("3. Set token in one of these ways:")
        print("   - .env file: HUGGINGFACE_TOKEN=your_token")
        print("   - Environment variable: export HUGGINGFACE_TOKEN='your_token'")
        print("\nOr enter token now (will not be saved):")
        hf_token = input("HuggingFace token: ").strip()
        if not hf_token:
            print("ERROR: Token required. Exiting.")
            sys.exit(1)
    else:
        print(f"✓ HuggingFace token loaded from .env file (length: {len(hf_token)})")
    
    # Prompt for number of speakers (or use environment variable/test mode)
    logger.debug("main: Determining speaker configuration")
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    num_speakers_env = os.getenv("NUM_SPEAKERS")
    logger.info(f"main: test_mode={test_mode}, num_speakers_env={num_speakers_env}")
    
    if test_mode:
        # Test mode: use 2 speakers, skip prompt
        num_speakers = 2
        logger.info("main: TEST MODE enabled - using 2 speakers, 1 minute limit")
        print("\n" + "="*60)
        print("TEST MODE: Using 2 speakers (auto-configured)")
        print("="*60)
    elif num_speakers_env:
        # Use environment variable
        try:
            num_speakers = int(num_speakers_env)
            if num_speakers < 1:
                print("ERROR: NUM_SPEAKERS must be at least 1. Using auto-detect instead.")
                num_speakers = None
            else:
                print(f"\n✓ Using {num_speakers} speaker(s) from NUM_SPEAKERS environment variable")
        except ValueError:
            print("ERROR: Invalid NUM_SPEAKERS value. Using auto-detect instead.")
            num_speakers = None
    else:
        print("\n" + "="*60)
        print("Speaker Configuration")
        print("="*60)
        print("Enter the number of speakers in the audio file.")
        print("Press Enter to auto-detect (may be less accurate)")
        print("="*60)
        
        num_speakers_input = input("Number of speakers (or Enter for auto-detect): ").strip()
        if num_speakers_input:
            try:
                num_speakers = int(num_speakers_input)
                if num_speakers < 1:
                    print("ERROR: Number of speakers must be at least 1. Using auto-detect instead.")
                    num_speakers = None
                else:
                    print(f"✓ Will look for exactly {num_speakers} speaker(s)")
            except ValueError:
                print("ERROR: Invalid input. Using auto-detect instead.")
                num_speakers = None
        else:
            num_speakers = None
            print("✓ Will auto-detect number of speakers")
    
    # Whisper model size (larger = better quality but slower)
    # Options: 'tiny', 'base', 'small', 'medium', 'large'
    # Use 'tiny' for fastest, 'base' for balanced, 'small' for better quality
    whisper_model = os.getenv("WHISPER_MODEL", "base")  # Can override with env var
    
    # TEST MODE: Process only first 1 minute for quick testing
    # Get audio duration
    max_duration = None
    waveform, sample_rate = load_audio_file(str(audio_path))
    total_duration = waveform.shape[1] / sample_rate
    
    if test_mode:
        max_duration = 60.0  # 1 minute for quick testing
    
    print("="*60)
    print("Speaker-Attributed Transcription")
    print("="*60)
    print(f"Audio file: {audio_file}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    if test_mode:
        print(f"TEST MODE: Processing first 10% ({max_duration/60:.1f} minutes)")
    else:
        print(f"Processing full file")
    print(f"Expected speakers: {num_speakers if num_speakers else 'auto-detect'}")
    print(f"Whisper model: {whisper_model}")
    print("="*60)
    
    # Step 1: Speaker diarization
    logger.info("main: ===== STEP 1: DIARIZATION =====")
    step1_start = time.time()
    diarization = perform_diarization(
        str(audio_path),
        hf_token=hf_token,
        num_speakers=num_speakers,
        max_duration=max_duration  # None for full file, or limited for test mode
    )
    logger.info(f"main: ✓ Step 1 completed in {time.time() - step1_start:.1f}s")
    
    # Step 2: Transcription + combination
    logger.info("main: ===== STEP 2: TRANSCRIPTION =====")
    step2_start = time.time()
    results = combine_diarization_and_transcription(
        str(audio_path),
        diarization,
        whisper_model_name=whisper_model
    )
    logger.info(f"main: ✓ Step 2 completed in {time.time() - step2_start:.1f}s")
    
    # Step 3: Save results
    logger.info("main: ===== STEP 3: SAVING RESULTS =====")
    output_base = audio_path.stem
    save_transcript(results, f"{output_base}_transcript.txt", format="txt")
    save_transcript(results, f"{output_base}_transcript.json", format="json")
    logger.info("main: ✓ Results saved")
    
    # Print summary
    total_time = time.time() - start_time
    logger.info("main: ===== SUMMARY =====")
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total segments: {len(results)}")
    speakers = set(r["speaker"] for r in results)
    print(f"Speakers found: {sorted(speakers)}")
    total_duration = sum(r["end"] - r["start"] for r in results)
    print(f"Total speech duration: {total_duration/60:.1f} minutes")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print("="*60)
    logger.info(f"main: COMPLETE - Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")


if __name__ == "__main__":
    main()

