#!/usr/bin/env python3
"""
Integration tests for GPU-accelerated speaker diarization.
"""
import unittest
import os
import sys

# Load .env and disable telemetry FIRST
from dotenv import load_dotenv
load_dotenv()
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'

import torch
import numpy as np
import subprocess
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


class TestGPUIntegration(unittest.TestCase):
    """Integration tests for GPU processing."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load audio and token."""
        cls.token = os.getenv("HUGGINGFACE_TOKEN")
        if not cls.token:
            raise ValueError("HUGGINGFACE_TOKEN not found in .env")

        # Load test audio chunk
        cls.audio_file = "test_chunk_2min.m4a"
        if not os.path.exists(cls.audio_file):
            raise FileNotFoundError(f"Test audio not found: {cls.audio_file}")

        # Load audio using ffmpeg
        cmd = ['ffmpeg', '-i', cls.audio_file,
               '-f', 's16le', '-acodec', 'pcm_s16le',
               '-ar', '16000', '-ac', '1', '-']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        audio_data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        cls.waveform = torch.from_numpy(audio_data).unsqueeze(0)
        cls.audio_input = {"waveform": cls.waveform, "sample_rate": 16000}

    def test_pipeline_loads_on_device(self):
        """Test that pipeline loads and moves to correct device."""
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self.token
        )

        # Determine expected device
        if torch.cuda.is_available():
            expected_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            expected_device = "mps"
        else:
            expected_device = "cpu"

        device = torch.device(expected_device)
        pipeline.to(device)

        # Verify pipeline device
        self.assertEqual(str(pipeline.device), expected_device,
                        f"Pipeline should be on {expected_device}")

    def test_diarization_produces_valid_output(self):
        """Test that diarization produces valid output on GPU."""
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self.token
        )

        # Move to best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        pipeline.to(device)

        # Run diarization
        with ProgressHook() as hook:
            diarization = pipeline(self.audio_input, hook=hook, num_speakers=2)

        # Extract annotation
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization

        # Verify output is valid
        self.assertIsNotNone(annotation, "Diarization should produce output")
        speakers = annotation.labels()
        self.assertGreater(len(speakers), 0, "Should detect at least one speaker")
        self.assertLessEqual(len(speakers), 3, "Should detect reasonable number of speakers")

        # Verify timing makes sense
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            self.assertGreaterEqual(segment.start, 0, "Segment start should be non-negative")
            self.assertGreater(segment.end, segment.start, "Segment end should be after start")
            self.assertLessEqual(segment.end, 120, "Segment should be within 2-minute test chunk")

    def test_mps_device_available_on_apple_silicon(self):
        """Test MPS availability on Apple Silicon."""
        if sys.platform != "darwin":
            self.skipTest("MPS only available on macOS")

        # Check if MPS is available
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if has_mps:
            # Test that we can create tensors on MPS
            device = torch.device("mps")
            tensor = torch.ones(10, device=device)
            self.assertEqual(str(tensor.device), "mps:0", "Should create tensor on MPS device")
        else:
            self.skipTest("MPS not available on this Mac")

    def test_cpu_fallback_works(self):
        """Test that CPU fallback works when forced."""
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self.token
        )

        # Force CPU
        device = torch.device("cpu")
        pipeline.to(device)

        # Run diarization
        diarization = pipeline(self.audio_input, num_speakers=2)

        # Extract annotation
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization

        # Verify output
        self.assertIsNotNone(annotation, "CPU diarization should work")
        speakers = annotation.labels()
        self.assertGreater(len(speakers), 0, "Should detect speakers on CPU")

    def test_device_movement_doesnt_break_pipeline(self):
        """Test that moving pipeline between devices doesn't break it."""
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self.token
        )

        # Start on CPU
        pipeline.to(torch.device("cpu"))
        self.assertEqual(str(pipeline.device), "cpu")

        # Move to best GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            self.assertEqual(str(pipeline.device), "cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
            self.assertEqual(str(pipeline.device), "mps")

        # Pipeline should still work
        diarization = pipeline(self.audio_input, num_speakers=2)
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization

        self.assertGreater(len(annotation.labels()), 0, "Pipeline should work after device movement")


if __name__ == "__main__":
    # Check for required test file
    if not os.path.exists("test_chunk_2min.m4a"):
        print("ERROR: test_chunk_2min.m4a not found")
        print("Run: python create_test_chunk.py")
        sys.exit(1)

    unittest.main()
