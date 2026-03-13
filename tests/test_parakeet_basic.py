#!/usr/bin/env python3
"""
Unit tests for NVIDIA Parakeet model loading and basic functionality.
"""
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import unittest
import torch
import numpy as np
import subprocess
import tempfile
import soundfile as sf
import nemo.collections.asr as nemo_asr


class TestParakeetBasic(unittest.TestCase):
    """Basic unit tests for Parakeet model loading and device handling."""

    def test_model_loads(self):
        """Test that Parakeet model loads successfully."""
        print("\n[TEST] Loading Parakeet model...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        self.assertIsNotNone(asr_model)
        print("✓ Model loaded successfully")

    def test_mps_device_movement(self):
        """Test that model can be moved to MPS device."""
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            self.skipTest("MPS not available")

        print("\n[TEST] Testing MPS device movement...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        asr_model = asr_model.to(torch.device("mps"))
        # Model should move without error
        self.assertTrue(True)
        print("✓ Model moved to MPS successfully")

    def test_transcribe_with_timestamps(self):
        """Test transcription with timestamps on test audio."""
        print("\n[TEST] Testing transcription with timestamps...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )

        # Use test chunk
        test_audio = "tests/test_chunk_2min.m4a"
        if not os.path.exists(test_audio):
            self.skipTest(f"Test audio not found: {test_audio}")

        # Load audio using ffmpeg subprocess (avoids torchcodec/FFmpeg extension issues)
        # Convert m4a to waveform, then save as WAV for NeMo
        cmd = ['ffmpeg', '-i', test_audio,
               '-f', 's16le', '-acodec', 'pcm_s16le',
               '-ar', '16000', '-ac', '1', '-']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        audio_data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio_data).unsqueeze(0)

        # Save waveform to temporary WAV file (NeMo requires file path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_audio_path = tmp_file.name
            # Convert back to int16 for saving
            audio_int16 = (waveform.squeeze().numpy() * 32768).astype(np.int16)
            sf.write(tmp_audio_path, audio_int16, 16000)

        try:
            hypotheses = asr_model.transcribe(
                [tmp_audio_path],
                timestamps=True
            )
        finally:
            # Clean up temp file
            os.unlink(tmp_audio_path)

        # Verify output structure
        self.assertIsNotNone(hypotheses)
        self.assertTrue(len(hypotheses) > 0)
        self.assertIn('segment', hypotheses[0].timestamp)

        segments = hypotheses[0].timestamp['segment']
        self.assertGreater(len(segments), 0)
        print(f"✓ Transcription complete - {len(segments)} segments")

        # Verify segment structure
        first_seg = segments[0]
        self.assertIn('start', first_seg)
        self.assertIn('end', first_seg)
        self.assertIn('segment', first_seg)
        print(f"✓ Segment structure verified: {first_seg['start']:.1f}s - {first_seg['end']:.1f}s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
