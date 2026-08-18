#!/usr/bin/env python3
"""
Unit tests for device detection logic.
"""
import unittest
from unittest.mock import patch, MagicMock
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import after setting up env
from dotenv import load_dotenv
load_dotenv()
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'

# Import the function to test (we'll load it from transcribe_simple)
import importlib.util
spec = importlib.util.spec_from_file_location("transcribe_simple", "transcribe_simple.py")
transcribe_simple = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcribe_simple)

get_device = transcribe_simple.get_device


class TestDeviceDetection(unittest.TestCase):
    """Test cases for device detection logic."""

    def test_cuda_priority(self):
        """Test that CUDA is preferred when available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.backends.mps.is_available', return_value=True):
                with patch.object(transcribe_simple, "_select_cuda_device_index", return_value=0):
                    device = get_device()
                    self.assertEqual(device, "cuda", "CUDA should be preferred over MPS")

    def test_mps_detection(self):
        """Test that MPS is selected on Apple Silicon."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = get_device()
                self.assertEqual(device, "mps", "MPS should be selected when CUDA unavailable")

    def test_cpu_fallback(self):
        """Test that CPU is used when no GPU available."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = get_device()
                self.assertEqual(device, "cpu", "CPU should be fallback when no GPU available")

    def test_whisper_uses_mps(self):
        """Test that Whisper uses MPS when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = get_device(for_whisper=True)
                self.assertEqual(device, "mps", "Whisper should use MPS when available")

    def test_whisper_allows_cuda(self):
        """Test that Whisper can use CUDA when available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch.object(transcribe_simple, "_select_cuda_device_index", return_value=0):
                device = get_device(for_whisper=True)
                self.assertEqual(device, "cuda", "Whisper should use CUDA when available")

    def test_actual_device_detection(self):
        """Test actual device detection on current hardware."""
        device = get_device()
        self.assertTrue(
            device in ("mps", "cpu") or device == "cuda" or device.startswith("cuda:"),
            "Device should be one of the valid options",
        )

        # Verify it matches torch's actual capabilities
        if torch.cuda.is_available():
            self.assertTrue(
                device == "cuda" or device.startswith("cuda:"),
                "Should detect CUDA when available",
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.assertEqual(device, "mps", "Should detect MPS when available")
        else:
            self.assertEqual(device, "cpu", "Should fallback to CPU")

    def test_device_is_valid_torch_device(self):
        """Test that returned device string can create valid torch.device."""
        device_name = get_device()
        try:
            device = torch.device(device_name)
            self.assertIsInstance(device, torch.device)
        except Exception as e:
            self.fail(f"get_device() returned invalid device string: {e}")

    def test_transcription_language_normalization(self):
        """Test supported UI/API language values."""
        normalize = transcribe_simple.normalize_transcription_language

        self.assertIsNone(normalize(None))
        self.assertIsNone(normalize("Automatic"))
        self.assertEqual(normalize("English"), "en")
        self.assertEqual(normalize("deutsch"), "de")

        with self.assertRaises(ValueError):
            normalize("French")

    def test_faster_whisper_selected_for_cuda_when_available(self):
        """Test faster-whisper is preferred on CUDA when installed."""
        with patch("importlib.util.find_spec", return_value=object()):
            backend = transcribe_simple._select_whisper_backend("cuda")
            self.assertEqual(backend, "faster-whisper")

    def test_openai_whisper_selected_for_mps(self):
        """Test MPS uses OpenAI Whisper because faster-whisper has no MPS device."""
        with patch("importlib.util.find_spec", return_value=object()):
            backend = transcribe_simple._select_whisper_backend("mps")
            self.assertEqual(backend, "openai-whisper")

    def test_speaker_turn_merger_combines_consecutive_same_speaker(self):
        """Test Whisper short segments are merged into speaker turns."""
        class FakeAnnotation:
            def argmax(self, segment):
                return "SPEAKER_00" if segment.start < 4.0 else "SPEAKER_01"

        emitted = []
        merged = []
        merger = transcribe_simple._SpeakerTurnMerger(
            merged,
            emitted.append,
            FakeAnnotation(),
        )

        merger.add(0.0, 1.0, "Hello")
        merger.add(1.0, 2.0, "again")
        merger.add(4.0, 5.0, "Guten Tag")
        merger.flush()

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertEqual(merged[0]["start"], 0.0)
        self.assertEqual(merged[0]["end"], 2.0)
        self.assertEqual(merged[0]["text"], "Hello again")
        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")
        self.assertEqual(merged[1]["text"], "Guten Tag")
        self.assertEqual(emitted, merged)

    def test_whisper_path_emits_merged_speaker_turns(self):
        """Test Whisper path returns merged same-speaker blocks."""
        class FakeAnnotation:
            def argmax(self, segment):
                return "SPEAKER_00" if segment.start < 4.0 else "SPEAKER_01"

        emitted = []
        merged = []
        fake_chunks = [(0, 1, 0.0, 5.0, object())]
        fake_result_segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "again"},
            {"start": 4.0, "end": 5.0, "text": "Guten Tag"},
        ]

        with patch.object(transcribe_simple, "get_device", return_value="cpu"), \
             patch.object(transcribe_simple, "_select_whisper_backend", return_value="faster-whisper"), \
             patch.object(transcribe_simple, "_load_faster_whisper_model", return_value=(object(), "tiny", "int8")), \
             patch.object(transcribe_simple, "_iter_audio_chunks", return_value=fake_chunks), \
             patch.object(transcribe_simple, "_write_temp_wav", return_value="/tmp/fake.wav"), \
             patch.object(transcribe_simple, "_transcribe_chunk_with_faster_whisper", return_value=fake_result_segments), \
             patch.object(transcribe_simple.os, "unlink"):
            transcribe_simple._transcribe_with_whisper(
                waveform=object(),
                annotation=FakeAnnotation(),
                language="en",
                all_segments=merged,
                progress_callback=None,
                segment_callback=emitted.append,
                check_interrupt=None,
            )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["text"], "Hello again")
        self.assertEqual(merged[1]["text"], "Guten Tag")
        self.assertEqual(emitted, merged)

    def test_select_cuda_device_prefers_smi_free_memory(self):
        """WSL should pick the GPU nvidia-smi reports as actually free."""
        smi = {
            "NVIDIA GeForce RTX 5090": 30000,
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 500,
        }
        names = [
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "NVIDIA GeForce RTX 5090",
        ]
        with patch.object(transcribe_simple, "_nvidia_smi_free_mib_by_name", return_value=smi), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.get_device_name", side_effect=lambda i: names[i]):
            self.assertEqual(transcribe_simple._select_cuda_device_index(), 1)

    def test_get_device_returns_cuda_index_when_not_zero(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch.object(transcribe_simple, "_select_cuda_device_index", return_value=1):
            self.assertEqual(get_device(), "cuda:1")

    def test_parse_nvidia_smi_free_memory_by_name(self):
        smi_output = (
            "NVIDIA GeForce RTX 5090, 29965\n"
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 558\n"
        )
        fake = MagicMock(stdout=smi_output)
        with patch.object(transcribe_simple.subprocess, "run", return_value=fake):
            mapping = transcribe_simple._nvidia_smi_free_mib_by_name()
        self.assertEqual(mapping["NVIDIA GeForce RTX 5090"], 29965)
        self.assertEqual(mapping["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"], 558)


if __name__ == "__main__":
    unittest.main()
