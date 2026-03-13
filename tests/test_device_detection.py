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

    def test_whisper_forces_cpu(self):
        """Test that Whisper always uses CPU (doesn't support MPS)."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = get_device(for_whisper=True)
                self.assertEqual(device, "cpu", "Whisper should always use CPU even when MPS available")

    def test_whisper_allows_cuda(self):
        """Test that Whisper can use CUDA when available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device(for_whisper=True)
            self.assertEqual(device, "cuda", "Whisper should use CUDA when available")

    def test_actual_device_detection(self):
        """Test actual device detection on current hardware."""
        device = get_device()
        self.assertIn(device, ["cuda", "mps", "cpu"], "Device should be one of the valid options")

        # Verify it matches torch's actual capabilities
        if torch.cuda.is_available():
            self.assertEqual(device, "cuda", "Should detect CUDA when available")
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


if __name__ == "__main__":
    unittest.main()
