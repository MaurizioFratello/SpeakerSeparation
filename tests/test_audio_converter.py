"""Video format support and Windows-to-WSL path mapping for GUI media input."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.audio_converter import (
    FILE_DIALOG_FILTER,
    convert_to_wav,
    is_supported_format,
    normalize_input_path,
)


class TestSupportedMediaFormats(unittest.TestCase):
    def test_is_supported_format_accepts_video_containers(self) -> None:
        self.assertTrue(is_supported_format("meeting.mp4"))
        self.assertTrue(is_supported_format("meeting.MKV"))
        self.assertTrue(is_supported_format("meeting.mov"))
        self.assertTrue(is_supported_format("meeting.avi"))
        self.assertTrue(is_supported_format("meeting.m4v"))

    def test_is_supported_format_still_rejects_unknown_extensions(self) -> None:
        self.assertFalse(is_supported_format("notes.txt"))
        self.assertFalse(is_supported_format("slides.pptx"))

    def test_file_dialog_filter_includes_video_extensions(self) -> None:
        self.assertIn("*.mp4", FILE_DIALOG_FILTER)
        self.assertIn("*.mkv", FILE_DIALOG_FILTER)
        self.assertIn("*.mov", FILE_DIALOG_FILTER)


class TestNormalizeInputPath(unittest.TestCase):
    def test_normalize_windows_drive_path_in_wsl(self) -> None:
        windows_path = r"E:\PhoneLink\BB JF-20260814_110255UTC-Meeting Recording.mp4"
        with patch("gui.audio_converter.running_in_wsl", return_value=True):
            self.assertEqual(
                normalize_input_path(windows_path),
                "/mnt/e/PhoneLink/BB JF-20260814_110255UTC-Meeting Recording.mp4",
            )

    def test_normalize_quoted_and_forward_slash_windows_paths_in_wsl(self) -> None:
        with patch("gui.audio_converter.running_in_wsl", return_value=True):
            self.assertEqual(
                normalize_input_path('"E:/PhoneLink/meeting.mp4"'),
                "/mnt/e/PhoneLink/meeting.mp4",
            )

    def test_normalize_wsl_unc_path_in_wsl(self) -> None:
        unc = r"\\wsl.localhost\Ubuntu\home\brude\meeting.mp4"
        with patch("gui.audio_converter.running_in_wsl", return_value=True):
            self.assertEqual(normalize_input_path(unc), "/home/brude/meeting.mp4")

    def test_normalize_leaves_linux_paths_unchanged(self) -> None:
        path = "/mnt/e/PhoneLink/meeting.mp4"
        with patch("gui.audio_converter.running_in_wsl", return_value=True):
            self.assertEqual(normalize_input_path(path), path)

    def test_normalize_leaves_windows_paths_unchanged_outside_wsl(self) -> None:
        windows_path = r"E:\PhoneLink\meeting.mp4"
        with patch("gui.audio_converter.running_in_wsl", return_value=False):
            self.assertEqual(normalize_input_path(windows_path), windows_path)


class TestConvertToWavPathMapping(unittest.TestCase):
    def test_convert_to_wav_uses_wsl_mapped_path_when_file_missing(self) -> None:
        with patch("gui.audio_converter.running_in_wsl", return_value=True):
            with self.assertRaises(FileNotFoundError) as ctx:
                convert_to_wav(r"E:\PhoneLink\missing.mp4")
            self.assertIn("/mnt/e/PhoneLink/missing.mp4", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
