"""Path validation for the transcription microservice."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import service.jobs as jobs


class TestValidateAudioPath(unittest.TestCase):
    def test_accepts_file_under_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d).resolve()
            old = jobs.settings.data_dir
            jobs.settings.data_dir = base
            try:
                f = base / "clip.wav"
                f.write_bytes(b"fake")
                resolved = jobs.validate_audio_path(str(f))
                self.assertEqual(Path(resolved), f.resolve())
            finally:
                jobs.settings.data_dir = old

    def test_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d).resolve()
            old = jobs.settings.data_dir
            jobs.settings.data_dir = base
            try:
                outside = base.parent / "secret.wav"
                outside.write_bytes(b"x")
                with self.assertRaises(ValueError):
                    jobs.validate_audio_path(str(outside))
            finally:
                jobs.settings.data_dir = old

    def test_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d).resolve()
            old = jobs.settings.data_dir
            jobs.settings.data_dir = base
            try:
                missing = base / "nope.wav"
                with self.assertRaises(ValueError):
                    jobs.validate_audio_path(str(missing))
            finally:
                jobs.settings.data_dir = old


if __name__ == "__main__":
    unittest.main()
