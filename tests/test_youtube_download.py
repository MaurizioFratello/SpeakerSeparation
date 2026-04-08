from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.youtube_download import download_youtube_audio, is_youtube_url


def test_is_youtube_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert not is_youtube_url("https://example.com/video")


def test_download_youtube_audio_success(tmp_path):
    audio_file = tmp_path / "abc_title.m4a"
    audio_file.write_text("dummy", encoding="utf-8")

    class DummyYDL:
        def __init__(self, _opts):
            self._info = {"id": "abc", "title": "Title"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, _url, download=True):
            assert download is True
            return self._info

        def prepare_filename(self, _info):
            return str(audio_file)

    with patch("gui.youtube_download._YoutubeDL", DummyYDL):
        result = download_youtube_audio("https://youtu.be/abc", str(tmp_path))

    assert Path(result["audio_path"]).exists()
    assert result["title"] == "Title"
    assert result["video_id"] == "abc"
