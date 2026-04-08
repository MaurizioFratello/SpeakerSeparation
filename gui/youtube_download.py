"""
YouTube download helper built on yt-dlp.
"""

import re
import tempfile
from pathlib import Path
from typing import Any, Dict

try:
    from yt_dlp import YoutubeDL as _YoutubeDL
except Exception:
    _YoutubeDL = None


def is_youtube_url(url: str) -> bool:
    if not url:
        return False
    pattern = r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$"
    return re.match(pattern, url.strip()) is not None


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^\w\-. ]+", "", value or "").strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:100] or "youtube_audio"


def download_youtube_audio(url: str, output_dir: str | None = None) -> Dict[str, Any]:
    """
    Download best YouTube audio with yt-dlp and return metadata.
    """
    if not is_youtube_url(url):
        raise ValueError("Invalid YouTube URL")

    if _YoutubeDL is None:
        try:
            from yt_dlp import YoutubeDL as downloaded_cls
        except Exception as exc:
            raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp") from exc
    else:
        downloaded_cls = _YoutubeDL

    if output_dir:
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = Path(tempfile.mkdtemp(prefix="yt_audio_"))

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(base_dir / "%(id)s_%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with downloaded_cls(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    path = Path(downloaded_path)
    if not path.exists():
        raise RuntimeError("yt-dlp finished but no audio file was produced")

    return {
        "audio_path": str(path),
        "title": info.get("title") or _safe_stem(path.stem),
        "video_id": info.get("id") or _safe_stem(path.stem),
        "source_url": url,
        "temp_dir": str(base_dir),
    }
