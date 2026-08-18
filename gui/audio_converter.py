"""
Audio format conversion utilities for GUI.

PURPOSE: Convert audio and video files (MP3, M4A, MP4, WAV, ...) to WAV format
required by the transcription backend (16kHz, mono, PCM).

CONTEXT: The backend expects audio in a specific format. This module handles
format detection, Windows-to-WSL path mapping, and conversion using ffmpeg.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path


SUPPORTED_AUDIO_FORMATS = {'.mp3', '.m4a', '.wav', '.aiff', '.aif', '.flac', '.ogg', '.wma', '.webm'}
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.mpeg', '.mpg'}
SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
SUPPORTED_FORMATS_LABEL = "MP3, M4A, WAV, AIFF, FLAC, OGG, WMA, WEBM, MP4, MKV, MOV, AVI, M4V"

_DIALOG_EXTS = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_FORMATS))
FILE_DIALOG_FILTER = f"Audio & Video ({_DIALOG_EXTS});;All Files (*)"

_DRIVE_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_WSL_UNC_RE = re.compile(
    r"^[/\\]{2}(?:wsl\.localhost|wsl\$)[/\\][^/\\]+[/\\](.*)$",
    re.IGNORECASE,
)


def running_in_wsl() -> bool:
    """Return True when this process is running inside WSL."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(
            encoding="utf-8", errors="ignore"
        ).lower()
    except OSError:
        return False


def normalize_input_path(file_path: str) -> str:
    """
    Map Windows / WSLg paths to a local path this process can open.

    Drag-and-drop from Explorer onto a WSL GUI often yields ``E:\\...`` or
    ``\\\\wsl.localhost\\Ubuntu\\...`` instead of ``/mnt/e/...``.
    """
    if not file_path:
        return file_path

    path = file_path.strip().strip('"')
    if path.lower().startswith("file://"):
        path = path[7:]
        if re.match(r"^/[A-Za-z]:", path):
            path = path[1:]

    if not running_in_wsl():
        return path

    unc = _WSL_UNC_RE.match(path)
    if unc:
        rest = unc.group(1).replace("\\", "/")
        return rest if rest.startswith("/") else f"/{rest}"

    drive = _DRIVE_PATH_RE.match(path)
    if drive:
        rest = drive.group(2).replace("\\", "/")
        return f"/mnt/{drive.group(1).lower()}/{rest}"

    return path.replace("\\", "/") if "\\" in path else path


def is_supported_format(file_path: str) -> bool:
    """
    Check if audio/video file format is supported.
    
    Args:
        file_path: Path to media file
    
    Returns:
        True if format is supported, False otherwise
    """
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_FORMATS


def convert_to_wav(input_file: str) -> str:
    """
    Convert audio or video file to WAV format required by backend.
    
    PURPOSE: Convert any supported media format to 16kHz mono WAV for processing.
    Uses ffmpeg for conversion, creating a temporary file that must be cleaned up.
    
    CONTEXT: Backend requires specific audio format (16kHz, mono, PCM). This function
    handles Windows/WSL path mapping, format detection, and conversion transparently.
    
    Args:
        input_file: Path to input audio or video file (any supported format)
    
    Returns:
        Path to temporary WAV file (must be cleaned up by caller)
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is not supported
        subprocess.CalledProcessError: If ffmpeg conversion fails
        RuntimeError: If ffmpeg is not installed or not found in PATH
    """
    input_file = normalize_input_path(input_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Audio file not found: {input_file}")
    
    # Check if format is supported
    if not is_supported_format(input_file):
        ext = Path(input_file).suffix.lower()
        raise ValueError(
            f"Unsupported audio format: {ext}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    
    # Check if already WAV format - verify if conversion is needed
    ext = Path(input_file).suffix.lower()
    if ext == '.wav':
        # Check if already in correct format (16kHz mono)
        # For simplicity, we'll always convert to ensure correct format
        # In production, could add format detection here
        pass
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: sudo apt-get install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )
    
    # Create temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    temp_wav_path = temp_wav.name
    
    try:
        # Convert to WAV: 16kHz, mono, PCM
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-vn',               # Ignore video streams (meeting recordings, etc.)
            '-ar', '16000',      # Sample rate: 16kHz
            '-ac', '1',          # Channels: mono
            '-acodec', 'pcm_s16le',  # Codec: PCM 16-bit little-endian
            '-y',                # Overwrite output file
            temp_wav_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return temp_wav_path
    
    except subprocess.CalledProcessError as e:
        # Clean up temp file on error
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        
        error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        raise RuntimeError(f"Audio conversion failed: {error_msg}")


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary audio file.
    
    PURPOSE: Remove temporary WAV file created by convert_to_wav().
    
    Args:
        file_path: Path to temporary file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except OSError:
        # Ignore errors during cleanup (file might already be deleted)
        pass

