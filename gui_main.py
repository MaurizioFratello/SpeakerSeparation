#!/usr/bin/env python3
"""
Main entry point for GUI application.

PURPOSE: Launch PySide6 GUI for speaker diarization and transcription.
Handles application initialization and error handling.

CONTEXT: Standalone GUI launcher. Can be run directly or via conda environment.
"""

import sys
import os
from pathlib import Path

# Load .env for HuggingFace token
from dotenv import load_dotenv
load_dotenv()

# Ensure model/cache writes stay inside the repo (helps on restricted systems).
_repo_root = Path(__file__).resolve().parent
_cache_root = _repo_root / ".cache"
_hf_home = _cache_root / "huggingface"
_hf_hub_cache = _hf_home / "hub"
_cache_root.mkdir(parents=True, exist_ok=True)
_hf_hub_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
os.environ.setdefault("HF_HOME", str(_hf_home))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_hf_hub_cache))

# PyTorch 2.6+ defaults torch.load(..., weights_only=True). pyannote checkpoints
# loaded via Lightning still need full unpickling (Hugging Face weights = trusted).
import torch

if not getattr(torch, "_speaker_sep_torch_load_patched", False):
    _orig_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        # Lightning passes weights_only=None; PyTorch 2.6+ then defaults to True.
        if kwargs.get("weights_only", None) is None:
            kwargs = {**kwargs, "weights_only": False}
        return _orig_torch_load(*args, **kwargs)

    _torch_load_compat.__doc__ = getattr(_orig_torch_load, "__doc__", None)
    torch.load = _torch_load_compat
    torch._speaker_sep_torch_load_patched = True

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from gui.main_window import MainWindow


def main():
    """Launch GUI application."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Speaker Diarization & Transcription")
    
    # Check for required environment variable
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Konfiguration fehlt")
        msg.setText("HUGGINGFACE_TOKEN nicht gefunden")
        msg.setInformativeText(
            "Bitte erstellen Sie eine .env-Datei im Projektverzeichnis mit:\n\n"
            "HUGGINGFACE_TOKEN=hf_your_token_here\n\n"
            "Token erhalten Sie unter: https://hf.co/settings/tokens"
        )
        msg.exec()
        return 1
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

