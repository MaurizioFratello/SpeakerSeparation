#!/usr/bin/env python3
"""
Main entry point for GUI application.

PURPOSE: Launch PySide6 GUI for speaker diarization and transcription.
Handles application initialization and error handling.

CONTEXT: Standalone GUI launcher. Can be run directly or via conda environment.
"""

import sys
import os

# Load .env for HuggingFace token
from dotenv import load_dotenv
load_dotenv()

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

