"""
Main GUI window for Speaker Diarization & Transcription application.

PURPOSE: Provides user-friendly interface for audio transcription with
drag & drop, speaker selection, live transcript display, and auto-save.

CONTEXT: PySide6-based GUI that integrates with transcription backend
via worker thread. Supports streaming transcript updates and cancellation.
"""

import os
import time
import logging
import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QRadioButton, QSpinBox, QTextEdit, QLabel, QFileDialog, QComboBox, QCheckBox,
    QMessageBox, QProgressBar, QStatusBar, QGroupBox, QButtonGroup, QLineEdit
)
from PySide6.QtCore import Qt, QSize, Signal, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont

from gui.transcription_worker import TranscriptionWorker
from gui.audio_converter import is_supported_format
from gui.markdown_export import segments_to_markdown, merge_consecutive_same_speaker
from gui.youtube_download import download_youtube_audio, is_youtube_url

# Import for pipeline loading
from dotenv import load_dotenv
load_dotenv()
from pyannote.audio import Pipeline
import torch

# Setup logging - use separate logger to avoid conflicts
# Set to WARNING level to suppress debug/info messages
logger = logging.getLogger('main_window')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [GUI] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # Only show warnings and errors


class DragDropWidget(QLabel):
    """
    Custom widget for drag & drop file input.
    
    PURPOSE: Visual drag & drop area with feedback for audio file selection.
    Provides alternative to file dialog for better UX.
    """
    
    fileDropped = Signal(str)  # Emitted when file is dropped
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #D1D1D6;
                border-radius: 12px;
                background-color: #F5F5F7;
                padding: 60px 40px;
                min-height: 120px;
                color: #1D1D1F;
                font-size: 13px;
                line-height: 1.6;
            }
            QLabel:hover {
                border-color: #007AFF;
                background-color: #EBF5FF;
            }
        """)
        self.setText("Drop audio file here or click to select\n\nSupported formats: MP3, M4A, WAV, AIFF, FLAC, WEBM")
        self._original_style = self.styleSheet()
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event - validate and accept if valid."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and len(urls) == 1:
                file_path = urls[0].toLocalFile()
                if os.path.isfile(file_path) and is_supported_format(file_path):
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        QLabel {
                            border: 2px dashed #007AFF;
                            border-radius: 12px;
                            background-color: #EBF5FF;
                            padding: 60px 40px;
                            min-height: 120px;
                            color: #1D1D1F;
                            font-size: 13px;
                        }
                    """)
                    return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Restore original style when drag leaves."""
        self.setStyleSheet(self._original_style)
        super().dragLeaveEvent(event)
    
    def dropEvent(self, event: QDropEvent):
        """Handle file drop - emit signal with file path."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.fileDropped.emit(file_path)
        self.setStyleSheet(self._original_style)
        event.acceptProposedAction()
    
    def mousePressEvent(self, event):
        """Open file dialog when clicked."""
        if event.button() == Qt.LeftButton:
            self._open_file_dialog()
    
    def _open_file_dialog(self):
        """Open file selection dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Audio-Datei auswählen",
            "",
            "Audio Files (*.mp3 *.m4a *.wav *.aiff *.aif *.flac *.ogg *.wma *.webm);;All Files (*)"
        )
        if file_path:
            self.fileDropped.emit(file_path)


class MainWindow(QMainWindow):
    """
    Main application window for transcription GUI.
    
    PURPOSE: Central GUI component providing all user interaction elements.
    Manages worker thread, displays streaming transcript, handles file operations.
    
    CONTEXT: PySide6 main window pattern. Coordinates between UI components
    and background worker thread via signals/slots.
    """
    LANGUAGE_OPTIONS = [
        ("Automatisch", "auto"),
        ("Deutsch", "de"),
        ("Englisch", "en"),
    ]

    def _selected_backend_label(self, language_code: str, strict_mode: bool) -> str:
        """Return human-readable backend label for current language settings."""
        if strict_mode and language_code in {"de", "en"}:
            return f"Whisper ({language_code})"
        return "Parakeet"
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speaker Diarization & Transcription")
        self.setMinimumSize(QSize(900, 700))
        self.resize(QSize(1000, 800))
        
        # State
        self._current_audio_file: Optional[str] = None
        self._worker: Optional[TranscriptionWorker] = None
        self._is_processing = False
        self._transcript_segments: list = []
        self._source_mode = "file"
        self._youtube_url: Optional[str] = None
        self._youtube_title: Optional[str] = None
        self._youtube_video_id: Optional[str] = None
        self._download_temp_dir: Optional[str] = None
        self._pipeline = None  # Pre-loaded pipeline (loaded in main thread to avoid threading issues)
        self._pipeline_loading = False  # Track if pipeline is currently loading
        
        # Setup UI
        self._setup_ui()
        
        # Load pipeline in background after UI is shown
        # This ensures pipeline is ready when user clicks Start
        self._load_pipeline_async()
    
    def _setup_ui(self):
        """Initialize and layout all UI components."""
        # Set window background color
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FAFAFA;
            }
            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #D1D1D6;
                border-radius: 8px;
                padding: 16px;
                margin-top: 8px;
                font-size: 13px;
                font-weight: 600;
                color: #1D1D1F;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                left: 8px;
            }
            QRadioButton {
                font-size: 13px;
                color: #1D1D1F;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #D1D1D6;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 13px;
                color: #1D1D1F;
                min-width: 60px;
            }
            QSpinBox:disabled {
                background-color: #F5F5F7;
                color: #C7C7CC;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #E5E5EA;
                height: 6px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007AFF;
                border-radius: 4px;
            }
            QStatusBar {
                background-color: #F5F5F7;
                border-top: 1px solid #E5E5EA;
                color: #86868B;
                font-size: 11px;
            }
        """)

        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #FAFAFA;")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Drag & Drop area
        self.drag_drop = DragDropWidget()
        self.drag_drop.fileDropped.connect(self._on_file_selected)
        layout.addWidget(self.drag_drop)

        # Optional YouTube URL input
        youtube_group = QGroupBox("YouTube Source (optional)")
        youtube_layout = QVBoxLayout()
        youtube_layout.setSpacing(8)
        youtube_hint = QLabel("Paste a YouTube URL to download audio and export transcript as markdown.")
        youtube_hint.setStyleSheet("font-size: 12px; color: #6E6E73; font-weight: 400;")
        self.youtube_url_input = QLineEdit()
        self.youtube_url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.youtube_url_input.setMinimumHeight(36)
        self.youtube_url_input.textChanged.connect(self._on_url_changed)
        youtube_layout.addWidget(youtube_hint)
        youtube_layout.addWidget(self.youtube_url_input)
        youtube_group.setLayout(youtube_layout)
        layout.addWidget(youtube_group)
        
        # Settings group
        settings_group = QGroupBox("Speaker Configuration")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(12)

        # Speaker selection
        speaker_layout = QHBoxLayout()
        speaker_layout.setSpacing(12)
        self.speaker_auto = QRadioButton("Automatic detection")
        self.speaker_manual = QRadioButton("Manual:")
        self.speaker_count = QSpinBox()
        self.speaker_count.setMinimum(1)
        self.speaker_count.setMaximum(10)
        self.speaker_count.setValue(2)
        self.speaker_count.setEnabled(False)
        
        self.speaker_auto.setChecked(True)
        self.speaker_manual.toggled.connect(
            lambda checked: self.speaker_count.setEnabled(checked)
        )
        
        speaker_layout.addWidget(self.speaker_auto)
        speaker_layout.addWidget(self.speaker_manual)
        speaker_layout.addWidget(self.speaker_count)
        speaker_layout.addStretch()
        settings_layout.addLayout(speaker_layout)

        # Language selection for ASR
        language_layout = QHBoxLayout()
        language_layout.setSpacing(12)
        language_label = QLabel("Transkriptionssprache:")
        self.language_combo = QComboBox()
        for label, code in self.LANGUAGE_OPTIONS:
            self.language_combo.addItem(label, code)
        self.language_combo.setCurrentIndex(0)  # Automatisch
        self.language_combo.setMinimumWidth(180)
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        settings_layout.addLayout(language_layout)

        # Strict language mode: force Whisper for selected manual language
        strict_layout = QHBoxLayout()
        strict_layout.setSpacing(12)
        self.strict_language_mode = QCheckBox("Strikte Sprache erzwingen (Whisper)")
        self.strict_language_mode.setChecked(True)
        self.strict_language_mode.setToolTip(
            "Aktiv: Deutsch/Englisch wird mit Whisper sprachlich erzwungen.\n"
            "Inaktiv: Parakeet wird verwendet (kann Sprachen mischen)."
        )
        strict_layout.addWidget(self.strict_language_mode)
        strict_layout.addStretch()
        settings_layout.addLayout(strict_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        self.start_button = QPushButton("Start Transcription")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setMinimumHeight(36)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                padding: 8px 24px;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #0051D5;
            }
            QPushButton:disabled {
                background-color: #E5E5EA;
                color: #C7C7CC;
            }
        """)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setMinimumHeight(36)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #F5F5F7;
                color: #1D1D1F;
                padding: 8px 24px;
                border: 1px solid #D1D1D6;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #EBEBED;
            }
            QPushButton:disabled {
                background-color: #F5F5F7;
                color: #C7C7CC;
                border-color: #E5E5EA;
            }
        """)

        self.open_button = QPushButton("Browse...")
        self.open_button.setMinimumHeight(36)
        self.open_button.clicked.connect(self._on_open_clicked)
        self.open_button.setStyleSheet("""
            QPushButton {
                background-color: #F5F5F7;
                color: #1D1D1F;
                padding: 8px 24px;
                border: 1px solid #D1D1D6;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #EBEBED;
            }
        """)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.open_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Transcript display
        transcript_label = QLabel("Transcript")
        transcript_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #1D1D1F; margin-top: 8px;")
        layout.addWidget(transcript_label)

        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setFont(QFont("SF Mono", 11))
        self.transcript_display.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #D1D1D6;
                border-radius: 8px;
                padding: 16px;
                color: #1D1D1F;
                selection-background-color: #007AFF;
                selection-color: white;
            }
        """)
        layout.addWidget(self.transcript_display)
        
        # Status bar with progress
        self.status_bar = QStatusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - loading pipeline in background...")
    
    def _load_pipeline_async(self):
        """
        Load pipeline asynchronously in main thread after GUI is shown.
        
        PURPOSE: Pre-load pipeline so it's ready when user clicks Start.
        Loads in main thread to avoid PyTorch threading issues.
        
        CONTEXT: Called after UI setup. Uses QTimer to defer loading slightly
        so GUI can render first, then loads pipeline in main thread.
        """
        # Defer loading slightly to let GUI render
        QTimer.singleShot(100, self._load_pipeline)

    def _detect_device_name(self) -> str:
        """Detect preferred inference device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _device_status_message(self) -> str:
        """Build a human-readable startup status message for the status bar."""
        device_name = self._detect_device_name()
        if device_name == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                return f"Ready - Using CUDA: {gpu_name}"
            except Exception:
                return "Ready - Using CUDA"
        if device_name == "mps":
            return "Ready - Using MPS (Apple GPU)"
        return "Ready - Using CPU"
    
    def _load_pipeline(self):
        """Load pipeline in main thread."""
        if self._pipeline is not None:
            return  # Already loaded
        
        if self._pipeline_loading:
            return  # Already loading
        
        self._pipeline_loading = True
        logger.debug("Starting pipeline load in main thread...")
        self.status_bar.showMessage("Loading AI pipeline...")

        try:
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                raise RuntimeError("HUGGINGFACE_TOKEN not found in .env file")

            logger.debug("Calling Pipeline.from_pretrained() in main thread...")
            load_start = time.time()

            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=token
            )

            load_elapsed = time.time() - load_start
            logger.debug(f"Pipeline.from_pretrained() completed in {load_elapsed:.2f}s")

            # Move to device
            device_name = self._detect_device_name()
            device = torch.device(device_name)
            self._pipeline.to(device)

            logger.debug(f"Pipeline loaded and moved to {device_name}")
            self.status_bar.showMessage(self._device_status_message())
            self._pipeline_loading = False

        except Exception as e:
            logger.error(f"Pipeline loading failed: {e}", exc_info=True)
            self.status_bar.showMessage(f"Pipeline loading failed: {str(e)}")
            self._pipeline_loading = False
            # Don't show error dialog here - user will see it when clicking Start
    
    def _on_file_selected(self, file_path: str):
        """Handle file selection from drag & drop or file dialog."""
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Fehler", f"Datei nicht gefunden: {file_path}")
            return
        
        if not is_supported_format(file_path):
            ext = Path(file_path).suffix.lower()
            QMessageBox.warning(
                self,
                "Nicht unterstütztes Format",
                f"Das Format '{ext}' wird nicht unterstützt.\n\n"
                f"Unterstützte Formate: MP3, M4A, WAV, AIFF, FLAC, OGG, WMA, WEBM"
            )
            return

        self._cleanup_youtube_temp()
        self._source_mode = "file"
        self._youtube_url = None
        self._youtube_title = None
        self._youtube_video_id = None
        self.youtube_url_input.blockSignals(True)
        self.youtube_url_input.clear()
        self.youtube_url_input.blockSignals(False)

        self._current_audio_file = file_path
        filename = os.path.basename(file_path)
        self.drag_drop.setText(f"{filename}\n\nClick to change selection")
        self.start_button.setEnabled(True)
        self.status_bar.showMessage(f"File loaded: {filename}")

    def _on_url_changed(self, text: str):
        """Handle YouTube URL updates from input."""
        url = text.strip()
        if url:
            self._source_mode = "youtube"
            self._current_audio_file = None
            self.start_button.setEnabled(True)
            self.drag_drop.setText("URL mode active\n\nYou can switch back by selecting a local file")
            self.status_bar.showMessage("YouTube URL ready")
            return

        if self._source_mode == "youtube":
            self._source_mode = "file"
        self._youtube_url = None
        if self._current_audio_file:
            filename = os.path.basename(self._current_audio_file)
            self.drag_drop.setText(f"{filename}\n\nClick to change selection")
            self.start_button.setEnabled(True)
        else:
            self.drag_drop.setText("Drop audio file here or click to select\n\nSupported formats: MP3, M4A, WAV, AIFF, FLAC, WEBM")
            self.start_button.setEnabled(False)
    
    def _on_open_clicked(self):
        """Open file dialog for audio file selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Audio-Datei auswählen",
            "",
            "Audio Files (*.mp3 *.m4a *.wav *.aiff *.aif *.flac *.ogg *.wma *.webm);;All Files (*)"
        )
        if file_path:
            self._on_file_selected(file_path)
    
    def _on_start_clicked(self):
        """Start transcription process."""
        if self._is_processing:
            return

        url_text = self.youtube_url_input.text().strip()
        if url_text:
            if not is_youtube_url(url_text):
                QMessageBox.warning(self, "Invalid URL", "Please enter a valid YouTube URL.")
                return
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.status_bar.showMessage("Downloading YouTube audio...")
                yt_result = download_youtube_audio(url_text)
                self._cleanup_youtube_temp()
                self._source_mode = "youtube"
                self._youtube_url = yt_result["source_url"]
                self._youtube_title = yt_result["title"]
                self._youtube_video_id = yt_result["video_id"]
                self._download_temp_dir = yt_result["temp_dir"]
                self._current_audio_file = yt_result["audio_path"]
                self.drag_drop.setText(f"{self._youtube_title}\n\nDownloaded from YouTube")
            except Exception as e:
                QMessageBox.critical(self, "YouTube download failed", str(e))
                return
            finally:
                QApplication.restoreOverrideCursor()
        elif not self._current_audio_file:
            QMessageBox.warning(self, "Fehler", "Bitte wählen Sie zuerst eine Audio-Datei aus.")
            return
        else:
            self._source_mode = "file"
        
        # Get speaker count
        num_speakers = None
        if self.speaker_manual.isChecked():
            num_speakers = self.speaker_count.value()
        
        # Get selected transcription language
        transcription_language = self.language_combo.currentData()
        strict_language_mode = self.strict_language_mode.isChecked()
        backend_label = self._selected_backend_label(transcription_language, strict_language_mode)
        
        # Clear previous transcript
        self.transcript_display.clear()
        self._transcript_segments = []
        
        # Update UI state
        self._is_processing = True
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.drag_drop.setEnabled(False)
        self.speaker_auto.setEnabled(False)
        self.speaker_manual.setEnabled(False)
        self.speaker_count.setEnabled(False)
        self.language_combo.setEnabled(False)
        self.strict_language_mode.setEnabled(False)
        self.open_button.setEnabled(False)
        self.youtube_url_input.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Check if pipeline is ready
        if self._pipeline is None:
            if self._pipeline_loading:
                QMessageBox.warning(
                    self,
                    "Pipeline wird geladen",
                    "Die Pipeline wird noch geladen. Bitte warten Sie einen Moment und versuchen Sie es erneut."
                )
                self._is_processing = False
                self.start_button.setEnabled(True)
                self.cancel_button.setEnabled(False)
                self.drag_drop.setEnabled(True)
                self.speaker_auto.setEnabled(True)
                self.speaker_manual.setEnabled(True)
                if self.speaker_manual.isChecked():
                    self.speaker_count.setEnabled(True)
                self.language_combo.setEnabled(True)
                self.strict_language_mode.setEnabled(True)
                self.open_button.setEnabled(True)
                self.youtube_url_input.setEnabled(True)
                self.progress_bar.setVisible(False)
                return
            else:
                # Pipeline loading failed - try to load now
                logger.debug("Pipeline not loaded, attempting to load now...")
                self.status_bar.showMessage("Lade Pipeline...")
                try:
                    token = os.getenv("HUGGINGFACE_TOKEN")
                    if not token:
                        raise RuntimeError("HUGGINGFACE_TOKEN not found in .env file")
                    
                    self._pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-community-1",
                        token=token
                    )
                    
                    device_name = self._detect_device_name()
                    device = torch.device(device_name)
                    self._pipeline.to(device)
                    logger.debug(f"Pipeline loaded and moved to {device_name}")
                except Exception as e:
                    logger.error(f"Pipeline loading failed: {e}", exc_info=True)
                    QMessageBox.critical(self, "Fehler", f"Pipeline konnte nicht geladen werden:\n{str(e)}")
                    self._is_processing = False
                    self.start_button.setEnabled(True)
                    self.cancel_button.setEnabled(False)
                    self.drag_drop.setEnabled(True)
                    self.speaker_auto.setEnabled(True)
                    self.speaker_manual.setEnabled(True)
                    if self.speaker_manual.isChecked():
                        self.speaker_count.setEnabled(True)
                    self.language_combo.setEnabled(True)
                    self.strict_language_mode.setEnabled(True)
                    self.open_button.setEnabled(True)
                    self.youtube_url_input.setEnabled(True)
                    self.progress_bar.setVisible(False)
                    return
        
        self.status_bar.showMessage(f"Processing... Backend: {backend_label}")
        
        # Create and start worker with pre-loaded pipeline
        self._worker = TranscriptionWorker(
            audio_file=self._current_audio_file,
            num_speakers=num_speakers,
            transcription_language=transcription_language,
            strict_language_mode=strict_language_mode,
            pipeline=self._pipeline
        )
        
        # Connect signals
        self._worker.progress.connect(self._on_progress)
        self._worker.segment_ready.connect(self._on_segment_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        
        # Start worker thread
        self._worker.start()
    
    def _on_cancel_clicked(self):
        """Cancel ongoing transcription."""
        if self._worker and self._is_processing:
            self._worker.cancel()
            self.status_bar.showMessage("Cancelling...")
    
    def _on_progress(self, message: str, progress: float):
        """Handle progress updates from worker."""
        logger.debug(f"Progress update received: '{message}' ({progress:.2%})")
        self.status_bar.showMessage(message)
        self.progress_bar.setValue(int(progress * 100))
    
    def _on_segment_ready(self, segment: dict):
        """
        Handle streaming segment updates.
        
        PURPOSE: Append new transcript segments immediately for live display.
        Implements auto-scroll logic: only scrolls if user was at bottom.
        
        CONTEXT: Called via signal from worker thread. Updates GUI thread-safely.
        """
        self._transcript_segments.append(segment)
        
        # Format segment for display
        start = segment['start']
        end = segment['end']
        speaker = segment['speaker']
        text = segment['text']
        
        formatted = f"[{start:6.1f}s - {end:6.1f}s] {speaker}: {text}\n"
        
        # Check if user is at bottom (for auto-scroll)
        scrollbar = self.transcript_display.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 10
        
        # Append text
        self.transcript_display.append(formatted)
        
        # Auto-scroll if user was at bottom
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
    
    def _on_finished(self, segments: list):
        """Handle transcription completion."""
        self._is_processing = False
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.drag_drop.setEnabled(True)
        self.speaker_auto.setEnabled(True)
        self.speaker_manual.setEnabled(True)
        if self.speaker_manual.isChecked():
            self.speaker_count.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.strict_language_mode.setEnabled(True)
        self.open_button.setEnabled(True)
        self.youtube_url_input.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if segments:
            merged_segments = merge_consecutive_same_speaker(segments)
            self.status_bar.showMessage(
                f"Transcription completed ({len(segments)} segments -> {len(merged_segments)} blocks)"
            )
            # Auto-save transcript
            self._auto_save_transcript(merged_segments)
        else:
            self.status_bar.showMessage("Transcription cancelled")
        
        # Cleanup worker
        if self._worker:
            self._worker.quit()
            self._worker.wait()
            self._worker = None
        self._cleanup_youtube_temp()
    
    def _on_error(self, error_message: str):
        """Handle errors from worker."""
        self._is_processing = False
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.drag_drop.setEnabled(True)
        self.speaker_auto.setEnabled(True)
        self.speaker_manual.setEnabled(True)
        if self.speaker_manual.isChecked():
            self.speaker_count.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.strict_language_mode.setEnabled(True)
        self.open_button.setEnabled(True)
        self.youtube_url_input.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Error", error_message)
        self.status_bar.showMessage("Error occurred")
        
        # Cleanup worker
        if self._worker:
            self._worker.quit()
            self._worker.wait()
            self._worker = None
        self._cleanup_youtube_temp()
    
    def _auto_save_transcript(self, segments: list):
        """
        Auto-save transcript to file in same directory as source audio.
        
        PURPOSE: Save transcript automatically after completion.
        File saved as {original_filename}_transcript.txt in source directory.
        
        CONTEXT: Called after transcription finishes. Uses same format as GUI display.
        """
        if not self._current_audio_file or not segments:
            return
        
        try:
            # Defensive merge so save/export paths always use grouped speaker turns.
            segments = merge_consecutive_same_speaker(segments)
            source_path = Path(self._current_audio_file)
            if self._source_mode == "youtube":
                exports_dir = Path(__file__).resolve().parent.parent / "exports"
                exports_dir.mkdir(parents=True, exist_ok=True)
                stem = self._safe_filename(self._youtube_video_id or self._youtube_title or source_path.stem)
                output_path = exports_dir / f"{stem}_transcript.md"
                markdown = segments_to_markdown(
                    segments,
                    title=self._youtube_title or source_path.stem,
                    source_url=self._youtube_url,
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown)
            else:
                output_path = source_path.parent / f"{source_path.stem}_transcript.txt"

                lines = []
                for seg in segments:
                    start = seg['start']
                    end = seg['end']
                    speaker = seg['speaker']
                    text = seg['text']
                    lines.append(f"[{start:6.1f}s - {end:6.1f}s] {speaker}: {text}")

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
            
            self.status_bar.showMessage(
                f"Transcript saved: {output_path.name}",
                5000  # Show for 5 seconds
            )

        except Exception as e:
            QMessageBox.warning(
                self,
                "Save Failed",
                f"Could not save transcript:\n{str(e)}"
            )

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
        return cleaned.strip("_")[:100] or "transcript"

    def _cleanup_youtube_temp(self):
        if self._source_mode != "youtube":
            return
        if self._download_temp_dir and os.path.isdir(self._download_temp_dir):
            shutil.rmtree(self._download_temp_dir, ignore_errors=True)
        self._download_temp_dir = None
        self._current_audio_file = None
    
    def closeEvent(self, event):
        """Handle window close - cancel worker if running."""
        if self._worker and self._is_processing:
            reply = QMessageBox.question(
                self,
                "Transcription in Progress",
                "A transcription is currently running. Do you want to quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self._worker.cancel()
                self._worker.quit()
                self._worker.wait()
            else:
                event.ignore()
                return

        event.accept()

