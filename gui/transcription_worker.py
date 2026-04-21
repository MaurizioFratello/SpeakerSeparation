"""
QThread worker for asynchronous transcription processing.

PURPOSE: Execute transcription in background thread to prevent GUI freezing.
Emits signals for progress updates, streaming segments, completion, and errors.

CONTEXT: PySide6 QThread pattern for long-running operations. Worker runs
transcribe_audio() from backend and communicates with GUI via Qt signals.
"""

import sys
import os
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add parent directory to path for backend import
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

from PySide6.QtCore import QThread, Signal
from transcribe_simple import transcribe_audio
from gui.audio_converter import convert_to_wav, cleanup_temp_file, is_supported_format

# Setup logging - use separate logger to avoid conflicts
# Set to WARNING level to suppress debug/info messages
logger = logging.getLogger('transcription_worker')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [WORKER] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # Only show warnings and errors


class TranscriptionWorker(QThread):
    """
    Worker thread for asynchronous transcription processing.
    
    PURPOSE: Run transcription in background without blocking GUI.
    Supports streaming output, progress updates, and cancellation.
    
    CONTEXT: Qt threading pattern - worker runs in separate thread,
    emits signals that are connected to GUI slots for updates.
    """
    
    # Signals for GUI communication
    progress = Signal(str, float)  # (message, progress_0_to_1)
    segment_ready = Signal(dict)   # Streaming: segment dict with start, end, speaker, text
    finished = Signal(list)         # All segments complete: list of all segment dicts
    error = Signal(str)             # Error message
    
    def __init__(
        self,
        audio_file: str,
        num_speakers: Optional[int] = None,
        transcription_language: str = "auto",
        strict_language_mode: bool = True,
        pipeline=None,
        parent=None
    ):
        """
        Initialize transcription worker.
        
        Args:
            audio_file: Path to audio file to transcribe
            num_speakers: Number of speakers (None = auto-detect)
            transcription_language: Language hint for ASR ("auto", "de", "en")
            strict_language_mode: Force selected language using Whisper (True/False)
            pipeline: Optional pre-loaded Pipeline object (loaded in main thread)
            parent: Parent QObject for Qt parent-child relationship
        """
        super().__init__(parent)
        self.audio_file = audio_file
        self.num_speakers = num_speakers
        self.transcription_language = transcription_language
        self.strict_language_mode = strict_language_mode
        self._pipeline = pipeline  # Pre-loaded pipeline to avoid threading issues
        self._temp_wav_path: Optional[str] = None
        self._is_cancelled = False
    
    def run(self):
        """
        Execute transcription in background thread.
        
        PURPOSE: Main worker method that runs in separate thread.
        Handles audio conversion, calls backend, emits signals for updates.
        
        CONTEXT: Called automatically by QThread.start(). Must not be called directly.
        """
        start_time = time.time()
        logger.debug(f"Worker thread started at {start_time:.2f}")
        
        try:
            # Check if format conversion is needed
            needs_conversion = not (self.audio_file.lower().endswith('.wav'))
            logger.debug(f"Audio file: {self.audio_file}, needs_conversion: {needs_conversion}")
            
            if needs_conversion:
                conv_start = time.time()
                logger.debug(f"Starting audio conversion at {conv_start:.2f}")
                self.progress.emit("Converting audio format...", 0.0)
                try:
                    self._temp_wav_path = convert_to_wav(self.audio_file)
                    audio_path = self._temp_wav_path
                    conv_elapsed = time.time() - conv_start
                    logger.debug(f"Audio conversion completed in {conv_elapsed:.2f}s")
                except Exception as e:
                    logger.error(f"Audio conversion failed: {e}")
                    self.error.emit(f"Audio conversion failed: {str(e)}")
                    return
            else:
                audio_path = self.audio_file
                logger.debug(f"Using audio file directly: {audio_path}")
            
            # Check for cancellation
            if self.isInterruptionRequested():
                logger.debug("Interruption requested before transcription")
                self._cleanup()
                return
            
            # Send initial progress update to ensure GUI is responsive
            logger.debug("Sending initial progress update")
            self.progress.emit("Starting transcription...", 0.01)
            
            # Progress callback wrapper
            def progress_callback(message: str, progress_val: float):
                """Forward progress updates to GUI via signal."""
                elapsed = time.time() - start_time
                logger.debug(f"[{elapsed:.2f}s] Progress: {message} ({progress_val:.2%})")
                if not self.isInterruptionRequested():
                    self.progress.emit(message, progress_val)
            
            # Segment callback wrapper for streaming
            def segment_callback(segment: Dict[str, Any]):
                """Forward segments to GUI immediately for streaming display."""
                elapsed = time.time() - start_time
                logger.debug(f"[{elapsed:.2f}s] Segment ready: {segment.get('speaker', '?')} at {segment.get('start', 0):.1f}s")
                if not self.isInterruptionRequested():
                    self.segment_ready.emit(segment)
            
            # Interrupt check function
            def check_interrupt() -> bool:
                """Check if processing should be cancelled."""
                return self.isInterruptionRequested()
            
            # Call backend transcription function with pre-loaded pipeline
            transcribe_start = time.time()
            logger.debug(f"Calling transcribe_audio() at {transcribe_start:.2f}")
            all_segments = transcribe_audio(
                audio_file=audio_path,
                num_speakers=self.num_speakers,
                transcription_language=self.transcription_language,
                strict_language_mode=self.strict_language_mode,
                progress_callback=progress_callback,
                segment_callback=segment_callback,
                check_interrupt=check_interrupt,
                pipeline=self._pipeline  # Use pre-loaded pipeline (loaded in main thread)
            )
            transcribe_elapsed = time.time() - transcribe_start
            logger.debug(f"transcribe_audio() completed in {transcribe_elapsed:.2f}s, got {len(all_segments)} segments")
            
            # Cleanup temporary file
            self._cleanup()
            
            # Emit final result (even if empty due to cancellation)
            total_elapsed = time.time() - start_time
            if self.isInterruptionRequested():
                logger.debug(f"Transcription cancelled after {total_elapsed:.2f}s")
                self.progress.emit("Transcription cancelled", 0.0)
                self.finished.emit([])
            else:
                logger.debug(f"Transcription completed successfully in {total_elapsed:.2f}s")
                self.finished.emit(all_segments)
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Exception after {elapsed:.2f}s: {e}", exc_info=True)
            self._cleanup()
            error_msg = f"Transcription failed: {str(e)}"
            self.error.emit(error_msg)
    
    def _cleanup(self):
        """Clean up temporary files created during processing."""
        if self._temp_wav_path and os.path.exists(self._temp_wav_path):
            cleanup_temp_file(self._temp_wav_path)
            self._temp_wav_path = None
    
    def cancel(self):
        """
        Request cancellation of transcription.
        
        PURPOSE: Gracefully stop processing. Worker checks interruption
        status between chunks and stops after current chunk completes.
        
        CONTEXT: Called from GUI when user clicks Cancel button.
        Uses Qt's interruption mechanism for thread-safe cancellation.
        """
        self.requestInterruption()

