#!/usr/bin/env python3
"""
Simplified speaker-attributed transcription using pyannote + Whisper.
Based on working test scripts.
"""

import os
import sys
import time
import logging
import importlib.util
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

# Load .env and disable telemetry FIRST
from dotenv import load_dotenv
load_dotenv()

# Keep Hugging Face cache under the repo to avoid home-dir permission issues.
_repo_root = Path(__file__).resolve().parent
_cache_root = _repo_root / ".cache"
_hf_home = _cache_root / "huggingface"
_hf_hub_cache = _hf_home / "hub"
_whisper_cache = _cache_root / "whisper"
_cache_root.mkdir(parents=True, exist_ok=True)
_hf_hub_cache.mkdir(parents=True, exist_ok=True)
_whisper_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
os.environ.setdefault("HF_HOME", str(_hf_home))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_hf_hub_cache))

os.environ['PYANNOTE_METRICS_ENABLED'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable MPS fallback for NeMo

# Suppress NeMo logging (INFO and below) - set before importing nemo
os.environ['NEMO_LOGGING_LEVEL'] = 'WARNING'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pyannote.audio.core.io')
warnings.filterwarnings('ignore', message='.*torchcodec.*')
warnings.filterwarnings('ignore', message='.*Megatron.*')
warnings.filterwarnings('ignore', message='.*OneLogger.*')
warnings.filterwarnings('ignore', message='.*No exporters.*')
warnings.filterwarnings('ignore', message='.*Redirects are currently not supported.*')

import torch

# PyTorch 2.6+ defaults torch.load(..., weights_only=True). pyannote / Lightning
# checkpoints from Hugging Face still expect legacy unpickling (trusted source).
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

torch.set_default_dtype(torch.float32)  # MPS benötigt float32, unterstützt kein float64
import numpy as np
import subprocess
import tempfile
import soundfile as sf
from dataclasses import fields, is_dataclass
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment

DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
FALLBACK_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "").strip() or "turbo"
DEFAULT_WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "auto").strip().lower()
SUPPORTED_MANUAL_LANGUAGES = {
    "en": "English",
    "de": "German",
}
SAMPLE_RATE = 16000
CHUNK_DURATION = 240.0  # 4 minutes per chunk
CHUNK_OVERLAP = 3.0     # 3 seconds overlap for context preservation
_WHISPER_MODEL_CACHE: Dict[tuple, Any] = {}
_FASTER_WHISPER_MODEL_CACHE: Dict[tuple, Any] = {}


def load_diarization_pipeline(token: str, model_id: str = DEFAULT_DIARIZATION_MODEL_ID):
    """
    Load a compatible pyannote diarization pipeline across API/model variants.
    """
    model_candidates = [model_id, FALLBACK_DIARIZATION_MODEL_ID]
    last_error = None

    for candidate in model_candidates:
        try:
            pipeline = Pipeline.from_pretrained(candidate, token=token)
            if pipeline is None:
                raise RuntimeError(
                    f"Unable to access '{candidate}'. Ensure Hugging Face token is "
                    "valid and model terms are accepted on huggingface.co."
                )
            return pipeline
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument 'token'" in message:
                try:
                    pipeline = Pipeline.from_pretrained(candidate, use_auth_token=token)
                    if pipeline is None:
                        raise RuntimeError(
                            f"Unable to access '{candidate}'. Ensure Hugging Face token "
                            "is valid and model terms are accepted on huggingface.co."
                        )
                    return pipeline
                except TypeError as inner_exc:
                    if (
                        "unexpected keyword argument 'plda'" in str(inner_exc)
                        and candidate != model_candidates[-1]
                    ):
                        logger.warning(
                            f"Model {candidate} incompatible with current "
                            "pyannote.audio version, trying alternate model."
                        )
                        last_error = inner_exc
                        continue
                    raise

            if (
                "unexpected keyword argument 'plda'" in message
                and candidate != model_candidates[-1]
            ):
                logger.warning(
                    f"Model {candidate} incompatible with current "
                    "pyannote.audio version, trying alternate model."
                )
                last_error = exc
                continue
            raise
        except Exception as exc:
            last_error = exc
            if candidate != model_candidates[-1]:
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Failed to load any compatible pyannote pipeline model.")


class CustomProgressHook:
    """
    Custom progress hook that forwards pyannote pipeline step updates to GUI.
    
    PURPOSE: Capture internal pipeline step progress (segmentation, embeddings, etc.)
    and forward to progress_callback for GUI display with actual progress bar updates.
    
    CONTEXT: pyannote's ProgressHook uses rich library for console output.
    This hook intercepts those updates and forwards them to our callback with
    continuous progress updates as items are processed.
    """
    
    # Step order and approximate weight for progress calculation
    STEP_WEIGHTS = {
        'segmentation': 0.1,      # ~10% of diarization phase
        'speaker_counting': 0.05,  # ~5% of diarization phase
        'embeddings': 0.7,        # ~70% of diarization phase (usually the longest)
        'discrete_diarization': 0.15,  # ~15% of diarization phase
    }
    
    def __init__(self, progress_callback: Optional[Callable[[str, float], None]] = None):
        """
        Initialize custom progress hook.
        
        Args:
            progress_callback: Callback function to receive progress updates
        """
        self.progress_callback = progress_callback
        self.current_step = None
        self.step_progress = {}  # Track progress for each step
        self.diarization_base_progress = 0.25  # Diarization starts at 25% overall progress
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        return False
    
    def __call__(
        self,
        step_name: str,
        step_artifact: Any = None,
        file: Optional[Any] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        """
        Called by pyannote pipeline for each step update.
        This is called multiple times during processing with increasing completed values.
        
        Args:
            step_name: Name of the current step (e.g., 'segmentation', 'embeddings')
            step_artifact: Artifact from the step (not used)
            file: File being processed (not used)
            total: Total items to process
            completed: Items completed (increases during processing)
        """
        if not self.progress_callback:
            return
        
        # Initialize step tracking if this is a new step
        if step_name not in self.step_progress:
            self.step_progress[step_name] = {
                'total': total or 1,
                'completed': 0,
                'last_update': 0
            }
        
        # Update completed count - always update, even if same step
        if completed is not None:
            self.step_progress[step_name]['completed'] = completed
        if total is not None:
            self.step_progress[step_name]['total'] = total
        
        # Format step name
        formatted_name = step_name.replace('_', ' ').title()
        
        # Calculate step progress (0.0 to 1.0)
        step_data = self.step_progress[step_name]
        if step_data['total'] > 0:
            step_progress = step_data['completed'] / step_data['total']
        else:
            step_progress = 0.5  # Unknown progress, assume halfway
        
        # Calculate overall progress within diarization phase (0.25 to 0.3)
        # We map each step's progress to its portion of the diarization phase
        diarization_progress = self._calculate_diarization_progress(step_name, step_progress)
        
        # Create message with progress info
        if step_data['total'] > 1:
            percent = int(step_progress * 100)
            message = f"Diarization: {formatted_name} ({step_data['completed']}/{step_data['total']} - {percent}%)"
        else:
            message = f"Diarization: {formatted_name}"
        
        # Throttle GUI updates to avoid too frequent message changes
        # ProgressHook handles console progress bars continuously (they update in-place)
        # We only update GUI message when progress changes significantly
        if step_name not in self.step_progress:
            self.step_progress[step_name]['last_gui_progress'] = -1
        
        last_gui_progress = self.step_progress[step_name].get('last_gui_progress', -1)
        
        # Update GUI only if:
        # - Progress changed by at least 1% OR
        # - Item count changed significantly (every 10-50 items depending on total) OR
        # - Step just started (first update) OR
        # - Step completed (100%)
        progress_changed = abs(step_progress - last_gui_progress) >= 0.01
        item_changed = False
        if step_data['total'] > 100:
            item_changed = (step_data['completed'] - self.step_progress[step_name].get('last_gui_completed', 0)) >= 50
        elif step_data['total'] > 10:
            item_changed = (step_data['completed'] - self.step_progress[step_name].get('last_gui_completed', 0)) >= 5
        else:
            item_changed = True  # Small totals, update on every change
        
        is_first_update = last_gui_progress < 0
        is_complete = step_data['completed'] >= step_data['total']
        
        should_update_gui = progress_changed or item_changed or is_first_update or is_complete
        
        if should_update_gui:
            # Store last GUI update values
            self.step_progress[step_name]['last_gui_progress'] = step_progress
            self.step_progress[step_name]['last_gui_completed'] = step_data['completed']
            # Update GUI (ProgressHook already handles console bars continuously)
            self.progress_callback(message, diarization_progress)
    
    def _calculate_diarization_progress(self, current_step: str, step_progress: float) -> float:
        """
        Calculate overall progress within diarization phase based on current step.
        
        Args:
            current_step: Name of current step
            step_progress: Progress within current step (0.0 to 1.0)
        
        Returns:
            Overall progress value (0.25 to 0.3 for diarization phase)
        """
        # Get step order
        step_names = list(self.STEP_WEIGHTS.keys())
        if current_step not in step_names:
            # Unknown step, use default
            return self.diarization_base_progress + (0.05 * step_progress)
        
        # Calculate cumulative progress up to current step
        current_step_idx = step_names.index(current_step)
        cumulative_weight = sum(self.STEP_WEIGHTS[s] for s in step_names[:current_step_idx])
        current_step_weight = self.STEP_WEIGHTS[current_step]
        
        # Progress within current step
        progress_in_step = step_progress * current_step_weight
        
        # Total progress within diarization phase (0.0 to 0.05 range)
        total_diarization_progress = cumulative_weight + progress_in_step
        
        # Map to overall progress range (0.25 to 0.3)
        return self.diarization_base_progress + (0.05 * total_diarization_progress)
import nemo.collections.asr as nemo_asr
import nemo.collections.common.data.utils as nemo_data_utils

# Suppress NeMo INFO and DEBUG logging
# Set NeMo loggers to WARNING level
nemo_logger = logging.getLogger('nemo')
nemo_logger.setLevel(logging.WARNING)
# Also suppress specific NeMo loggers
for logger_name in ['nemo', 'nemo_logging', 'nemo.collections', 'nemo.collections.asr', 'nemo.utils.logging']:
    nemo_sub_logger = logging.getLogger(logger_name)
    nemo_sub_logger.setLevel(logging.WARNING)

# Suppress PyTorch distributed warnings
torch_dist_logger = logging.getLogger('torch.distributed.elastic.multiprocessing.redirects')
torch_dist_logger.setLevel(logging.ERROR)

# Setup logging - use separate logger to avoid conflicts
# Set to WARNING level to suppress debug/info messages
logger = logging.getLogger('transcribe_simple')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [BACKEND] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # Only show warnings and errors

# === MPS FIX: Patch move_data_to_device to convert float64 -> float32 ===
_original_move_data_to_device = nemo_data_utils.move_data_to_device

def _mps_safe_move_data_to_device(inputs, device, non_blocking=True):
    """MPS-safe version: converts float64 tensors to float32 before moving to MPS."""
    if inputs is None:
        return None
    if isinstance(inputs, torch.Tensor):
        # MPS does not support float64 - convert to float32
        if 'mps' in str(device) and inputs.dtype == torch.float64:
            inputs = inputs.to(torch.float32)
        return inputs.to(device, non_blocking=non_blocking)
    elif isinstance(inputs, (list, tuple, set)):
        return inputs.__class__([_mps_safe_move_data_to_device(i, device, non_blocking) for i in inputs])
    elif isinstance(inputs, dict):
        return {k: _mps_safe_move_data_to_device(v, device, non_blocking) for k, v in inputs.items()}
    elif is_dataclass(inputs):
        return type(inputs)(
            **{
                field.name: _mps_safe_move_data_to_device(getattr(inputs, field.name), device, non_blocking)
                for field in fields(inputs)
            }
        )
    else:
        return inputs

# Apply the patch
nemo_data_utils.move_data_to_device = _mps_safe_move_data_to_device
# === END MPS FIX ===

def get_device(for_whisper: bool = False) -> str:
    """
    Detect the best available device for PyTorch.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU

    The for_whisper argument is retained for compatibility with older callers.
    Whisper now uses the same accelerator preference as the rest of the app.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def normalize_transcription_language(language: Optional[str]) -> Optional[str]:
    """
    Normalize UI/API language selections.

    Returns None for automatic Parakeet mode, or an ISO code for Whisper.
    """
    if language is None:
        return None

    value = str(language).strip().lower()
    if value in ("", "auto", "automatic", "parakeet"):
        return None
    if value in ("en", "eng", "english"):
        return "en"
    if value in ("de", "deu", "ger", "german", "deutsch"):
        return "de"

    supported = ", ".join(["automatic", *SUPPORTED_MANUAL_LANGUAGES.values()])
    raise ValueError(f"Unsupported transcription language '{language}'. Use one of: {supported}.")


def describe_transcription_language(language: Optional[str]) -> str:
    code = normalize_transcription_language(language)
    if code is None:
        return "Automatic (Parakeet automatic model)"
    return SUPPORTED_MANUAL_LANGUAGES[code]


def _iter_audio_chunks(waveform: torch.Tensor):
    total_samples = waveform.shape[1]
    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    overlap_samples = int(CHUNK_OVERLAP * SAMPLE_RATE)
    step_samples = chunk_samples - overlap_samples
    num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / step_samples)))

    for chunk_idx in range(num_chunks):
        chunk_start_sample = chunk_idx * step_samples
        chunk_end_sample = min(chunk_start_sample + chunk_samples, total_samples)
        chunk_start_time = chunk_start_sample / SAMPLE_RATE
        chunk_waveform = waveform[:, chunk_start_sample:chunk_end_sample]
        chunk_duration = chunk_waveform.shape[1] / SAMPLE_RATE
        yield chunk_idx, num_chunks, chunk_start_time, chunk_duration, chunk_waveform


def _write_temp_wav(chunk_waveform: torch.Tensor) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_audio_path = tmp_file.name

    audio_int16 = (chunk_waveform.squeeze().numpy() * 32768).astype(np.int16)
    sf.write(tmp_audio_path, audio_int16, SAMPLE_RATE)
    return tmp_audio_path


def _segment_speaker(annotation, start: float, end: float) -> str:
    safe_end = end if end > start else start + 0.01
    speaker = annotation.argmax(Segment(start, safe_end))
    return speaker if speaker is not None else "UNKNOWN"


def _append_segment(
    all_segments: List[Dict[str, Any]],
    segment_callback: Optional[Callable[[Dict[str, Any]], None]],
    annotation,
    start: float,
    end: float,
    text: str,
) -> None:
    clean_text = text.strip()
    if not clean_text:
        return

    segment_dict = {
        "start": start,
        "end": end,
        "speaker": _segment_speaker(annotation, start, end),
        "text": clean_text,
    }
    all_segments.append(segment_dict)

    if segment_callback:
        segment_callback(segment_dict)


class _SpeakerTurnMerger:
    """
    Accumulate short ASR segments into continuous speaker turns before emitting.
    """

    def __init__(
        self,
        all_segments: List[Dict[str, Any]],
        segment_callback: Optional[Callable[[Dict[str, Any]], None]],
        annotation,
    ):
        self.all_segments = all_segments
        self.segment_callback = segment_callback
        self.annotation = annotation
        self.current: Optional[Dict[str, Any]] = None

    def add(self, start: float, end: float, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return

        speaker = _segment_speaker(self.annotation, start, end)
        if self.current and self.current["speaker"] == speaker:
            self.current["end"] = end
            self.current["text"] = f"{self.current['text']} {clean_text}".strip()
            return

        self.flush()
        self.current = {
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": clean_text,
        }

    def flush(self) -> None:
        if not self.current:
            return

        segment = self.current
        self.current = None
        self.all_segments.append(segment)

        if self.segment_callback:
            self.segment_callback(segment)


def _load_whisper_model(device_name: str):
    model_name = os.getenv("WHISPER_MODEL", "").strip() or DEFAULT_WHISPER_MODEL_NAME
    cache_key = (model_name, device_name)
    if cache_key not in _WHISPER_MODEL_CACHE:
        import whisper

        logger.info(f"Loading Whisper model {model_name} on {device_name}")
        _WHISPER_MODEL_CACHE[cache_key] = whisper.load_model(
            model_name,
            device=device_name,
            download_root=str(_whisper_cache),
        )
    return _WHISPER_MODEL_CACHE[cache_key], model_name


def _select_whisper_backend(device_name: str) -> str:
    backend = DEFAULT_WHISPER_BACKEND
    if backend not in ("auto", "faster-whisper", "openai-whisper"):
        logger.warning("Unknown WHISPER_BACKEND=%s; using auto", backend)
        backend = "auto"

    if backend == "openai-whisper":
        return "openai-whisper"

    faster_available = importlib.util.find_spec("faster_whisper") is not None
    if backend == "faster-whisper" and not faster_available:
        raise RuntimeError(
            "WHISPER_BACKEND=faster-whisper was requested, but faster-whisper is not installed."
        )

    # faster-whisper is optimized for CUDA and CPU via CTranslate2. MPS is not a
    # supported faster-whisper device, so Apple GPU runs use OpenAI Whisper.
    if faster_available and device_name in ("cuda", "cpu"):
        return "faster-whisper"

    return "openai-whisper"


def _faster_whisper_compute_type(device_name: str) -> str:
    override = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "").strip()
    if override:
        return override
    if device_name == "cuda":
        return "float16"
    return "int8"


def _load_faster_whisper_model(device_name: str):
    model_name = os.getenv("WHISPER_MODEL", "").strip() or DEFAULT_WHISPER_MODEL_NAME
    compute_type = _faster_whisper_compute_type(device_name)
    cache_key = (model_name, device_name, compute_type)
    if cache_key not in _FASTER_WHISPER_MODEL_CACHE:
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading faster-whisper model {model_name} on {device_name} ({compute_type})"
        )
        _FASTER_WHISPER_MODEL_CACHE[cache_key] = WhisperModel(
            model_name,
            device=device_name,
            compute_type=compute_type,
            download_root=str(_whisper_cache),
        )
    return _FASTER_WHISPER_MODEL_CACHE[cache_key], model_name, compute_type


def _transcribe_chunk_with_faster_whisper(model, tmp_audio_path: str, language: str):
    segments, _info = model.transcribe(
        tmp_audio_path,
        language=language,
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    return [
        {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text,
        }
        for seg in segments
    ]


def _transcribe_chunk_with_openai_whisper(
    model,
    tmp_audio_path: str,
    language: str,
    fp16: bool,
):
    result = model.transcribe(
        tmp_audio_path,
        language=language,
        task="transcribe",
        fp16=fp16,
        verbose=False,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    return result.get("segments", [])


def _transcribe_with_parakeet(
    waveform: torch.Tensor,
    annotation,
    all_segments: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[str, float], None]],
    segment_callback: Optional[Callable[[Dict[str, Any]], None]],
    check_interrupt: Optional[Callable[[], bool]],
) -> None:
    if progress_callback:
        progress_callback("Loading Parakeet transcription model...", 0.35)

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL_ID)

    device_name = get_device()
    if device_name in ("mps", "cuda"):
        asr_model = asr_model.to(torch.device(device_name))

    chunks = list(_iter_audio_chunks(waveform))
    num_chunks = len(chunks)

    if progress_callback:
        progress_callback(f"Processing {num_chunks} chunk(s) with Parakeet...", 0.4)

    if check_interrupt and check_interrupt():
        return

    for chunk_idx, _, chunk_start_time, chunk_duration, chunk_waveform in chunks:
        if check_interrupt and check_interrupt():
            break

        tmp_audio_path = _write_temp_wav(chunk_waveform)
        try:
            hypotheses = asr_model.transcribe([tmp_audio_path], timestamps=True)
            chunk_segments = hypotheses[0].timestamp["segment"]

            for seg in chunk_segments:
                seg_start_in_chunk = float(seg["start"])
                seg_end_in_chunk = float(seg["end"])

                if chunk_idx > 0 and seg_start_in_chunk < (CHUNK_OVERLAP / 2):
                    continue

                global_start = chunk_start_time + seg_start_in_chunk
                global_end = chunk_start_time + seg_end_in_chunk
                _append_segment(
                    all_segments,
                    segment_callback,
                    annotation,
                    global_start,
                    global_end,
                    seg["segment"],
                )

            if progress_callback:
                progress = 0.4 + 0.6 * (chunk_idx + 1) / num_chunks
                progress_callback(
                    f"Processing Parakeet chunk {chunk_idx + 1}/{num_chunks} ({chunk_duration:.1f}s)",
                    progress,
                )
        except Exception as e:
            if progress_callback:
                progress_callback(
                    f"Parakeet chunk {chunk_idx + 1}/{num_chunks} failed: {e}",
                    0.4 + 0.6 * (chunk_idx + 1) / num_chunks,
                )
        finally:
            os.unlink(tmp_audio_path)


def _transcribe_with_whisper(
    waveform: torch.Tensor,
    annotation,
    language: str,
    all_segments: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[str, float], None]],
    segment_callback: Optional[Callable[[Dict[str, Any]], None]],
    check_interrupt: Optional[Callable[[], bool]],
) -> None:
    language_name = SUPPORTED_MANUAL_LANGUAGES[language]
    device_name = get_device(for_whisper=True)
    backend = _select_whisper_backend(device_name)

    if progress_callback:
        progress_callback(
            f"Loading {backend} model for {language_name} on {device_name.upper()}...",
            0.35,
        )

    compute_type = None
    if backend == "faster-whisper":
        try:
            whisper_model, model_name, compute_type = _load_faster_whisper_model(device_name)
        except Exception as exc:
            if DEFAULT_WHISPER_BACKEND == "faster-whisper":
                raise
            logger.warning(
                "faster-whisper unavailable for this run (%s); falling back to openai-whisper",
                exc,
            )
            backend = "openai-whisper"
            whisper_model, model_name = _load_whisper_model(device_name)
    else:
        whisper_model, model_name = _load_whisper_model(device_name)

    fp16 = device_name == "cuda"
    chunks = list(_iter_audio_chunks(waveform))
    num_chunks = len(chunks)

    if progress_callback:
        backend_detail = backend
        if compute_type:
            backend_detail = f"{backend} {compute_type}"
        progress_callback(
            f"Processing {num_chunks} chunk(s) with {backend_detail} {model_name}...",
            0.4,
        )

    if check_interrupt and check_interrupt():
        return

    turn_merger = _SpeakerTurnMerger(all_segments, segment_callback, annotation)

    for chunk_idx, _, chunk_start_time, chunk_duration, chunk_waveform in chunks:
        if check_interrupt and check_interrupt():
            break

        tmp_audio_path = _write_temp_wav(chunk_waveform)
        try:
            if backend == "faster-whisper":
                result_segments = _transcribe_chunk_with_faster_whisper(
                    whisper_model,
                    tmp_audio_path,
                    language,
                )
            else:
                result_segments = _transcribe_chunk_with_openai_whisper(
                    whisper_model,
                    tmp_audio_path,
                    language,
                    fp16,
                )

            for seg in result_segments:
                seg_start_in_chunk = float(seg.get("start", 0.0))
                seg_end_in_chunk = float(seg.get("end", seg_start_in_chunk))

                if chunk_idx > 0 and seg_start_in_chunk < (CHUNK_OVERLAP / 2):
                    continue

                global_start = chunk_start_time + seg_start_in_chunk
                global_end = chunk_start_time + seg_end_in_chunk
                turn_merger.add(
                    global_start,
                    global_end,
                    str(seg.get("text", "")),
                )

            if progress_callback:
                progress = 0.4 + 0.6 * (chunk_idx + 1) / num_chunks
                progress_callback(
                    f"Processing {backend} chunk {chunk_idx + 1}/{num_chunks} ({chunk_duration:.1f}s)",
                    progress,
                )
        except Exception as e:
            if progress_callback:
                progress_callback(
                    f"{backend} chunk {chunk_idx + 1}/{num_chunks} failed: {e}",
                    0.4 + 0.6 * (chunk_idx + 1) / num_chunks,
                )
        finally:
            os.unlink(tmp_audio_path)

    turn_merger.flush()


def transcribe_audio(
    audio_file: str,
    num_speakers: Optional[int] = None,
    transcription_language: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    segment_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    check_interrupt: Optional[Callable[[], bool]] = None,
    pipeline: Optional[Pipeline] = None,
    test_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Transcribe audio with speaker diarization.
    
    PURPOSE: Main transcription function that processes audio files with speaker
    diarization and ASR. Supports streaming output via callbacks for GUI integration.
    
    CONTEXT: Designed for both CLI and GUI usage. Callbacks enable live updates
    during processing, especially useful for long audio files.
    
    Args:
        audio_file: Path to audio file (supports various formats via ffmpeg)
        num_speakers: Number of speakers (None = auto-detect)
        transcription_language: None/automatic uses Parakeet. "en" and "de"
                                force Whisper transcription in that language.
        progress_callback: Callback(status_message, progress_0_to_1) for progress updates
        segment_callback: Callback(segment_dict) - called immediately after each chunk
                          for streaming output. Segment dict: {'start': float, 'end': float,
                          'speaker': str, 'text': str}
        check_interrupt: Optional callable that returns True if processing should stop
        pipeline: Optional pre-loaded Pipeline object. If None, will load pipeline.
                  Use this to avoid loading pipeline in worker thread (threading issues).
        test_mode: If True, process only the first 60 seconds of audio (quick tests).
    
    Returns:
        List of all transcription segments with timestamps and speakers.
        Format: [{'start': float, 'end': float, 'speaker': str, 'text': str}, ...]
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        subprocess.CalledProcessError: If ffmpeg conversion fails
        RuntimeError: If HuggingFace token is missing or processing fails
    """
    # Validate HuggingFace token
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN not found in .env file")

    transcription_language = normalize_transcription_language(transcription_language)
    
    all_segments = []
    function_start = time.time()
    logger.debug(f"transcribe_audio() called at {function_start:.2f}")
    
    # === STEP 1: LOAD DIARIZATION PIPELINE ===
    # If pipeline is provided, use it (loaded in main thread to avoid threading issues)
    # Otherwise, load it here (may be slow in worker thread)
    if pipeline is None:
        pipeline_start = time.time()
        logger.debug(f"Starting pipeline loading at {pipeline_start:.2f}")
        
        if progress_callback:
            progress_callback("Loading diarization pipeline...", 0.05)
        
        # Load pipeline - this is a blocking call that may take time on first run
        # (when models need to be downloaded). On subsequent runs with cached models,
        # this should be fast. GUI stays responsive because this runs in worker thread.
        try:
            logger.debug("Calling Pipeline.from_pretrained()...")
            logger.debug("This may take a while - checking model cache...")
            pipeline_load_start = time.time()
            
            pipeline = load_diarization_pipeline(token)
            
            pipeline_load_elapsed = time.time() - pipeline_load_start
            logger.debug(f"Pipeline.from_pretrained() completed in {pipeline_load_elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Pipeline loading failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Pipeline loading failed: {str(e)}", 0.0)
            raise RuntimeError(f"Failed to load diarization pipeline: {str(e)}")
        
        pipeline_elapsed = time.time() - pipeline_start
        logger.debug(f"Pipeline loading phase completed in {pipeline_elapsed:.2f}s")
        
        if progress_callback:
            progress_callback("Pipeline loaded, moving to device...", 0.08)
        
        # Move pipeline to best available device
        device_start = time.time()
        device_name = get_device()
        logger.debug(f"Moving pipeline to device: {device_name}")
        device = torch.device(device_name)
        pipeline.to(device)
        device_elapsed = time.time() - device_start
        logger.debug(f"Pipeline moved to device in {device_elapsed:.2f}s")
        
        if progress_callback:
            progress_callback(f"Pipeline ready on {device_name.upper()}", 0.1)
    else:
        # Pipeline already loaded - assume it's already on the correct device
        # (was moved to device when loaded in main thread)
        logger.debug("Using pre-loaded pipeline")
        if progress_callback:
            progress_callback("Using pre-loaded pipeline...", 0.05)
        
        device_name = get_device()
        if progress_callback:
            progress_callback(f"Pipeline ready on {device_name.upper()}", 0.1)
    
    # Check for interrupt
    if check_interrupt and check_interrupt():
        return []
    
    # === STEP 2: LOAD AUDIO ===
    if progress_callback:
        progress_callback("Loading audio...", 0.15)
    
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    cmd = ['ffmpeg', '-i', audio_file,
           '-f', 's16le', '-acodec', 'pcm_s16le',
           '-ar', '16000', '-ac', '1', '-']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    audio_data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio_data).unsqueeze(0)

    if test_mode:
        max_samples = 60 * 16000
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        if progress_callback:
            progress_callback("Test mode: using first 60 seconds only", 0.19)

    total_duration = waveform.shape[1] / 16000

    if progress_callback:
        progress_callback(f"Audio loaded: {total_duration/60:.1f} minutes", 0.2)
    
    # Check for interrupt
    if check_interrupt and check_interrupt():
        return []
    
    # === STEP 3: RUN DIARIZATION ===
    diarization_start = time.time()
    logger.debug(f"Starting diarization at {diarization_start:.2f}")
    
    if progress_callback:
        progress_callback("Running speaker diarization...", 0.25)
    
    audio_input = {"waveform": waveform, "sample_rate": 16000}
    
    diar_kwargs = {}
    if num_speakers:
        diar_kwargs['num_speakers'] = num_speakers
        logger.debug(f"Diarization with num_speakers={num_speakers}")
    else:
        logger.debug("Diarization with auto-detect speakers")
    
    logger.debug("Calling pipeline() for diarization...")
    pipeline_call_start = time.time()
    
    # Use original ProgressHook for console output (shows beautiful progress bars)
    # No GUI updates during diarization - console bars are sufficient
    with ProgressHook() as hook:
        diarization = pipeline(audio_input, hook=hook, **diar_kwargs)
    
    pipeline_call_elapsed = time.time() - pipeline_call_start
    logger.debug(f"pipeline() call completed in {pipeline_call_elapsed:.2f}s")
    
    # Extract annotation
    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization
    
    speakers_found = len(annotation.labels())
    diarization_elapsed = time.time() - diarization_start
    logger.debug(f"Diarization completed in {diarization_elapsed:.2f}s, found {speakers_found} speaker(s)")
    
    if progress_callback:
        progress_callback(f"Diarization complete - found {speakers_found} speaker(s)", 0.3)
    
    # Check for interrupt
    if check_interrupt and check_interrupt():
        return []
    
    # === STEP 4: TRANSCRIBE WITH PARAKEET OR FORCED-LANGUAGE WHISPER ===
    if transcription_language is None:
        _transcribe_with_parakeet(
            waveform=waveform,
            annotation=annotation,
            all_segments=all_segments,
            progress_callback=progress_callback,
            segment_callback=segment_callback,
            check_interrupt=check_interrupt,
        )
    else:
        _transcribe_with_whisper(
            waveform=waveform,
            annotation=annotation,
            language=transcription_language,
            all_segments=all_segments,
            progress_callback=progress_callback,
            segment_callback=segment_callback,
            check_interrupt=check_interrupt,
        )
    
    # Sort segments by start time (in case of any ordering issues)
    all_segments.sort(key=lambda x: x['start'])
    
    if progress_callback:
        progress_callback(f"Transcription complete ({len(all_segments)} segments)", 1.0)
    
    return all_segments


def merge_consecutive_same_speaker(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge contiguous segments when the speaker does not change.
    """
    if not segments:
        return []

    merged: List[Dict[str, Any]] = []
    current = {
        "start": float(segments[0].get("start", 0.0)),
        "end": float(segments[0].get("end", 0.0)),
        "speaker": str(segments[0].get("speaker", "UNKNOWN")),
        "text": str(segments[0].get("text", "")).strip(),
    }

    for seg in segments[1:]:
        speaker = str(seg.get("speaker", "UNKNOWN"))
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()

        if speaker == current["speaker"]:
            current["end"] = end
            if text:
                current["text"] = f"{current['text']} {text}".strip()
        else:
            merged.append(current)
            current = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }

    merged.append(current)
    return merged


def main():
    """
    CLI entry point for speaker-attributed transcription.
    Maintains backward compatibility with existing command-line usage.
    """
    # Get configuration from environment
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "Music Company Media Productions 10.m4a"
    test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
    num_speakers = int(os.getenv('NUM_SPEAKERS', '0')) or None
    transcription_language = os.getenv("TRANSCRIPTION_LANGUAGE", "automatic")
    try:
        transcription_language_label = describe_transcription_language(transcription_language)
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("="*60)
    print("Speaker-Attributed Transcription")
    print("="*60)
    print(f"Audio file: {audio_file}")
    print(f"Test mode: {test_mode}")
    print(f"Expected speakers: {num_speakers if num_speakers else 'auto-detect'}")
    print(f"Transcription language: {transcription_language_label}")
    print("="*60)

    # Progress callback for CLI output
    def progress_callback(message: str, progress: float):
        """Print progress updates to console."""
        print(f"\n[{progress*100:.0f}%] {message}")

    # Segment callback for CLI output (streaming)
    def segment_callback(segment: Dict[str, Any]):
        """Print segments immediately as they're processed."""
        seg_start = segment['start']
        seg_end = segment['end']
        speaker = segment['speaker']
        text = segment['text']
        print(f"[{seg_start:6.1f}s - {seg_end:6.1f}s] {speaker}: {text}")

    try:
        # Call refactored function
        segments = transcribe_audio(
            audio_file=audio_file,
            num_speakers=num_speakers,
            transcription_language=transcription_language,
            progress_callback=progress_callback,
            segment_callback=segment_callback,
            test_mode=test_mode,
        )
        merged_segments = merge_consecutive_same_speaker(segments)
        
        print("\n" + "="*60)
        print(f"✓ Complete! ({len(segments)} segments total -> {len(merged_segments)} merged blocks)")
        print("="*60)
        
        # Auto-save transcript to file (same as GUI behavior)
        if merged_segments:
            try:
                from pathlib import Path
                source_path = Path(audio_file)
                output_path = source_path.parent / f"{source_path.stem}_transcript.txt"
                
                # Format transcript
                lines = []
                for seg in merged_segments:
                    start = seg['start']
                    end = seg['end']
                    speaker = seg['speaker']
                    text = seg['text']
                    lines.append(f"[{start:6.1f}s - {end:6.1f}s] {speaker}: {text}")
                
                # Write file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                print(f"\n✓ Transcript saved to: {output_path}")
            
            except Exception as e:
                print(f"\n⚠ Warning: Could not save transcript: {e}", file=sys.stderr)
    
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
