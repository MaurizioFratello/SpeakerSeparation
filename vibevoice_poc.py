"""VibeVoice-ASR inference helpers for the local diarization POC."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

DEFAULT_MODEL_ID = "microsoft/VibeVoice-ASR"

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".mpeg", ".mpg"}
)
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_AUDIO_CODEC = "pcm_s16le"


def setup_cache_env() -> Path:
    """Use repo-local Hugging Face cache, matching transcribe_simple.py."""
    repo_root = Path(__file__).resolve().parent
    cache_root = repo_root / ".cache"
    hf_home = cache_root / "huggingface"
    hf_hub_cache = hf_home / "hub"
    cache_root.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub_cache))
    return repo_root


def detect_device_and_attn(
    device: str = "auto",
    attn_implementation: str = "auto",
) -> tuple[str, str]:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        device = "cpu"

    if attn_implementation == "auto":
        if device == "cuda":
            try:
                import flash_attn  # noqa: F401

                attn_implementation = "flash_attention_2"
            except ImportError:
                attn_implementation = "sdpa"
        else:
            attn_implementation = "sdpa"

    return device, attn_implementation


SKIP_SEGMENT_LABELS = frozenset(
    {
        "silence",
        "silent",
        "no speech",
        "no_speech",
        "speech",
        "music",
        "human sounds",
        "environmental sounds",
        "noise",
        "applause",
        "laughter",
        "cough",
        "breathing",
    }
)


def _normalize_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_time": seg.get("start_time", seg.get("Start")),
        "end_time": seg.get("end_time", seg.get("End")),
        "speaker_id": seg.get("speaker_id", seg.get("Speaker", "Unknown")),
        "text": (seg.get("text") or seg.get("Content") or "").strip(),
    }


def _is_skippable_segment(text: str) -> bool:
    normalized = text.strip().strip("[]").casefold()
    return not normalized or normalized in SKIP_SEGMENT_LABELS


def _seconds_to_timestamp(value: Any, *, precise: bool = False) -> str:
    if value is None:
        return "??:??"
    if isinstance(value, str) and ":" in value:
        return value
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600) - (minutes * 60)

    if precise:
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
        return f"{minutes:02d}:{secs:05.2f}"

    if hours:
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"
    return f"{minutes:02d}:{int(secs):02d}"


def _speaker_key(speaker: Any) -> str:
    if speaker in (None, "", "Unknown"):
        return "Unknown"
    return str(speaker)


def _speaker_label(speaker: Any, speaker_names: Optional[Dict[str, str]] = None) -> str:
    key = _speaker_key(speaker)
    if speaker_names:
        custom = (speaker_names.get(key) or "").strip()
        if custom:
            return custom
    if key == "Unknown":
        return "Unknown"
    return f"Speaker {key}"


def is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def convert_media_to_wav(media_path: str) -> str:
    """Convert any audio or video file to 24 kHz mono 16-bit PCM WAV."""
    import subprocess

    source = Path(media_path)
    if not source.is_file():
        raise FileNotFoundError(f"Media not found: {media_path}")

    out_dir = Path(tempfile.mkdtemp(prefix="vibevoice_media_"))
    wav_path = out_dir / f"{source.stem}.wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                str(TARGET_CHANNELS),
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-c:a",
                TARGET_AUDIO_CODEC,
                str(wav_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
        tail = "\n".join(detail[-8:]) if detail else str(exc)
        raise RuntimeError(f"Failed to convert {source.name} to WAV:\n{tail}") from exc
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to convert {media_path} to WAV")
    return str(wav_path)


def extract_audio_from_video(video_path: str) -> str:
    """Extract mono WAV audio from a video file for transcription."""
    return convert_media_to_wav(video_path)


def prepare_media_for_transcription(
    audio_path: Optional[str],
    video_path: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve upload to an audio file path suitable for VibeVoice.

    Returns (audio_path, display_stem). display_stem prefers the original media name.
    Any audio or video source is converted to 24 kHz mono WAV.
    """
    source = video_path or audio_path
    if not source:
        return None, None
    return convert_media_to_wav(source), Path(source).stem


def prepare_dropped_media(media_path: Optional[str]) -> Dict[str, Optional[str]]:
    """Convert a dropped audio/video file to the target WAV format."""
    if not media_path:
        return {
            "wav_path": None,
            "source_stem": None,
            "source_name": None,
            "status": "Drop an audio or video file.",
        }
    source = Path(media_path)
    wav_path = convert_media_to_wav(str(source))
    return {
        "wav_path": wav_path,
        "source_stem": source.stem,
        "source_name": source.name,
        "status": f"Ready: `{source.name}` → `{Path(wav_path).name}`",
    }


def _load_audio_array(audio_path: str):
    """Load audio as a mono array. Falls back to ffmpeg for mp3/m4a."""
    import numpy as np
    import soundfile as sf

    try:
        data, sample_rate = sf.read(audio_path, always_2d=False)
    except Exception:
        import subprocess

        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        data, sample_rate = sf.read(wav_path, always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)
    return data, int(sample_rate)


def _segment_duration_seconds(seg: Dict[str, Any]) -> float:
    try:
        return max(0.0, float(seg["end_time"]) - float(seg["start_time"]))
    except (TypeError, ValueError, KeyError):
        return 0.0


def choose_speaker_sample_segments(
    segments: List[Dict[str, Any]],
    *,
    min_seconds: float = 3.0,
) -> Dict[str, Dict[str, Any]]:
    """Pick one sample turn per speaker, preferring clips of at least min_seconds."""
    by_speaker: Dict[str, List[Dict[str, Any]]] = {}
    for seg in merge_consecutive_segments(segments):
        speaker_key = _speaker_key(seg["speaker_id"])
        if speaker_key == "Unknown":
            continue
        by_speaker.setdefault(speaker_key, []).append(seg)

    chosen: Dict[str, Dict[str, Any]] = {}
    for speaker_key, speaker_segs in by_speaker.items():
        long_enough = [seg for seg in speaker_segs if _segment_duration_seconds(seg) >= min_seconds]
        if long_enough:
            chosen[speaker_key] = long_enough[0]
        else:
            chosen[speaker_key] = max(speaker_segs, key=_segment_duration_seconds)
    return chosen


def extract_speaker_samples(
    audio_path: str,
    segments: List[Dict[str, Any]],
    *,
    sample_seconds: float = 8.0,
    min_segment_seconds: float = 3.0,
) -> Dict[str, str]:
    """Cut a short clip per speaker from a substantial turn when available."""
    import soundfile as sf

    chosen = choose_speaker_sample_segments(segments, min_seconds=min_segment_seconds)
    if not chosen:
        return {}

    data, sample_rate = _load_audio_array(audio_path)
    samples: Dict[str, str] = {}
    for speaker_key, seg in chosen.items():
        start = float(seg["start_time"])
        end = float(seg["end_time"])
        clip_end = min(end, start + sample_seconds)
        start_idx = max(0, int(start * sample_rate))
        end_idx = min(len(data), int(clip_end * sample_rate))
        if end_idx <= start_idx:
            continue
        clip = data[start_idx:end_idx]
        unique = f"speaker_{speaker_key}_{int(start * 1000)}.wav"
        out_dir = Path(tempfile.mkdtemp(prefix="vibevoice_samples_"))
        out_path = out_dir / unique
        sf.write(str(out_path), clip, sample_rate, subtype="PCM_16")
        samples[speaker_key] = str(out_path)
    return samples


def speaker_samples_to_html(
    samples: Dict[str, str],
    speaker_names: Optional[Dict[str, str]] = None,
) -> str:
    """Embed per-speaker clips as HTML audio players (avoids Gradio multi-file cache bugs)."""
    import base64
    from html import escape

    if not samples:
        return ""

    blocks = ['<div style="display:flex;flex-direction:column;gap:12px;">']
    for speaker_id, path in sorted(samples.items(), key=lambda item: str(item[0])):
        wav_path = Path(path)
        if not wav_path.is_file():
            continue
        payload = base64.b64encode(wav_path.read_bytes()).decode("ascii")
        label = _speaker_label(speaker_id, speaker_names)
        blocks.append(
            '<div style="border:1px solid var(--border-color-primary, #ddd);'
            'border-radius:10px;padding:12px;">'
            f'<div style="font-weight:600;margin-bottom:8px;">{escape(label)} sample</div>'
            f'<audio controls preload="metadata" src="data:audio/wav;base64,{payload}" '
            'style="width:100%;"></audio>'
            "</div>"
        )
    blocks.append("</div>")
    return "".join(blocks)


def segments_to_markdown(
    segments: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
    speaker_names: Optional[Dict[str, str]] = None,
) -> str:
    merged = merge_consecutive_segments(segments)
    heading = (title or "Transcript").strip() or "Transcript"
    lines = [f"# {heading}", ""]

    for seg in merged:
        start = _seconds_to_timestamp(seg["start_time"], precise=True)
        end = _seconds_to_timestamp(seg["end_time"], precise=True)
        label = _speaker_label(seg["speaker_id"], speaker_names)
        text = (seg.get("text") or "").strip()
        lines.append(f"**[{start} - {end}] {label}**")
        lines.append(text if text else "_(no text)_")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_markdown_file(
    segments: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
    speaker_names: Optional[Dict[str, str]] = None,
    filename: str = "transcript.md",
) -> str:
    markdown = segments_to_markdown(segments, title=title, speaker_names=speaker_names)
    out_dir = Path(tempfile.gettempdir()) / "vibevoice_poc_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(markdown, encoding="utf-8")
    return str(out_path)


def merge_consecutive_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for seg in segments:
        normalized = _normalize_segment(seg)
        if _is_skippable_segment(normalized["text"]):
            continue
        if merged and merged[-1]["speaker_id"] == normalized["speaker_id"]:
            merged[-1]["end_time"] = normalized["end_time"]
            merged[-1]["text"] = f"{merged[-1]['text']} {normalized['text']}".strip()
        else:
            merged.append(normalized)
    return merged


def format_segments(
    segments: List[Dict[str, Any]],
    *,
    style: str = "dialogue",
    speaker_names: Optional[Dict[str, str]] = None,
) -> str:
    """Render structured segments as a readable speaker transcript."""
    merged = merge_consecutive_segments(segments)
    if not merged:
        return ""

    if style == "lines":
        lines: List[str] = []
        for seg in merged:
            start = _seconds_to_timestamp(seg["start_time"])
            end = _seconds_to_timestamp(seg["end_time"])
            lines.append(
                f"[{start} - {end}] {_speaker_label(seg['speaker_id'], speaker_names)}: {seg['text']}"
            )
        return "\n".join(lines)

    blocks: List[str] = []
    for seg in merged:
        start = _seconds_to_timestamp(seg["start_time"], precise=True)
        end = _seconds_to_timestamp(seg["end_time"], precise=True)
        label = _speaker_label(seg["speaker_id"], speaker_names)
        blocks.append(f"**{label}** · `{start}` – `{end}`\n\n{seg['text']}")
    return "\n\n---\n\n".join(blocks)


def parse_model_output(raw_text: str) -> List[Dict[str, Any]]:
    """Parse VibeVoice JSON even when prefixed with `assistant` or truncated."""
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.strip()
    text = re.sub(r"^(assistant|system|user)\s*", "", text, flags=re.IGNORECASE)

    start = text.find("[")
    if start == -1:
        start = text.find("{")
    if start == -1:
        return []

    payload = text[start:]
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    recovered: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = payload.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(payload, idx)
        except json.JSONDecodeError:
            idx = payload.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            recovered.append(obj)
        idx = payload.find("{", end)
    return recovered


def format_transcript(
    result: Dict[str, Any],
    speaker_names: Optional[Dict[str, str]] = None,
) -> str:
    segments = result.get("segments") or []
    if not segments:
        segments = parse_model_output(result.get("raw_text") or "")
    formatted = format_segments(segments, speaker_names=speaker_names)
    if formatted:
        return formatted
    return "No spoken transcript could be parsed from the model output."


@dataclass
class VibeVoiceTranscriber:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    attn_implementation: str = "auto"
    max_new_tokens: int = 8192

    def __post_init__(self) -> None:
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        self._processor_cls = VibeVoiceASRProcessor
        self._model_cls = VibeVoiceASRForConditionalGeneration
        self.processor = None
        self.model = None
        self._resolved_device = "cpu"
        self._resolved_attn = "sdpa"

    def load(self) -> str:
        import torch

        device, attn = detect_device_and_attn(self.device, self.attn_implementation)
        dtype = torch.float32 if device in ("mps", "cpu") else torch.bfloat16

        self.processor = self._processor_cls.from_pretrained(self.model_id)
        if device == "mps":
            self.model = self._model_cls.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=None,
                attn_implementation=attn,
                trust_remote_code=True,
            )
            self.model = self.model.to("mps")
        elif device == "auto":
            self.model = self._model_cls.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map="auto",
                attn_implementation=attn,
                trust_remote_code=True,
            )
        else:
            self.model = self._model_cls.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=None,
                attn_implementation=attn,
                trust_remote_code=True,
            )
            self.model = self.model.to(device)

        self.model.eval()
        self._resolved_device = (
            device if device != "auto" else str(next(self.model.parameters()).device)
        )
        self._resolved_attn = attn
        return f"Loaded {self.model_id} on {self._resolved_device} ({attn}, {dtype})"

    def transcribe(
        self,
        audio_path: str,
        hotwords: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        context_info = hotwords.strip() if hotwords and hotwords.strip() else None
        inputs = self.processor(
            audio=audio_path,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info,
        )
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        generation_config = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_config)

        elapsed = time.time() - start
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        raw_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        segments: List[Dict[str, Any]] = []
        try:
            segments = self.processor.post_process_transcription(raw_text) or []
        except Exception:
            segments = []
        if not segments:
            segments = parse_model_output(raw_text)

        return {
            "raw_text": raw_text,
            "segments": segments,
            "generation_time": elapsed,
            "device": self._resolved_device,
        }


def create_transcriber(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
) -> VibeVoiceTranscriber:
    load_dotenv()
    setup_cache_env()
    transcriber = VibeVoiceTranscriber(model_id=model_id, device=device)
    transcriber.load()
    return transcriber
