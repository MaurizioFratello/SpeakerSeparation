#!/usr/bin/env python3
"""Local Gradio POC for VibeVoice-ASR speaker diarization."""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import gradio as gr

from vibevoice_poc import (
    DEFAULT_MODEL_ID,
    create_transcriber,
    extract_speaker_samples,
    format_transcript,
    merge_consecutive_segments,
    parse_model_output,
    prepare_dropped_media,
    setup_cache_env,
    speaker_samples_to_html,
    write_markdown_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app_vibevoice")

MAX_SPEAKERS = 6
transcriber = None


def _ensure_model(model_id: str, device: str) -> str:
    global transcriber
    if transcriber is None:
        logger.info("Loading model %s on %s", model_id, device)
        transcriber = create_transcriber(model_id=model_id, device=device)
        return f"Loaded {transcriber.model_id} on {transcriber._resolved_device}"
    return f"Reusing loaded model {transcriber.model_id}"


def _normalize_audio_input(audio_input: Any) -> str | None:
    if audio_input is None:
        return None
    if isinstance(audio_input, str):
        return audio_input
    if isinstance(audio_input, Path):
        return str(audio_input)
    if isinstance(audio_input, dict):
        path = audio_input.get("path") or audio_input.get("name")
        return str(path) if path else None
    if isinstance(audio_input, (list, tuple)) and len(audio_input) == 2:
        sample_rate, data = audio_input
        import numpy as np
        import soundfile as sf
        import tempfile

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp.close()
        array = np.asarray(data)
        if array.dtype != np.int16:
            array = (array * 32768.0).astype(np.int16)
        sf.write(temp.name, array, int(sample_rate), subtype="PCM_16")
        return temp.name
    return None


def _empty_speaker_updates() -> List[Any]:
    html = "<p>Speaker samples will appear here after transcription.</p>"
    name_updates = [gr.update(visible=False, value="") for _ in range(MAX_SPEAKERS)]
    return [html, *name_updates]


def _speaker_panel_updates(
    samples: Dict[str, str],
    speaker_names: Optional[Dict[str, str]] = None,
) -> List[Any]:
    ordered_ids = sorted(samples.keys(), key=str)
    html = speaker_samples_to_html(samples, speaker_names)
    name_updates: List[Any] = []
    for index in range(MAX_SPEAKERS):
        if index < len(ordered_ids):
            speaker_id = ordered_ids[index]
            default_name = (speaker_names or {}).get(str(speaker_id), f"Speaker {speaker_id}")
            name_updates.append(
                gr.update(
                    value=default_name,
                    visible=True,
                    label=f"Name for Speaker {speaker_id}",
                )
            )
        else:
            name_updates.append(gr.update(visible=False, value=""))
    return [html, *name_updates]


def _build_transcript_header(result: Dict[str, Any], segments: List[Dict[str, Any]]) -> str:
    elapsed = result.get("generation_time", 0.0)
    spoken_turns = len(merge_consecutive_segments(segments))
    return (
        f"**Done in {elapsed:.1f}s** · {spoken_turns} speaker turn(s) · "
        f"device=`{result.get('device', 'unknown')}`"
    )


def _file_path_and_name(file_input: Any) -> tuple[str | None, str | None]:
    path = _normalize_audio_input(file_input)
    if not path:
        return None, None
    orig = None
    if isinstance(file_input, dict):
        orig = file_input.get("orig_name") or file_input.get("origName")
    orig = orig or getattr(file_input, "orig_name", None)
    display_name = Path(orig).name if orig else Path(path).name
    return path, display_name


def on_media_drop(file_input: Any) -> Generator[tuple[str, Any, Dict[str, Any]], None, None]:
    empty: Dict[str, Any] = {"wav_path": None, "source_stem": None, "source_name": None}
    path, display_name = _file_path_and_name(file_input)
    if not path:
        yield "Drop an audio or video file.", None, empty
        return

    shown = display_name or Path(path).name
    yield f"Converting `{shown}` to 24 kHz mono WAV...", None, empty
    try:
        prepared = prepare_dropped_media(path)
    except Exception as exc:
        logger.exception("Media conversion failed")
        yield f"Conversion failed: {exc}", None, empty
        return

    prepared["source_name"] = shown
    prepared["source_stem"] = Path(shown).stem
    prepared["status"] = f"Ready: `{shown}` → `{Path(prepared['wav_path']).name}`"
    yield prepared["status"], prepared["wav_path"], prepared


def transcribe_file(
    media_state: Dict[str, Any],
    hotwords: str,
    max_new_tokens: int,
    progress: gr.Progress = gr.Progress(),
) -> Generator[tuple[Any, ...], None, None]:
    empty_state: Dict[str, Any] = {}
    media_state = media_state or {}
    resolved_path = media_state.get("wav_path")
    display_stem = media_state.get("source_stem")

    if not resolved_path:
        yield (
            "Please drop an audio or video file and wait for conversion to finish.",
            "",
            empty_state,
            None,
            *_empty_speaker_updates(),
        )
        return

    if transcriber is None:
        yield ("Model is not loaded yet. Restart the app.", "", empty_state, None, *_empty_speaker_updates())
        return

    display_name = display_stem or Path(resolved_path).stem
    logger.info("Transcribe requested for %s (converted: %s)", display_name, resolved_path)
    yield (
        f"Transcribing `{display_name}`...\nThis can take several minutes for longer media. Please wait.",
        "",
        empty_state,
        None,
        *_empty_speaker_updates(),
    )
    progress(0.1, desc="Running VibeVoice-ASR")

    try:
        result = transcriber.transcribe(
            audio_path=resolved_path,
            hotwords=hotwords,
            max_new_tokens=int(max_new_tokens),
        )
    except Exception as exc:
        logger.exception("Transcription failed")
        yield (
            f"Transcription failed: {exc}",
            traceback.format_exc(),
            empty_state,
            None,
            *_empty_speaker_updates(),
        )
        return

    try:
        segments = result.get("segments") or parse_model_output(result.get("raw_text") or "")
        samples = extract_speaker_samples(resolved_path, segments)
        speaker_ids = sorted(samples.keys(), key=str)
        default_names = {speaker_id: f"Speaker {speaker_id}" for speaker_id in speaker_ids}

        state = {
            "segments": segments,
            "source_name": display_name,
            "speaker_ids": speaker_ids,
            "speaker_names": default_names,
            "result": {
                "segments": segments,
                "raw_text": result.get("raw_text", ""),
                "generation_time": result.get("generation_time", 0.0),
                "device": result.get("device", "unknown"),
            },
        }

        transcript = format_transcript(result, speaker_names=default_names)
        header = _build_transcript_header(result, segments)
        raw = result.get("raw_text", "")
        progress(1.0, desc="Complete")
        logger.info(
            "Transcription complete in %.1fs (%s speakers)",
            result.get("generation_time", 0.0),
            len(samples),
        )

        yield (
            f"{header}\n\n{transcript}",
            raw,
            state,
            None,
            *_speaker_panel_updates(samples, default_names),
        )
    except Exception as exc:
        logger.exception("Failed to format transcript or speaker samples")
        yield (
            f"Transcription succeeded, but display failed: {exc}",
            traceback.format_exc(),
            empty_state,
            None,
            *_empty_speaker_updates(),
        )


def apply_speaker_names(
    state: Dict[str, Any],
    *name_values: str,
) -> tuple[str, Dict[str, Any], None]:
    if not state or not state.get("segments"):
        return "Transcribe audio first.", state or {}, None

    speaker_ids = state.get("speaker_ids") or []
    speaker_names = {
        str(speaker_ids[index]): (name_values[index] or f"Speaker {speaker_ids[index]}").strip()
        for index in range(min(len(speaker_ids), len(name_values)))
    }
    state = {**state, "speaker_names": speaker_names}

    result = state.get("result") or {"segments": state["segments"]}
    transcript = format_transcript(result, speaker_names=speaker_names)
    header = _build_transcript_header(result, state["segments"])
    return f"{header}\n\n{transcript}", state, None


def download_markdown(state: Dict[str, Any]) -> tuple[Optional[str], str]:
    if not state or not state.get("segments"):
        return None, "Transcribe audio before downloading markdown."

    speaker_names = state.get("speaker_names") or {}
    filename = f"{state.get('source_name', 'transcript')}.md"
    path = write_markdown_file(
        state["segments"],
        title=state.get("source_name", "Transcript"),
        speaker_names=speaker_names,
        filename=filename,
    )
    return path, f"Saved markdown to `{path}`."


def build_ui(model_id: str, device: str) -> gr.Blocks:
    status = _ensure_model(model_id, device)

    with gr.Blocks(title="VibeVoice Diarization POC") as demo:
        gr.Markdown("# VibeVoice Diarization POC")
        gr.Markdown(
            "Local-only experiment using "
            "[microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR). "
            "Drop an audio or video file; it is converted to 24 kHz mono WAV before transcription. "
            "Video files contribute audio only (no visual speaker detection)."
        )
        gr.Markdown(f"**Status:** {status}")

        transcript_state = gr.State({})
        media_state = gr.State({})

        media_input = gr.File(
            label="Drop audio or video",
            file_types=[
                "audio",
                "video",
                ".wav",
                ".mp3",
                ".m4a",
                ".flac",
                ".ogg",
                ".aac",
                ".wma",
                ".opus",
                ".mp4",
                ".mkv",
                ".webm",
                ".mov",
                ".avi",
                ".m4v",
            ],
            type="filepath",
        )
        conversion_status = gr.Markdown("Drop an audio or video file to convert it to 24 kHz mono WAV.")
        audio_preview = gr.Audio(
            label="Converted audio preview",
            type="filepath",
            interactive=False,
        )

        hotwords = gr.Textbox(
            label="Hotwords / context (optional)",
            placeholder="Names, technical terms, topics — one per line",
            lines=3,
        )
        max_tokens = gr.Slider(
            minimum=512,
            maximum=32768,
            value=4096,
            step=512,
            label="Max new tokens",
        )
        run_btn = gr.Button("Transcribe", variant="primary")

        gr.Markdown("## Speaker names")
        gr.Markdown(
            "After transcription, listen to each speaker sample and enter a display name. "
            "Then click **Apply speaker names**."
        )

        speaker_samples_html = gr.HTML(
            value="<p>Speaker samples will appear here after transcription.</p>"
        )
        speaker_names: List[gr.Textbox] = []
        for index in range(MAX_SPEAKERS):
            name = gr.Textbox(
                label=f"Name for Speaker {index}",
                placeholder="Enter a display name",
                visible=False,
            )
            speaker_names.append(name)

        with gr.Row():
            apply_names_btn = gr.Button("Apply speaker names")
            download_btn = gr.Button("Download markdown")

        download_status = gr.Markdown("")
        markdown_file = gr.File(label="Markdown download", interactive=False)

        transcript_out = gr.Markdown(label="Speaker transcript")
        raw_out = gr.Textbox(
            label="Raw model output (debug)",
            lines=12,
            max_lines=30,
        )

        media_input.change(
            fn=on_media_drop,
            inputs=[media_input],
            outputs=[conversion_status, audio_preview, media_state],
            show_progress="full",
        )

        run_btn.click(
            fn=transcribe_file,
            inputs=[media_state, hotwords, max_tokens],
            outputs=[
                transcript_out,
                raw_out,
                transcript_state,
                markdown_file,
                speaker_samples_html,
                *speaker_names,
            ],
            show_progress="full",
        )

        apply_names_btn.click(
            fn=apply_speaker_names,
            inputs=[transcript_state, *speaker_names],
            outputs=[transcript_out, transcript_state, download_status],
        )

        download_btn.click(
            fn=download_markdown,
            inputs=[transcript_state],
            outputs=[markdown_file, download_status],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeVoice-ASR Gradio POC")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id or path")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (local only)")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    args = parser.parse_args()

    setup_cache_env()
    demo = build_ui(args.model, args.device)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
