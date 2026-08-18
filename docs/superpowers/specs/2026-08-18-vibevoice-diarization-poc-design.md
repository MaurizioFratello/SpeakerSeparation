# VibeVoice-ASR Diarization POC Design

## Goal

Local-only browser POC for speaker-attributed transcription using
[microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR),
isolated from the existing pyannote + Whisper/Parakeet pipeline.

## Scope (v1)

- Single Gradio app bound to localhost
- Upload one audio file, optional hotwords/context
- Run VibeVoice-ASR and show speaker-labeled transcript with timestamps
- Optional raw model output pane for debugging
- No GUI/service integration, no queue, no auth, no YouTube input

## Architecture

```
app_vibevoice.py          # Gradio UI entrypoint
vibevoice_poc.py          # model load, inference, output formatting
scripts/requirements_vibevoice_poc.txt
```

Flow:

1. App startup configures repo-local HF cache (same pattern as `transcribe_simple.py`)
2. Load `VibeVoiceASRProcessor` + `VibeVoiceASRForConditionalGeneration` once
3. User uploads audio and clicks Transcribe
4. Processor runs inference with optional `context_info` (hotwords)
5. `post_process_transcription()` yields segments; formatter renders readable text

## Output Format

```
[00:12 - 00:18] Speaker 1: Hello everyone.
[00:18 - 00:24] Speaker 2: Thanks for joining.
```

If parsing fails, show raw model text in the transcript pane.

## Dependencies

Install Microsoft `vibevoice` package from GitHub plus Gradio and audio libs.
Requires GPU strongly recommended (~9B model). CPU fallback allowed but slow.

## Error Handling

- Missing audio upload
- Model load failure (VRAM/CUDA)
- Inference exceptions
- Unsupported audio format

All surfaced as readable Gradio messages.

## Out of Scope

- Integration with `transcribe_simple.py`, GUI, or FastAPI service
- Streaming transcription UI
- Per-segment audio players
- Benchmarking against pyannote pipeline
