# VibeVoice Unified Media Drop Design

## Goal

Replace the Audio/Video tabs in the VibeVoice Gradio POC with a single drop zone. Any dropped audio or video file is converted immediately to the model target format before Transcribe.

## Target format

24 kHz mono 16-bit PCM WAV (`ffmpeg -vn -ac 1 -ar 24000 -c:a pcm_s16le`). This matches the existing video-extract path so VibeVoice always sees the same audio.

WAV files are converted too (normalized), not passed through.

## UI

- One `gr.File` drop zone accepting audio and video (no tabs).
- On drop (Gradio `change`), convert immediately.
- Status line: converting / ready / failed.
- On success, a non-interactive `gr.Audio` preview plays the converted WAV.
- Transcribe stays a separate click so hotwords and max tokens still apply.
- Transcribe uses the converted WAV. Markdown export keeps the original file stem.
- Dropping a new file replaces the previous conversion. Clearing the drop zone clears preview and converted state.
- Failed conversion blocks transcription with a readable error.

## Conversion

`convert_media_to_wav(media_path) -> wav_path` in `vibevoice_poc.py` is the single ffmpeg helper for audio and video. `extract_audio_from_video` becomes a thin wrapper around it. `prepare_media_for_transcription` always converts the chosen source (video if provided, else audio).

## Error handling

- Missing file / empty drop: ask the user to drop a file.
- ffmpeg failure or empty output: show the conversion error; Transcribe must not run on the original upload.
- Missing ffmpeg: surface the subprocess error as a readable Gradio message.

## Out of scope

- Auto-start transcription on drop
- Visual speaker detection from video
- Changing the existing pyannote / GUI pipeline
