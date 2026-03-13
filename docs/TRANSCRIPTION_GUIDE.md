# Speaker-Attributed Transcription Guide

This guide explains how to transcribe your 45-minute audio file with two speakers using pyannote-audio (diarization) + Whisper (transcription).

## Overview

The solution combines:
1. **pyannote-audio**: Identifies who spoke when (speaker diarization)
2. **Whisper**: Converts speech to text (transcription)
3. **Combination**: Produces speaker-attributed transcription

## Prerequisites

1. **ffmpeg** (already installed ✓)
2. **Python 3.10+** (you have 3.14.2 ✓)
3. **HuggingFace account and token** (required for pyannote models)

## Setup Steps

### 1. Accept Model Terms

Visit and accept terms for:
- https://hf.co/pyannote/speaker-diarization-community-1
- https://hf.co/pyannote/segmentation (used internally)

### 2. Create HuggingFace Access Token

1. Go to https://hf.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Copy the token

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements_transcription.txt

# Or install individually:
pip install pyannote.audio openai-whisper torch torchaudio huggingface-hub
```

### 4. Set HuggingFace Token

**Option A: Environment variable (recommended)**
```bash
export HUGGINGFACE_TOKEN='your_token_here'
```

**Option B: Enter when prompted**
The script will ask for it if not set.

## Usage

### Basic Usage

```bash
python transcribe_with_speakers.py
```

The script will:
1. Load the audio file: `Music Company Media Productions 10.m4a`
2. Perform speaker diarization (identify speaker segments)
3. Transcribe each segment with Whisper
4. Save results to:
   - `Music Company Media Productions 10_transcript.txt` (human-readable)
   - `Music Company Media Productions 10_transcript.json` (structured data)

### Expected Processing Time

For a 45-minute file:
- **Diarization**: ~5-10 minutes (depends on CPU/GPU)
- **Transcription**: ~15-30 minutes (depends on Whisper model size)

**Total**: ~20-40 minutes

### Customization

Edit `transcribe_with_speakers.py` to change:

- **Whisper model size** (line ~200):
  - `"tiny"`: Fastest, lowest quality (~1GB RAM)
  - `"base"`: Balanced (default, ~1GB RAM)
  - `"small"`: Better quality (~2GB RAM)
  - `"medium"`: High quality (~5GB RAM)
  - `"large"`: Best quality (~10GB RAM)

- **Speaker count** (line ~250):
  - Set `num_speakers = 2` (you mentioned 2 speakers)
  - Or set to `None` to auto-detect

## Output Format

### Text Format (.txt)
```
[0.2s - 1.5s] SPEAKER_00: Hello, how are you?

[1.8s - 3.9s] SPEAKER_01: I'm doing well, thank you.

[4.2s - 5.7s] SPEAKER_00: That's great to hear.
```

### JSON Format (.json)
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.2,
    "end": 1.5,
    "text": "Hello, how are you?"
  },
  ...
]
```

## Troubleshooting

### "HuggingFace token required"
- Make sure you've accepted model terms on HuggingFace
- Create and set the token (see Setup Step 4)

### "whisper not installed"
```bash
pip install openai-whisper
```

### "torchaudio not installed"
```bash
pip install torchaudio
```

### Audio format issues
The script handles .m4a via torchaudio (requires ffmpeg). If issues persist:
- Convert to WAV: `ffmpeg -i input.m4a output.wav`
- Update `audio_file` variable in script

### Out of memory
- Use smaller Whisper model (`"tiny"` or `"base"`)
- Process in chunks (requires script modification)

### Slow processing
- Use GPU if available (automatically detected)
- Use smaller Whisper model
- Consider processing shorter segments

## Advanced: Processing in Chunks

For very long files or memory constraints, you can modify the script to:
1. Split audio into chunks
2. Process each chunk separately
3. Merge results

This requires additional code modifications.

## Notes

- Speaker labels (SPEAKER_00, SPEAKER_01) are arbitrary - they don't identify actual people
- For 45-minute files, expect processing time of 20-40 minutes
- First run will download models (~500MB-3GB depending on Whisper model)
- Models are cached for subsequent runs

