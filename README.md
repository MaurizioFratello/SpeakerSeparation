# Speaker Diarization & Transcription

Speaker-attributed transcription using pyannote-audio for diarization and faster-whisper for transcription. **Optimized for Apple Silicon MPS GPU.**

## Quick Start

```bash
# Activate environment
conda activate speaker_separation

# Process audio with GPU acceleration
NUM_SPEAKERS=2 python transcribe_simple.py "Music Company Media Productions 10.m4a"

# Test mode (1 minute only)
TEST_MODE=true NUM_SPEAKERS=2 python transcribe_simple.py "your_audio.m4a"
```

## Output Format

```
[   0.0s -    1.6s] SPEAKER_01: Text here...
[   1.6s -    5.1s] SPEAKER_00: More text...
```

## Requirements

- **Environment**: Create `.env` file with `HUGGINGFACE_TOKEN=hf_your_token_here`
- **Token**: Get from https://hf.co/settings/tokens
- **Model Access**: Accept terms at https://hf.co/pyannote/speaker-diarization-community-1

## Features

- **MPS GPU Acceleration**: Automatically uses Apple Silicon GPU
- **Progress Tracking**: Real-time progress display
- **Test Mode**: Process first 60 seconds for quick testing
- **Speaker Detection**: Auto-detect or specify number of speakers

## Performance

- **Hardware**: Apple Silicon Mac with MPS GPU
- **Test Audio**: 2 minutes → ~7 seconds processing
- **Full Audio**: 44.7 minutes → ~20-30 minutes estimated

## Project Structure

```
.
├── transcribe_simple.py      # Main script (MPS GPU enabled)
├── SOLUTION_SUMMARY.md        # Technical documentation
├── .env                       # Configuration (create this)
├── tests/                     # Test scripts and benchmarks
├── docs/                      # Additional documentation
├── scripts/                   # Setup and utility scripts
└── archive/                   # Old versions and logs
```

## Environment Variables

- `HUGGINGFACE_TOKEN`: Required - your HuggingFace API token
- `NUM_SPEAKERS`: Optional - force specific number of speakers
- `TEST_MODE`: Optional - set to `true` to process only first minute
- `PYANNOTE_METRICS_ENABLED`: Automatically set to `0` (disables telemetry)

## Troubleshooting

See [SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md) for detailed technical information.
