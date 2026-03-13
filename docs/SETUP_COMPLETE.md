# Setup Complete! 🎉

## What Was Done

✅ **Conda environment created**: `speaker_separation` (Python 3.11)  
✅ **All dependencies installed**:
   - PyTorch 2.8.0 + torchaudio 2.8.0
   - pyannote.audio 4.0.3 (speaker diarization)
   - openai-whisper (transcription)
   - huggingface-hub (model access)
   - All supporting libraries

✅ **Verification**: All imports successful, MPS (Apple Silicon GPU) available

## Next Steps: HuggingFace Token Setup

To use pyannote-audio models, you need to:

### 1. Accept Model Terms

Visit and accept terms for these models:
- **Main pipeline**: https://hf.co/pyannote/speaker-diarization-community-1
- **Segmentation model**: https://hf.co/pyannote/segmentation

Click "Agree and access repository" on each page.

### 2. Create Access Token

1. Go to: https://hf.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "pyannote-audio")
4. Select "Read" access
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

### 3. Set the Token

**Option A: Environment variable (recommended)**
```bash
export HUGGINGFACE_TOKEN='your_token_here'
```

**Option B: Enter when prompted**
The script will ask for it if not set.

## Running the Transcription

### Method 1: Using the helper script
```bash
./run_transcription.sh
```

### Method 2: Manual activation
```bash
conda activate speaker_separation
python transcribe_with_speakers.py
```

### Method 3: Direct with conda run
```bash
conda run -n speaker_separation python transcribe_with_speakers.py
```

## What to Expect

1. **First run**: Models will be downloaded (~500MB-3GB depending on Whisper model)
2. **Processing time** for 45-minute file:
   - Diarization: ~5-10 minutes
   - Transcription: ~15-30 minutes
   - **Total: ~20-40 minutes**

3. **Output files**:
   - `Music Company Media Productions 10_transcript.txt` (human-readable)
   - `Music Company Media Productions 10_transcript.json` (structured data)

## Configuration

Edit `transcribe_with_speakers.py` to customize:

- **Whisper model** (line ~200): `"tiny"`, `"base"` (default), `"small"`, `"medium"`, `"large"`
- **Speaker count** (line ~250): Set to `2` (you mentioned 2 speakers) or `None` for auto-detect

## Troubleshooting

### "HuggingFace token required"
- Make sure you accepted model terms
- Create and set the token (see steps above)

### torchcodec warning
- This is just a warning, not an error
- The script uses torchaudio which works fine with your ffmpeg installation

### Out of memory
- Use smaller Whisper model (`"tiny"` or `"base"`)
- Close other applications

### Slow processing
- This is normal for 45-minute files
- First run downloads models (one-time)
- Subsequent runs will be faster

## Environment Info

- **Conda environment**: `speaker_separation`
- **Python version**: 3.11
- **Location**: `/opt/anaconda3/envs/speaker_separation`
- **GPU**: MPS (Apple Silicon) available ✓

## Files Created

- `transcribe_with_speakers.py` - Main transcription script
- `requirements_transcription.txt` - Dependencies list
- `run_transcription.sh` - Helper script to run transcription
- `TRANSCRIPTION_GUIDE.md` - Detailed usage guide
- `SETUP_COMPLETE.md` - This file

---

**Ready to transcribe!** Just set your HuggingFace token and run the script. 🚀

