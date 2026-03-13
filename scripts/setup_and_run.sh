#!/bin/bash
# Complete setup and run script for speaker-separated transcription

set -e  # Exit on error

echo "============================================================"
echo "Speaker-Separated Transcription - Setup & Run"
echo "============================================================"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your HuggingFace token:"
    echo "  HUGGINGFACE_TOKEN=your_token_here"
    echo ""
    exit 1
fi

# Activate conda environment
echo ""
echo "Step 1: Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate speaker_separation 2>/dev/null || {
    echo "ERROR: Failed to activate conda environment 'speaker_separation'"
    echo "Please create it first with:"
    echo "  conda create -n speaker_separation python=3.11"
    echo "  conda activate speaker_separation"
    echo "  pip install -r requirements_transcription.txt"
    exit 1
}
echo "✓ Environment activated"

# Check if models need to be downloaded
echo ""
echo "Step 2: Checking models..."
if [ ! -d "$HOME/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1" ]; then
    echo "Models not found. Running download script..."
    python download_models.py || {
        echo ""
        echo "ERROR: Model download failed!"
        echo "Please check your internet connection and HuggingFace token."
        exit 1
    }
else
    echo "✓ Models already downloaded"
fi

# Ask user if they want to run in test mode
echo ""
echo "============================================================"
echo "Run Configuration"
echo "============================================================"
read -p "Run in TEST MODE? (1 minute only, faster) [y/N]: " test_mode
if [[ $test_mode =~ ^[Yy]$ ]]; then
    export TEST_MODE="true"
    echo "✓ Test mode enabled (will process first 60 seconds only)"
else
    export TEST_MODE="false"
    echo "✓ Full mode enabled (will process entire file)"
fi

# Run the transcription
echo ""
echo "Step 3: Running transcription..."
echo "============================================================"
python transcribe_with_speakers.py

echo ""
echo "============================================================"
echo "Transcription complete!"
echo "============================================================"
