#!/bin/bash
# Fix hanging model loading issue by clearing cache and re-downloading

echo "============================================================"
echo "Model Loading Fix Script"
echo "============================================================"
echo ""
echo "This script will:"
echo "1. Clear the HuggingFace cache for pyannote models"
echo "2. Force a fresh download of all required models"
echo ""
read -p "Continue? [y/N]: " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Clear pyannote cache
echo ""
echo "Step 1: Clearing pyannote cache..."
CACHE_DIR="$HOME/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1"
if [ -d "$CACHE_DIR" ]; then
    echo "Removing: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
    echo "✓ Cache cleared"
else
    echo "✓ Cache already empty"
fi

# Download models using huggingface-cli
echo ""
echo "Step 2: Downloading models with huggingface-cli..."
echo ""

# Load token from .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Token loaded from .env"
else
    echo "ERROR: .env file not found!"
    echo "Please create .env with: HUGGINGFACE_TOKEN=your_token"
    exit 1
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate speaker_separation 2>/dev/null

# Download using huggingface-cli (more reliable than Python API)
echo ""
echo "Downloading pyannote/speaker-diarization-community-1..."
echo "This may take 2-5 minutes..."
echo ""

huggingface-cli download pyannote/speaker-diarization-community-1 \
    --token "$HUGGINGFACE_TOKEN" \
    --local-dir-use-symlinks False \
    --resume-download

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Models downloaded successfully!"
    echo ""
    echo "Now you can run:"
    echo "  ./setup_and_run.sh"
    echo ""
else
    echo ""
    echo "✗ Download failed!"
    echo ""
    echo "Alternative: Try downloading manually at:"
    echo "  https://huggingface.co/pyannote/speaker-diarization-community-1"
    echo ""
    exit 1
fi
