#!/bin/bash
# Activation script for running transcription with conda environment

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate speaker_separation

# Check if token is set, if not prompt user
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "============================================================"
    echo "HuggingFace Token Not Set"
    echo "============================================================"
    echo "You can either:"
    echo "1. Set it before running: export HUGGINGFACE_TOKEN='your_token'"
    echo "2. Enter it now (will be used for this run only)"
    echo ""
    read -sp "Enter HuggingFace token (or press Enter to skip): " token
    echo ""
    if [ -n "$token" ]; then
        export HUGGINGFACE_TOKEN="$token"
    fi
fi

# Run the transcription script
python transcribe_with_speakers.py

