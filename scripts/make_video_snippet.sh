#!/usr/bin/env bash
set -euo pipefail
SRC="/mnt/c/Users/brude/Downloads/BB JF-20260814_110255UTC-Meeting Recording.mp4"
OUT="/home/brude/projects/SpeakerSeparation/data/BB_JF_meeting_2min.mp4"
LOG="/home/brude/projects/SpeakerSeparation/data/snippet_test.log"

{
  echo "=== $(date -Iseconds) ==="
  ls -lh "$SRC"
  ffmpeg -y -i "$SRC" -t 120 -c copy "$OUT"
  ls -lh "$OUT"
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT"
  cd /home/brude/projects/SpeakerSeparation
  .venv-vibevoice/bin/python <<'PY'
from vibevoice_poc import prepare_media_for_transcription
p = "/home/brude/projects/SpeakerSeparation/data/BB_JF_meeting_2min.mp4"
audio, stem = prepare_media_for_transcription(None, p)
import os
print("stem:", stem)
print("audio:", audio)
print("audio_bytes:", os.path.getsize(audio))
PY
  echo "OK ffmpeg + extract only (no model load)"
} >"$LOG" 2>&1
