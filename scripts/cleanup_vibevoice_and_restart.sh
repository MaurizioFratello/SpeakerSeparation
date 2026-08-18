#!/usr/bin/env bash
# Remove stray VibeVoice HF caches and restart app_vibevoice.py
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
LOG="$REPO/data/cache_cleanup.log"
mkdir -p "$REPO/data"

exec > >(tee "$LOG") 2>&1

echo "=== VibeVoice cache cleanup $(date -Iseconds) ==="

REPO_HUB="$REPO/.cache/huggingface/hub"
HOME_HUB="$HOME/.cache/huggingface/hub"
MODELS=(
  "models--microsoft--VibeVoice-ASR"
  "models--Qwen--Qwen2.5-1.5B"
)

audit_dir() {
  local label="$1"
  local dir="$2"
  if [[ -d "$dir" ]]; then
    echo "--- $label: $dir ---"
    du -sh "$dir" || true
    for m in "${MODELS[@]}"; do
      if [[ -d "$dir/$m" ]]; then
        du -sh "$dir/$m"
        stat -c '%y %n' "$dir/$m" 2>/dev/null || ls -ld "$dir/$m"
      fi
    done
  else
    echo "--- $label: (missing) $dir ---"
  fi
}

echo "Before cleanup:"
audit_dir "repo hub" "$REPO_HUB"
audit_dir "home hub" "$HOME_HUB"

# Prefer repo-local cache for this project. Drop home-hub duplicates when repo already has them.
if [[ -d "$HOME_HUB" ]]; then
  for m in "${MODELS[@]}"; do
    home_model="$HOME_HUB/$m"
    repo_model="$REPO_HUB/$m"
    if [[ -d "$home_model" && -d "$repo_model" ]]; then
      echo "REMOVE duplicate home cache: $home_model"
      rm -rf "$home_model"
    elif [[ -d "$home_model" && ! -d "$repo_model" ]]; then
      echo "MOVE home cache into repo: $home_model -> $repo_model"
      mkdir -p "$REPO_HUB"
      mv "$home_model" "$repo_model"
    fi
  done
fi

# Drop incomplete HF downloads and temp extraction dirs.
find "$REPO/.cache" "$HOME/.cache/huggingface" -name '*.incomplete' -print -delete 2>/dev/null || true
rm -rf /tmp/vibevoice_video_* 2>/dev/null || true

echo "After cleanup:"
audit_dir "repo hub" "$REPO_HUB"
audit_dir "home hub" "$HOME_HUB"

echo "=== Restarting app on :7861 ==="
pkill -f 'app_vibevoice.py' 2>/dev/null || true
sleep 2
nohup "$REPO/.venv-vibevoice/bin/python" "$REPO/app_vibevoice.py" --port 7861 \
  >"$REPO/data/app_vibevoice.log" 2>&1 &
echo "app pid: $!"
sleep 5
tail -25 "$REPO/data/app_vibevoice.log" || true
echo "=== Done. Log: $LOG ==="
