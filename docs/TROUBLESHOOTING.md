# Fehlerbehebung - Pipeline hängt beim Laden

## Problem: Pipeline.from_pretrained() hängt

Das ist ein bekanntes Problem mit pyannote-audio und HuggingFace model loading.

### Symptome:
- Skript bleibt bei "Loading pipeline..." hängen
- Keine Fehlermeldung, nur HTTP-Requests im Log
- Prozess läuft stundenlang ohne Fortschritt

### Ursache:
Das Skript versucht, die Modelle von HuggingFace zu verifizieren, aber:
- Langsame Netzwerkverbindung
- HuggingFace-Server antwortet langsam
- Cache ist korrupt oder unvollständig

## Lösungen

### Lösung 1: Fix-Script verwenden (EMPFOHLEN)

```bash
./fix_models.sh
```

Das Script:
- Löscht den Cache komplett
- Lädt alle Modelle neu herunter
- Verwendet `huggingface-cli` (robuster als Python API)

### Lösung 2: Manueller Cache-Clear

```bash
# Cache löschen
rm -rf ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1

# Neu downloaden
python download_models.py
```

### Lösung 3: Timeout erhöhen

In `transcribe_with_speakers.py`, Zeile ~738:

```python
diarization = perform_diarization(
    str(audio_path),
    hf_token=hf_token,
    num_speakers=num_speakers,
    max_duration=max_duration,
    pipeline_timeout=600  # 10 Minuten statt 3
)
```

### Lösung 4: Offline-Modus (wenn Modelle vorhanden)

Füge in `transcribe_with_speakers.py` nach Zeile 187 ein:

```python
from huggingface_hub import snapshot_download
import os

# Force local cache usage
os.environ["HF_HUB_OFFLINE"] = "1"
```

Dies verhindert, dass online nach Updates gesucht wird.

### Lösung 5: Alternative Pipeline verwenden

Statt `pyannote/speaker-diarization-community-1` kannst du auch
`pyannote/speaker-diarization-3.1` verwenden (neuere Version):

In `transcribe_with_speakers.py`, Zeile 187:

```python
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",  # statt community-1
    token=hf_token,
    use_auth_token=hf_token
)
```

**WICHTIG**: Du musst die Nutzungsbedingungen bei HuggingFace akzeptieren!

## Debug-Informationen sammeln

Wenn nichts hilft, sammle diese Informationen:

```bash
# 1. Cache-Größe prüfen
du -sh ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1

# 2. Netzwerk-Test
curl -I https://huggingface.co/pyannote/speaker-diarization-community-1

# 3. Token-Test
python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami(token='$HUGGINGFACE_TOKEN'))"

# 4. Log analysieren
tail -50 transcription_debug.log
```

## Alternative: Nur Transkription ohne Diarization

Falls die Diarization nicht funktioniert, kannst du auch nur Whisper verwenden:

```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.m4a")

for segment in segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
```

Das gibt dir die Transkription ohne Speaker-Zuordnung.
