# Speaker-Separated Transcription - Verbesserte Version

## Probleme behoben

1. **Pipeline-Hänger**: Timeout-Handling für HuggingFace Model-Downloads
2. **Lange Wartezeiten**: faster-whisper Integration (4-10x schneller)
3. **Fehlende Progress-Anzeigen**: Detaillierte Progress-Meldungen
4. **Unklare Fehler**: Bessere Fehlermeldungen und Debugging

## Schnellstart

### Option 1: Automatisches Setup (Empfohlen)

```bash
./setup_and_run.sh
```

Das Skript wird:
- Conda-Environment aktivieren
- Fehlende Modelle herunterladen
- Test-Modus abfragen (60 Sekunden vs. volle Datei)
- Transkription starten

### Option 2: Manuelles Setup

1. **Modelle vorab herunterladen**:
```bash
conda activate speaker_separation
python download_models.py
```

2. **Transkription starten**:
```bash
# Test-Modus (erste 60 Sekunden, schnell)
export TEST_MODE=true
python transcribe_with_speakers.py

# Volle Datei
export TEST_MODE=false
python transcribe_with_speakers.py
```

## Was wurde verbessert?

### 1. Timeout-Handling
- Pipeline-Loading hat jetzt einen 3-Minuten-Timeout
- Progress-Meldungen alle 10 Sekunden
- Klare Fehlermeldungen bei Timeout

### 2. Model-Download-Skript
- `download_models.py` lädt alle Modelle vorab herunter
- Vermeidet Hänger während der Verarbeitung
- Verifiziert, dass alle Modelle korrekt geladen werden

### 3. faster-whisper Integration
- 4-10x schneller als openai-whisper
- Bereits installiert und konfiguriert
- Automatische Fallback zu openai-whisper

### 4. Besseres Logging
- Detaillierte Debug-Logs in `transcription_debug.log`
- Timestamps für jeden Schritt
- Klare Progress-Meldungen im Terminal

## Erwartete Laufzeiten

### Test-Modus (60 Sekunden Audio):
- Diarization: ~30-60 Sekunden
- Transkription: ~20-40 Sekunden
- **Gesamt: ~1-2 Minuten**

### Volle Datei (45 Minuten Audio):
- Diarization: ~5-10 Minuten
- Transkription: ~3-8 Minuten (mit faster-whisper)
- **Gesamt: ~8-18 Minuten**

## Fehlerbehebung

### Problem: Pipeline lädt nicht
```bash
# Lösung 1: Cache leeren
rm -rf ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1

# Lösung 2: Modelle neu herunterladen
python download_models.py
```

### Problem: "Timeout after 180s"
```bash
# Timeout erhöhen (in transcribe_with_speakers.py, Zeile ~738):
pipeline_timeout=300  # 5 Minuten statt 3
```

### Problem: Zu langsam
```bash
# Test-Modus verwenden (nur erste 60s):
export TEST_MODE=true

# Oder kleineres Whisper-Modell (in .env):
WHISPER_MODEL=tiny  # statt base
```

## Nächste Schritte

1. **Test durchführen**:
```bash
export TEST_MODE=true
./setup_and_run.sh
```

2. **Ergebnis prüfen**:
   - `Music Company Media Productions 10_transcript.txt`
   - `Music Company Media Productions 10_transcript.json`

3. **Volle Datei verarbeiten**:
```bash
export TEST_MODE=false
./setup_and_run.sh
```

## Konfiguration

### Umgebungsvariablen (.env)
```bash
HUGGINGFACE_TOKEN=hf_...        # Erforderlich
NUM_SPEAKERS=2                  # Optional: Anzahl Speaker
WHISPER_MODEL=base              # Optional: tiny, base, small, medium, large
TEST_MODE=false                 # Optional: true für Test-Modus
```

## Ausgabe-Formate

### Text-Format (.txt)
```
[0.0s - 2.5s] SPEAKER_00: Hallo, wie geht es dir?
[2.7s - 5.1s] SPEAKER_01: Mir geht es gut, danke!
```

### JSON-Format (.json)
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.0,
    "end": 2.5,
    "text": "Hallo, wie geht es dir?"
  }
]
```
