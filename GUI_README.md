# GUI für Speaker Diarization & Transcription

Moderne PySide6-basierte GUI für die Speaker-Diarization und Transkription.

## Installation

1. **Conda-Umgebung aktivieren:**
   ```bash
   conda activate speaker_separation
   ```

2. **GUI-Dependencies installieren:**
   ```bash
   pip install -r requirements_gui.txt
   ```

## Verwendung

### GUI starten

```bash
python gui_main.py
```

### Features

- **Drag & Drop:** Audio-Dateien direkt in die GUI ziehen
- **Format-Unterstützung:** MP3, M4A, WAV, AIFF, FLAC, OGG, WMA
- **Speaker-Auswahl:**
  - Auto: Automatische Erkennung der Sprecheranzahl
  - Manuell: Vordefinierte Anzahl von Sprechern
- **Sprach-Auswahl:**
  - Automatic: Parakeet mit automatischer Sprache
  - English / German: faster-whisper mit fest vorgegebener Sprache (OpenAI Whisper fallback auf MPS)
- **Live-Transkript:** Segmente erscheinen in Echtzeit während der Verarbeitung
- **Auto-Save:** Transkript wird automatisch als `{dateiname}_transcript.txt` gespeichert
- **YouTube-URL Input:** YouTube-Link einfügen, Audio via yt-dlp laden und verarbeiten
- **Markdown Export:** YouTube-Transkripte werden als `exports/{video_id}_transcript.md` gespeichert
- **Cancel-Funktion:** Verarbeitung kann jederzeit abgebrochen werden
- **Fortschrittsanzeige:** Visueller Fortschrittsbalken und Status-Updates

## Anforderungen

- **HUGGINGFACE_TOKEN:** Muss in `.env`-Datei vorhanden sein
- **Transkriptions-Dependencies:** `scripts/requirements_transcription.txt` installieren
- **Optional:** `WHISPER_MODEL` (Standard: `turbo`) und `WHISPER_BACKEND` (`auto`, `faster-whisper`, `openai-whisper`)
- **ffmpeg:** Muss im System-PATH verfügbar sein
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`
  - Windows: Download von https://ffmpeg.org/download.html
- **yt-dlp:** Für YouTube-Downloads (`pip install yt-dlp`)

## Dateistruktur

```
SpeakerSeparation/
├── gui_main.py              # GUI-Einstiegspunkt
├── gui/
│   ├── __init__.py
│   ├── main_window.py       # Hauptfenster
│   ├── transcription_worker.py  # Background-Worker
│   ├── audio_converter.py   # Audio-Konvertierung
│   ├── youtube_download.py  # YouTube Audio-Download
│   └── markdown_export.py   # Markdown-Export
└── requirements_gui.txt     # GUI-Dependencies
```

## Technische Details

- **Framework:** PySide6 (Qt6)
- **Threading:** QThread für asynchrone Verarbeitung
- **Streaming:** Live-Updates während der Transkription
- **Architektur:** MVC-ähnliches Pattern mit Signal/Slot-Kommunikation
