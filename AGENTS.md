## Learned User Preferences

- Prefer creating the project virtual environment with `uv` when system `python3` lacks `venv` or `pip`.
- Use `UV_CACHE_DIR=.uv-cache` for `uv` installs when cache or home-directory permissions are tight.
- For local GUI and transcription runs, install from `requirements_gui.txt` and `scripts/requirements_transcription.txt`.

## Learned Workspace Facts

- `gui_main.py` and `transcribe_simple.py` default Hugging Face caches to a repo-local `./.cache` tree (`XDG_CACHE_HOME`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`) so model downloads work without relying on `~/.cache` permissions.
- Diarization needs `HUGGINGFACE_TOKEN` in `.env` and Hugging Face terms/access for each gated checkpoint the pipeline pulls (not only the top-level diarization model; dependent repos must match the same HF account as the token).
- `scripts/requirements_transcription.txt` pins `pyannote.audio>=3.1.1,<4.0` and `numpy<2`, and includes `soundfile` and `nemo_toolkit[asr]`.
- `gui_main.py` and `transcribe_simple.py` patch `torch.load` so `weights_only=None` becomes `False`, matching Lightning/pyannote expectations under PyTorch 2.6+.
- Headless agent API lives under `service/` (FastAPI); the CUDA `Dockerfile` uses `scripts/docker_constraints.txt` to keep `torch` / `torchvision` / `torchaudio` aligned with the CUDA 12.4 base image.
