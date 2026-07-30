---
generated: 2026-07-30
---
# Testing

**Analysis Date:** 2026-07-30

## Framework

**Runner:**
- Mixed: `unittest.TestCase` (primary) and pytest-style function tests.
- No `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`, or `conftest.py` exists, so pytest auto-discovery applies.
- No test runner configuration for coverage.

**Assertion library:**
- `unittest.TestCase` methods: `assertEqual`, `assertTrue`, `assertIn`, `assertGreater`, `assertLessEqual`, `assertRaises`, `assertIsNone`, `assertIsNotNone`, `assertIsInstance`, `assertGreaterEqual`, `assertLess`, `fail`.
- Pytest-style plain `assert` statements (in `test_markdown_export.py` and `test_youtube_download.py`).

**Run commands:**
```bash
python -m pytest tests/               # Run all tests via pytest
python -m unittest tests.test_service_paths  # Run a specific unittest module
python tests/test_device_detection.py        # Run a test file directly
python -m pytest tests/test_markdown_export.py  # Run pytest-style tests
```
- Some test files are executable scripts (`chmod +x`) with `if __name__ == "__main__":` blocks:
  - `tests/test_device_detection.py`
  - `tests/test_parakeet_basic.py`
  - `tests/test_parakeet_integration.py`
  - `tests/test_gpu_integration.py`
  - `tests/test_steps.py`
  - `tests/create_test_chunk.py`
  - `tests/quick_benchmark.py`
  - `tests/benchmark_gpu_performance.py`
  - `tests/benchmark_whisper_vs_parakeet.py`

**Coverage:**
- No coverage tool is configured (no `.coveragerc`, no `coverage` in requirements).
- No coverage threshold or enforcement.

## Structure

**Location:**
- All tests live in `tests/` at the repository root.
- Test files are co-located with benchmark scripts and test-data generators in the same `tests/` directory.

**Naming:**
- Test files: `test_*.py` (e.g., `test_service_paths.py`, `test_markdown_export.py`, `test_device_detection.py`).
- Non-test helper/benchmark scripts in `tests/`: `benchmark_gpu_performance.py`, `benchmark_whisper_vs_parakeet.py`, `quick_benchmark.py`, `create_test_chunk.py`, `test_steps.py`.

**Structure:**
```
tests/
├── test_service_paths.py       # unittest - service path validation
├── test_device_detection.py    # unittest - device detection + speaker turn merger
├── test_markdown_export.py     # pytest-style - markdown export
├── test_youtube_download.py    # pytest-style - youtube download
├── test_parakeet_basic.py      # unittest - Parakeet model loading
├── test_parakeet_integration.py # unittest - full pipeline smoke test
├── test_gpu_integration.py     # unittest - GPU diarization integration
├── test_diarization_only.py    # script - diarization only
├── test_minimal_pipeline.py    # script - pipeline loading
├── test_steps.py               # script - step-by-step testing
├── benchmark_gpu_performance.py # script - GPU benchmark
├── benchmark_whisper_vs_parakeet.py # script - model comparison
├── quick_benchmark.py          # script - quick CPU vs MPS
├── create_test_chunk.py        # script - test audio extraction
└── __pycache__/
```

**Test categories:**
1. **Unit tests** (pure logic, no external dependencies):
   - `test_service_paths.py` — `validate_audio_path()` path traversal protection.
   - `test_markdown_export.py` — `segments_to_markdown()` and `merge_consecutive_same_speaker()`.
   - `test_youtube_download.py` — `is_youtube_url()` and `download_youtube_audio()` (with mocked yt-dlp).
   - `test_device_detection.py` — `get_device()`, `normalize_transcription_language()`, `_SpeakerTurnMerger` (with mocked torch).

2. **Integration tests** (require models, audio files, GPU):
   - `test_parakeet_basic.py` — Parakeet model loading and transcription.
   - `test_parakeet_integration.py` — full pipeline smoke test.
   - `test_gpu_integration.py` — GPU diarization with actual audio.
   - `test_diarization_only.py` — diarization pipeline only.
   - `test_minimal_pipeline.py` — pipeline loading verification.

3. **Benchmark scripts** (performance comparison, not assertions):
   - `benchmark_gpu_performance.py`, `benchmark_whisper_vs_parakeet.py`, `quick_benchmark.py`.

4. **Utility scripts** (test data generation):
   - `create_test_chunk.py` — extracts a 2-minute test audio chunk.

## Test Structure

**Suite organization (unittest):**
```python
class TestValidateAudioPath(unittest.TestCase):
    def test_accepts_file_under_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d).resolve()
            old = jobs.settings.data_dir
            jobs.settings.data_dir = base
            try:
                f = base / "clip.wav"
                f.write_bytes(b"fake")
                resolved = jobs.validate_audio_path(str(f))
                self.assertEqual(Path(resolved), f.resolve())
            finally:
                jobs.settings.data_dir = old
```

**Suite organization (pytest-style):**
```python
def test_markdown_includes_speaker_and_time():
    segments = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "Hello."},
        {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01", "text": "Hi there."},
    ]

    md = segments_to_markdown(
        segments,
        title="Test",
        source_url="https://youtu.be/abc",
    )

    assert "# Test" in md
    assert "https://youtu.be/abc" in md
    assert "SPEAKER_00" in md and "Hello." in md
    assert "[00:00 - 00:02]" in md
```

**Patterns:**
- **Setup/teardown:** `unittest.TestCase.setUpClass` for class-level fixtures (e.g., loading audio in `test_gpu_integration.py`). No `setUp`/`tearDown` methods observed.
- **Temporary directories:** `tempfile.TemporaryDirectory()` as context manager for filesystem-based tests.
- **Path manipulation:** `sys.path.insert(0, ...)` to add repo root to `PYTHONPATH` for imports (e.g., `tests/test_device_detection.py:12`, `tests/test_markdown_export.py:4`, `tests/test_youtube_download.py:5`).
- **Environment setup:** `load_dotenv()` called at module level before importing application code (e.g., `test_device_detection.py:16`, `test_parakeet_basic.py:7`, `test_gpu_integration.py:11`).
- **Environment variables:** `PYANNOTE_METRICS_ENABLED=0` set before importing pyannote (all integration tests). `PYTORCH_ENABLE_MPS_FALLBACK=1` set in Parakeet tests.
- **Module loading:** `importlib.util.spec_from_file_location` used to load `transcribe_simple.py` as a module in `test_device_detection.py` (lines 20–23).
- **Direct imports:** `from transcribe_simple import main` in `test_parakeet_integration.py:16`.

## Mocking

**Framework:**
- `unittest.mock.patch` and `unittest.mock.MagicMock` (from `unittest.mock`).

**Patterns:**
```python
# Patching torch functions
with patch('torch.cuda.is_available', return_value=True):
    with patch('torch.backends.mps.is_available', return_value=True):
        device = get_device()
        self.assertEqual(device, "cuda", "CUDA should be preferred over MPS")

# Patching importlib.util.find_spec
with patch("importlib.util.find_spec", return_value=object()):
    backend = transcribe_simple._select_whisper_backend("cuda")
    self.assertEqual(backend, "faster-whisper")

# Patching object methods
with patch.object(transcribe_simple, "get_device", return_value="cpu"), \
     patch.object(transcribe_simple, "_select_whisper_backend", return_value="faster-whisper"), \
     patch.object(transcribe_simple, "_load_faster_whisper_model", return_value=(object(), "tiny", "int8")), \
     patch.object(transcribe_simple, "_iter_audio_chunks", return_value=fake_chunks), \
     patch.object(transcribe_simple, "_write_temp_wav", return_value="/tmp/fake.wav"), \
     patch.object(transcribe_simple, "_transcribe_chunk_with_faster_whisper", return_value=fake_result_segments), \
     patch.object(transcribe_simple.os, "unlink"):
    transcribe_simple._transcribe_with_whisper(...)

# Patching a class for dependency injection
class DummyYDL:
    def __init__(self, _opts):
        self._info = {"id": "abc", "title": "Title"}
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def extract_info(self, _url, download=True):
        assert download is True
        return self._info
    def prepare_filename(self, _info):
        return str(audio_file)

with patch("gui.youtube_download._YoutubeDL", DummyYDL):
    result = download_youtube_audio("https://youtu.be/abc", str(tmp_path))
```

**What to mock:**
- External services: `torch.cuda.is_available`, `torch.backends.mps.is_available`, `importlib.util.find_spec`.
- Model loading: `_load_faster_whisper_model`, `_load_whisper_model`, `_select_whisper_backend`.
- File I/O: `_write_temp_wav`, `os.unlink`, `tempfile.NamedTemporaryFile`.
- Third-party libraries: `yt_dlp.YoutubeDL` (patched as `gui.youtube_download._YoutubeDL`).
- Audio processing: `_iter_audio_chunks`, `_transcribe_chunk_with_faster_whisper`.

**What NOT to mock:**
- Pure logic functions: `merge_consecutive_same_speaker`, `segments_to_markdown`, `is_youtube_url`, `validate_audio_path`, `normalize_transcription_language`.
- These are tested directly with real inputs.

## Fixtures and Factories

**Test data:**
- Inline data structures (dicts/lists) for unit tests (e.g., segment dicts in `test_markdown_export.py`).
- `tempfile.TemporaryDirectory()` for filesystem tests.
- `tmp_path` (pytest fixture) in `test_youtube_download.py:16` for temporary output directories.
- Actual audio files (`test_chunk_2min.m4a`) for integration tests — generated by `tests/create_test_chunk.py`.

**Location:**
- No dedicated `fixtures/` or `conftest.py` directory.
- Test data is inline or generated on-the-fly.
- Audio test files are referenced relative to repo root (e.g., `"test_chunk_2min.m4a"`, `"tests/test_chunk_2min.m4a"`).

## Coverage

**Requirements:**
- No coverage requirements enforced. No `coverage.py` configuration exists.

**View coverage:**
```bash
# Not configured; would require:
# pip install coverage
# coverage run -m pytest tests/
# coverage report
```

## Test Types

**Unit Tests:**
- Pure logic tests with no external dependencies or model loading.
- Located in `test_service_paths.py`, `test_markdown_export.py`, `test_youtube_download.py`, and parts of `test_device_detection.py`.
- Use mocking for torch functions and external libraries.
- Run fast (seconds).

**Integration Tests:**
- Require Hugging Face models, audio files, and optionally GPU.
- Located in `test_parakeet_basic.py`, `test_parakeet_integration.py`, `test_gpu_integration.py`, `test_diarization_only.py`, `test_minimal_pipeline.py`.
- Require `HUGGINGFACE_TOKEN` in `.env` (checked at runtime).
- Require `test_chunk_2min.m4a` audio file (generated by `create_test_chunk.py`).
- Some tests are skipped if prerequisites are missing (e.g., `test_parakeet_basic.py:34` skips if MPS not available; `test_gpu_integration.py:28` raises if token not found).

**E2E Tests:**
- Not explicitly labeled as E2E. The closest equivalent is `test_parakeet_integration.py` which calls `transcribe_simple.main()` end-to-end.
- No Playwright, Selenium, or HTTP-level testing framework is used.

## Common Patterns

**Async testing:**
- Not applicable. The FastAPI service (`service/api.py`) has no tests. All tests are synchronous.

**Error testing:**
```python
# Testing for expected exceptions
with self.assertRaises(ValueError):
    jobs.validate_audio_path(str(missing))

# Testing for ValueError with specific message
with self.assertRaises(ValueError):
    normalize("French")

# Testing for FileNotFoundError
if not os.path.exists(test_audio):
    self.skipTest(f"Test audio not found: {test_audio}")

# Testing for skip conditions
if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    self.skipTest("MPS not available")
```

**Settings mutation for testing:**
```python
# Temporarily override settings.data_dir for path validation tests
old = jobs.settings.data_dir
jobs.settings.data_dir = base
try:
    # ... test code ...
finally:
    jobs.settings.data_dir = old
```

**Environment-based test mode:**
```python
# test_parakeet_integration.py uses environment variables to configure the pipeline
os.environ['TEST_MODE'] = 'true'
os.environ['NUM_SPEAKERS'] = '2'
sys.argv = ['transcribe_simple.py', 'tests/test_chunk_2min.m4a']
main()
```

---

*Testing analysis: 2026-07-30*
