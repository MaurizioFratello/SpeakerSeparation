from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.markdown_export import segments_to_markdown, merge_consecutive_same_speaker


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


def test_merge_consecutive_same_speaker_only():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello"},
        {"start": 1.1, "end": 2.0, "speaker": "SPEAKER_00", "text": "world"},
        {"start": 2.1, "end": 3.0, "speaker": "SPEAKER_01", "text": "Other"},
        {"start": 3.1, "end": 4.0, "speaker": "SPEAKER_01", "text": "speaker"},
        {"start": 4.1, "end": 5.0, "speaker": "SPEAKER_00", "text": "Back"},
    ]

    merged = merge_consecutive_same_speaker(segments)

    assert len(merged) == 3
    assert merged[0]["speaker"] == "SPEAKER_00"
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 2.0
    assert merged[0]["text"] == "Hello world"
    assert merged[1]["speaker"] == "SPEAKER_01"
    assert merged[1]["text"] == "Other speaker"
    assert merged[2]["speaker"] == "SPEAKER_00"
    assert merged[2]["text"] == "Back"
