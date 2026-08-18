"""Unit tests for VibeVoice transcript formatting (no model required)."""

from unittest.mock import patch

from vibevoice_poc import (
    choose_speaker_sample_segments,
    format_segments,
    format_transcript,
    is_video_path,
    merge_consecutive_segments,
    parse_model_output,
    prepare_media_for_transcription,
    segments_to_markdown,
    speaker_samples_to_html,
)


def test_format_segments_with_speaker_and_timestamps():
    segments = [
        {
            "start_time": 12.0,
            "end_time": 18.5,
            "speaker_id": "1",
            "text": "Hello everyone.",
        },
        {
            "start_time": 18.5,
            "end_time": 24.0,
            "speaker_id": "2",
            "text": "Thanks for joining.",
        },
    ]
    text = format_segments(segments)
    assert "**Speaker 1**" in text
    assert "Hello everyone." in text
    assert "**Speaker 2**" in text
    assert "Thanks for joining." in text


def test_format_transcript_parses_raw_json_when_segments_missing():
    result = {
        "segments": [],
        "raw_text": 'assistant [{"Start":0,"End":2.5,"Speaker":0,"Content":"Hallo."}]',
    }
    text = format_transcript(result)
    assert "**Speaker 0**" in text
    assert "Hallo." in text
    assert "assistant" not in text
    assert '"Start"' not in text


def test_parse_model_output_recovers_truncated_json():
    raw = (
        'assistant [{"Start":0,"End":1,"Speaker":0,"Content":"Eins."},'
        '{"Start":1,"End":2,"Speaker":1,"Content":"Zwei."},{"Start":2'
    )
    segments = parse_model_output(raw)
    assert len(segments) == 2
    assert segments[0]["Content"] == "Eins."
    assert segments[1]["Content"] == "Zwei."


def test_format_segments_skips_non_speech_markers():
    segments = [
        {"Start": 0.0, "End": 8.0, "Speaker": "Unknown", "Content": "[Silence]"},
        {"Start": 8.0, "End": 10.0, "Speaker": 0, "Content": "Hallo Norbert."},
        {"Start": 10.0, "End": 12.0, "Speaker": "Unknown", "Content": "[Speech]"},
        {"Start": 12.0, "End": 14.0, "Speaker": 1, "Content": "[Music]"},
        {"Start": 14.0, "End": 16.0, "Speaker": "Unknown", "Content": "[Human Sounds]"},
    ]
    text = format_segments(segments)
    assert "Silence" not in text
    assert "Speech" not in text
    assert "Music" not in text
    assert "Human Sounds" not in text
    assert "Hallo Norbert." in text


def test_merge_consecutive_same_speaker():
    segments = [
        {"Start": 0.0, "End": 6.91, "Speaker": 0, "Content": "Erster Satz."},
        {"Start": 6.91, "End": 10.0, "Speaker": 0, "Content": "Zweiter Satz."},
        {"Start": 10.0, "End": 15.0, "Speaker": 1, "Content": "Antwort."},
    ]
    merged = merge_consecutive_segments(segments)
    assert len(merged) == 2
    assert merged[0]["text"] == "Erster Satz. Zweiter Satz."
    assert merged[1]["text"] == "Antwort."


def test_format_segments_with_custom_speaker_names():
    segments = [
        {"start_time": 0.0, "end_time": 2.0, "speaker_id": 0, "text": "Hallo."},
        {"start_time": 2.0, "end_time": 4.0, "speaker_id": 1, "text": "Hi."},
    ]
    text = format_segments(segments, speaker_names={"0": "Thomas", "1": "Mark"})
    assert "**Thomas**" in text
    assert "**Mark**" in text
    assert "Speaker 0" not in text


def test_segments_to_markdown_uses_custom_names():
    segments = [
        {"Start": 0.0, "End": 5.0, "Speaker": 0, "Content": "Opening."},
    ]
    md = segments_to_markdown(segments, title="Meeting", speaker_names={"0": "Roland"})
    assert "# Meeting" in md
    assert "**[00:00.00 - 00:05.00] Roland**" in md
    assert "Opening." in md


def test_format_segments_lines_style():
    segments = [
        {"start_time": 1.0, "end_time": 2.0, "speaker_id": 0, "text": "Hi."},
    ]
    text = format_segments(segments, style="lines")
    assert text == "[00:01 - 00:02] Speaker 0: Hi."


def test_choose_speaker_sample_prefers_at_least_three_seconds():
    segments = [
        {"Start": 0.0, "End": 1.0, "Speaker": 2, "Content": "Ja."},
        {"Start": 1.0, "End": 2.0, "Speaker": 0, "Content": "Zwischenruf."},
        {"Start": 2.0, "End": 10.0, "Speaker": 2, "Content": "Längere Aussage."},
        {"Start": 10.0, "End": 11.2, "Speaker": 3, "Content": "Kurz."},
    ]
    chosen = choose_speaker_sample_segments(segments, min_seconds=3.0)
    assert chosen["2"]["text"] == "Längere Aussage."
    assert chosen["3"]["text"] == "Kurz."


def test_speaker_samples_html_skips_missing_files():
    html = speaker_samples_to_html({"0": "/tmp/does-not-exist.wav"})
    assert html == "<div style=\"display:flex;flex-direction:column;gap:12px;\"></div>"


def test_is_video_path():
    assert is_video_path("/tmp/meeting.mp4")
    assert is_video_path("clip.MKV")
    assert not is_video_path("/tmp/audio.wav")


def test_prepare_media_for_transcription_prefers_video():
    with patch("vibevoice_poc.extract_audio_from_video", return_value="/tmp/meeting.wav"):
        audio_path, stem = prepare_media_for_transcription("/tmp/audio.wav", "/tmp/meeting.mp4")
    assert stem == "meeting"
    assert audio_path == "/tmp/meeting.wav"


def test_prepare_media_for_transcription_audio_only():
    audio_path, stem = prepare_media_for_transcription("/tmp/audio.wav", None)
    assert audio_path == "/tmp/audio.wav"
    assert stem == "audio"
