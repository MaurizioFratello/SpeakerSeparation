"""
Markdown export utilities for diarized transcript segments.
"""

from typing import Any, Dict, List, Optional


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def merge_consecutive_same_speaker(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge contiguous segments when the speaker remains the same.
    """
    if not segments:
        return []

    merged: List[Dict[str, Any]] = []
    current = {
        "start": float(segments[0].get("start", 0.0)),
        "end": float(segments[0].get("end", 0.0)),
        "speaker": str(segments[0].get("speaker", "UNKNOWN")),
        "text": str(segments[0].get("text", "")).strip(),
    }

    for seg in segments[1:]:
        speaker = str(seg.get("speaker", "UNKNOWN"))
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()

        if speaker == current["speaker"]:
            current["end"] = end
            if text:
                current["text"] = f"{current['text']} {text}".strip()
        else:
            merged.append(current)
            current = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }

    merged.append(current)
    return merged


def segments_to_markdown(
    segments: List[Dict[str, Any]],
    title: Optional[str] = None,
    source_url: Optional[str] = None,
) -> str:
    """
    Convert diarized segments into markdown text.
    """
    # Defensive merge so markdown export is always grouped by continuous speaker turns.
    segments = merge_consecutive_same_speaker(segments)
    heading = title.strip() if title and title.strip() else "Transcript"
    lines = [f"# {heading}", ""]

    if source_url:
        lines.append(f"Source: {source_url}")
        lines.append("")

    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        speaker = str(seg.get("speaker", "UNKNOWN"))
        text = str(seg.get("text", "")).strip()

        lines.append(f"**[{_format_timestamp(start)} - {_format_timestamp(end)}] {speaker}**")
        lines.append(text if text else "_(no text)_")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
