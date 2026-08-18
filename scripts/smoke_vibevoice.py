#!/usr/bin/env python3
import sys
import traceback

from vibevoice_poc import create_transcriber, format_transcript

audio = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_vv.wav"
print("loading...")
t = create_transcriber(device="cuda")
print("transcribing", audio)
try:
    result = t.transcribe(audio, max_new_tokens=512)
    print("done in", result.get("generation_time"))
    print(format_transcript(result)[:1000])
    print("raw:", (result.get("raw_text") or "")[:500])
except Exception:
    traceback.print_exc()
    raise
