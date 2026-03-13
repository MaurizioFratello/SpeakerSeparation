#!/usr/bin/env python3
"""
Extract a 2-minute test chunk from audio file for faster testing.
"""
import sys
import subprocess

def create_test_chunk(input_file, output_file="test_chunk_2min.m4a", duration=120):
    """
    Extract first N seconds from audio file.

    Args:
        input_file: Path to input audio file
        output_file: Path to output file
        duration: Duration in seconds (default: 120 = 2 minutes)
    """
    print(f"Extracting first {duration} seconds from {input_file}...")

    cmd = [
        'ffmpeg', '-i', input_file,
        '-t', str(duration),
        '-c', 'copy',
        '-y',  # Overwrite without asking
        output_file
    ]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        print(f"✓ Created {duration}s test chunk: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to create test chunk")
        print(e.stderr.decode())
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "Music Company Media Productions 10.m4a"

    create_test_chunk(input_file)
