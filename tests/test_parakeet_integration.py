#!/usr/bin/env python3
"""
Integration tests for Parakeet with speaker diarization pipeline.
"""
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import unittest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the main function from modified script
from transcribe_simple import main


class TestParakeetIntegration(unittest.TestCase):
    """Integration tests for Parakeet with speaker diarization."""

    def test_full_pipeline_with_parakeet(self):
        """Test complete pipeline with Parakeet on 2-minute chunk."""
        print("\n[TEST] Running full pipeline with Parakeet...")

        # Set test mode
        os.environ['TEST_MODE'] = 'true'
        os.environ['NUM_SPEAKERS'] = '2'

        # Run main - should complete without errors
        # This is a smoke test
        try:
            sys.argv = ['transcribe_simple.py', 'tests/test_chunk_2min.m4a']
            main()
            success = True
            print("✓ Pipeline completed successfully")
        except Exception as e:
            success = False
            self.fail(f"Pipeline failed: {e}")

        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main(verbosity=2)
