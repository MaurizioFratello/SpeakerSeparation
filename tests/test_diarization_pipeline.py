"""community-1 defaults and DiarizeOutput annotation extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transcribe_simple import (
    DEFAULT_DIARIZATION_MODEL_ID,
    FALLBACK_DIARIZATION_MODEL_ID,
    extract_diarization_annotation,
    load_diarization_pipeline,
    prepare_diarization_pipeline,
    transcribe_audio,
)


class TestCommunity1Defaults(unittest.TestCase):
    def test_default_model_is_community_1(self) -> None:
        self.assertEqual(
            DEFAULT_DIARIZATION_MODEL_ID,
            "pyannote/speaker-diarization-community-1",
        )

    def test_legacy_3_1_is_fallback_only(self) -> None:
        self.assertEqual(
            FALLBACK_DIARIZATION_MODEL_ID,
            "pyannote/speaker-diarization-3.1",
        )


class TestExtractDiarizationAnnotation(unittest.TestCase):
    def test_prefers_exclusive_speaker_diarization_for_asr(self) -> None:
        exclusive = object()
        overlapping = object()
        output = SimpleNamespace(
            exclusive_speaker_diarization=exclusive,
            speaker_diarization=overlapping,
        )
        self.assertIs(extract_diarization_annotation(output), exclusive)

    def test_falls_back_to_speaker_diarization(self) -> None:
        overlapping = object()
        output = SimpleNamespace(speaker_diarization=overlapping)
        self.assertIs(extract_diarization_annotation(output), overlapping)

    def test_legacy_annotation_passthrough(self) -> None:
        annotation = object()
        self.assertIs(extract_diarization_annotation(annotation), annotation)


class TestPrepareDiarizationPipeline(unittest.TestCase):
    def test_moves_pipeline_to_selected_cuda_device(self) -> None:
        pipeline = MagicMock()
        with patch("transcribe_simple.torch.cuda.set_device") as set_device:
            result = prepare_diarization_pipeline(pipeline, "cuda:1")

        self.assertIs(result, pipeline)
        set_device.assert_called_once_with(1)
        pipeline.to.assert_called_once_with(__import__("torch").device("cuda:1"))


class TestPreloadedPipelineReuse(unittest.TestCase):
    def test_preloaded_pipeline_is_not_moved_for_each_job(self) -> None:
        pipeline = MagicMock()
        with patch("transcribe_simple.prepare_diarization_pipeline") as prepare:
            result = transcribe_audio(
                "unused.wav",
                pipeline=pipeline,
                check_interrupt=lambda: True,
            )

        self.assertEqual(result, [])
        prepare.assert_not_called()


class TestLoadDiarizationPipeline(unittest.TestCase):
    def test_tries_community_1_first(self) -> None:
        loaded = MagicMock(name="pipeline")
        with patch("transcribe_simple.Pipeline.from_pretrained", return_value=loaded) as from_pretrained:
            result = load_diarization_pipeline("hf_token")
        self.assertIs(result, loaded)
        from_pretrained.assert_called_once()
        self.assertEqual(
            from_pretrained.call_args.args[0],
            "pyannote/speaker-diarization-community-1",
        )


if __name__ == "__main__":
    unittest.main()
