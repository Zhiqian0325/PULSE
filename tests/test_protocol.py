"""Guard the review-sensitive data and measurement rules."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pulse.data import load_manifest
from pulse.metrics import summarize_lvef
from pulse.sampling import refinement_clip, tiled_clips


class ProtocolTests(unittest.TestCase):
    def test_sampling(self):
        self.assertEqual(tiled_clips(range(14)), [tuple(range(8)), tuple(range(6, 14))])
        self.assertEqual(refinement_clip(12, 11, 0, 42), tuple(range(12)))
        self.assertEqual(refinement_clip(40, 30, 2, 42)[::19], (2, 30))
        self.assertEqual(len(set(refinement_clip(40, 30, 2, 42))), 20)
        self.assertLessEqual(set(range(11, 19)), set(refinement_clip(40, 11, 18, 42)))

    def test_cross_split_patient_is_rejected_without_reading_video_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "echonet",
                        "videos": [
                            {"video_id": "a", "patient_id": "same", "split": "train"},
                            {"video_id": "b", "patient_id": "same", "split": "val"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                load_manifest(path, root, "echonet", "test")

    def test_invalid_lvef_cannot_be_silently_removed(self):
        result = summarize_lvef(
            [
                {"reference": 50, "prediction": 45},
                {"reference": 40, "prediction": None},
            ]
        )
        self.assertNotIn("metrics", result)
        self.assertEqual(result["failed_N"], 1)
        self.assertEqual(result["valid_subset_diagnostic"]["MAE"], 5)


if __name__ == "__main__":
    unittest.main()
