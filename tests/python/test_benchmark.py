from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.benchmark import validate_reports


class BenchmarkValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = {
            "rom_sha256": "rom",
            "fceux_sha256": "fceux",
            "world": 1,
            "action_repeat_frames": 12,
            "start_protocol": "clean-start-v1",
            "episodes_requested": 10,
            "episodes_finished": 10,
        }

    def test_accepts_complete_reports_with_identical_conditions(self) -> None:
        validate_reports([("NEAT", dict(self.report)), ("Rainbow", dict(self.report))])

    def test_rejects_different_worlds(self) -> None:
        other = {**self.report, "world": 2}
        with self.assertRaisesRegex(ValueError, "conditions differ"):
            validate_reports([("NEAT", self.report), ("Rainbow", other)])

    def test_rejects_incomplete_evaluation(self) -> None:
        incomplete = {**self.report, "episodes_finished": 9}
        with self.assertRaisesRegex(ValueError, "only 9 of 10"):
            validate_reports([("NEAT", self.report), ("Rainbow", incomplete)])


if __name__ == "__main__":
    unittest.main()
