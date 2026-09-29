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
            "algorithm": "Rainbow",
            "rom_sha256": "rom",
            "fceux_sha256": "fceux",
            "world": 1,
            "level": 1,
            "action_repeat_frames": 12,
            "start_protocol": "clean-start-v1",
            "evaluation_mode": "greedy_no_learning",
            "evaluation_seed": 2026,
            "episodes_requested": 10,
            "episodes_finished": 10,
            "episodes": [{"episode": index, "reason": "death", "max_x": 0,
                           "action_decisions": 12, "elapsed_seconds": 5.0}
                         for index in range(10)],
            "victories": 0,
            "completion_rate": 0.0,
        }

    def test_accepts_complete_reports_with_identical_conditions(self) -> None:
        validate_reports([("NEAT", dict(self.report)), ("Rainbow", dict(self.report))])

    def test_rejects_different_worlds(self) -> None:
        other = {**self.report, "world": 2}
        with self.assertRaisesRegex(ValueError, "conditions differ"):
            validate_reports([("NEAT", self.report), ("Rainbow", other)])

    def test_rejects_different_levels(self) -> None:
        other = {**self.report, "level": 2}
        with self.assertRaisesRegex(ValueError, "conditions differ"):
            validate_reports([("NEAT", self.report), ("Rainbow", other)])

    def test_rejects_incomplete_evaluation(self) -> None:
        incomplete = {**self.report, "episodes_finished": 9}
        with self.assertRaisesRegex(ValueError, "only 9 of 10"):
            validate_reports([("NEAT", self.report), ("Rainbow", incomplete)])

    def test_rejects_training_metrics_mislabeled_as_greedy_evaluation(self) -> None:
        training = {**self.report, "evaluation_mode": "epsilon_greedy_training"}
        with self.assertRaisesRegex(ValueError, "disable learning and exploration"):
            validate_reports([("NEAT", self.report), ("Rainbow", training)])

    def test_rejects_inconsistent_victory_metrics(self) -> None:
        inconsistent = {**self.report, "victories": 1}
        with self.assertRaisesRegex(ValueError, "victory count"):
            validate_reports([("NEAT", self.report), ("Rainbow", inconsistent)])

    def test_rejects_nan_episode_metrics(self) -> None:
        invalid = {**self.report, "episodes": [dict(item) for item in self.report["episodes"]]}
        invalid["episodes"][0]["elapsed_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "malformed episode records"):
            validate_reports([("NEAT", self.report), ("Rainbow", invalid)])

    def test_rejects_negative_or_boolean_action_counts(self) -> None:
        invalid = {**self.report, "episodes": [dict(item) for item in self.report["episodes"]]}
        invalid["episodes"][0]["action_decisions"] = True
        with self.assertRaisesRegex(ValueError, "malformed episode records"):
            validate_reports([("NEAT", self.report), ("Rainbow", invalid)])

    def test_rejects_non_object_report(self) -> None:
        with self.assertRaisesRegex(ValueError, "reports must be JSON objects"):
            validate_reports([("NEAT", self.report), ("Rainbow", [])])

    def test_rejects_missing_algorithm_name(self) -> None:
        invalid = {**self.report, "algorithm": "  "}
        with self.assertRaisesRegex(ValueError, "missing policy algorithm label"):
            validate_reports([("NEAT", self.report), ("Rainbow", invalid)])


if __name__ == "__main__":
    unittest.main()
