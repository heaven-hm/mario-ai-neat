from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.evaluate import write_episode_csv
from mario_ai_fceux.apex_eval import build_benchmark_report, write_benchmark_report
from mario_ai_fceux.benchmark import validate_reports


class EvaluationOutputTests(unittest.TestCase):
    def test_csv_persists_full_episode_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            episode = {"episode": 1, "reason": "death", "max_x": 128,
                       "terminal_x": 128, "action_decisions": 42,
                       "elapsed_seconds": 8.5}
            write_episode_csv(path, [episode])
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["action_decisions"], "42")
            self.assertEqual(rows[0]["elapsed_seconds"], "8.5")

    def test_apex_evaluation_exports_strict_benchmark_json_and_csv(self) -> None:
        run_metadata = {
            "algorithm": "Ape-X Rainbow", "eval_world": 1, "seed": 7,
            "rom_sha256": "rom-hash", "fceux_sha256": "fceux-hash",
            "action_repeat_frames": 12, "start_protocol": "fixed-start-v1",
            "eval_episodes": 2, "source_revision": "abc123",
        }
        episodes = [
            {"reason": "death", "max_x": 80, "terminal_x": 72,
             "action_decisions": 31, "elapsed_seconds": 3.2},
            {"reason": "victory", "max_x": 120, "terminal_x": 120,
             "action_decisions": 45, "elapsed_seconds": 4.1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = build_benchmark_report(episodes, run_metadata, 3)
            validate_reports([("test-policy", report), ("comparison-policy", dict(report))])
            json_path = write_benchmark_report(Path(directory), report)
            stored = json.loads(json_path.read_text(encoding="utf-8"))
            csv_path = json_path.with_name("episodes.csv")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(stored["evaluation_mode"], "greedy_no_learning")
        self.assertEqual(stored["evaluation_seed"], 7)
        self.assertEqual(stored["victories"], 1)
        self.assertEqual(stored["completion_rate"], 0.5)
        self.assertEqual(stored["source_revision"], "abc123")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["reason"], "victory")


if __name__ == "__main__":
    unittest.main()
