from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.evaluate import write_episode_csv


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


if __name__ == "__main__":
    unittest.main()
