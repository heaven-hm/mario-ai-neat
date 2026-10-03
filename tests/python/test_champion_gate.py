from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import champion_gate


def make_report(victories: int, episodes: int, seed: int = 2026, world: int = 1) -> dict:
    records = [{"episode": index + 1,
                "reason": "victory" if index < victories else "death",
                "max_x": 3161 if index < victories else 2019,
                "terminal_x": 3161 if index < victories else 2019,
                "action_decisions": 100,
                "elapsed_seconds": 10.0}
               for index in range(episodes)]
    return {
        "algorithm": "Ape-X Rainbow",
        "rom_sha256": "rom",
        "fceux_sha256": "fceux",
        "world": world,
        "level": 1,
        "action_repeat_frames": 12,
        "start_protocol": "clean-start-v1",
        "evaluation_mode": "greedy_no_learning",
        "evaluation_seed": seed,
        "episodes_requested": episodes,
        "episodes_finished": episodes,
        "victories": victories,
        "completion_rate": victories / episodes if episodes else 0.0,
        "checkpoint_sha256": "candidate-sha",
        "episodes": records,
    }


class WilsonIntervalTests(unittest.TestCase):
    def test_recorded_champion_baseline(self) -> None:
        low, high = champion_gate.wilson_interval(15, 77)
        self.assertAlmostEqual(low, 0.122, places=3)
        self.assertAlmostEqual(high, 0.297, places=3)

    def test_zero_clears_upper_bound_not_zero(self) -> None:
        low, high = champion_gate.wilson_interval(0, 20)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_rejects_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            champion_gate.wilson_interval(5, 0)
        with self.assertRaises(ValueError):
            champion_gate.wilson_interval(21, 20)


class DecisionTests(unittest.TestCase):
    def test_decisive_candidate_promotes(self) -> None:
        candidate = champion_gate.summarize([make_report(18, 20)])
        incumbent = champion_gate.summarize([make_report(1, 20)])
        self.assertEqual(champion_gate.decide(candidate, incumbent)["decision"], "promote")

    def test_strictly_worse_candidate_archives(self) -> None:
        candidate = champion_gate.summarize([make_report(0, 20)])
        incumbent = champion_gate.summarize([make_report(15, 20)])
        self.assertEqual(champion_gate.decide(candidate, incumbent)["decision"], "archive")

    def test_overlap_is_inconclusive(self) -> None:
        candidate = champion_gate.summarize([make_report(4, 20)])
        incumbent = champion_gate.summarize([make_report(3, 20)])
        self.assertEqual(champion_gate.decide(candidate, incumbent)["decision"], "inconclusive")

    def test_empty_evidence_is_inconclusive(self) -> None:
        candidate = champion_gate.summarize([make_report(0, 0)])
        incumbent = champion_gate.summarize([make_report(1, 20)])
        self.assertEqual(champion_gate.decide(candidate, incumbent)["decision"], "inconclusive")


class ComparabilityTests(unittest.TestCase):
    def test_rejects_mismatched_conditions(self) -> None:
        problems = champion_gate.check_comparable(
            [make_report(2, 10)], [make_report(2, 10, seed=7)])
        self.assertTrue(problems)

    def test_rejects_victory_count_mismatch(self) -> None:
        report = make_report(2, 10)
        report["victories"] = 5
        problems = champion_gate.check_comparable([report], [make_report(2, 10)])
        self.assertTrue(any("victory count" in problem for problem in problems))

    def test_accepts_identical_conditions(self) -> None:
        problems = champion_gate.check_comparable(
            [make_report(2, 10)], [make_report(1, 10)])
        self.assertEqual(problems, [])

    def test_rejects_unattributed_sweeps(self) -> None:
        report = make_report(2, 10)
        report["checkpoint_sha256"] = None
        problems = champion_gate.check_comparable([report], [make_report(2, 10)])
        self.assertTrue(any("unattributed" in problem for problem in problems))

    def test_rejects_evaluator_mismatch(self) -> None:
        candidate = make_report(2, 10)
        candidate["evaluator_sha256"] = "evaluator-a"
        incumbent = make_report(2, 10)
        incumbent["evaluator_sha256"] = "evaluator-b"
        problems = champion_gate.check_comparable([candidate], [incumbent])
        self.assertTrue(any("evaluator" in problem for problem in problems))


class CliTests(unittest.TestCase):
    def test_recorded_champion_gate_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "results.json"
            candidate_path.write_text(json.dumps(make_report(8, 20)), encoding="utf-8")
            import contextlib
            import io
            output = io.StringIO()
            argv = sys.argv
            sys.argv = ["champion_gate.py", "--candidate", str(candidate_path),
                        "--use-recorded-champion", "--json"]
            try:
                with contextlib.redirect_stdout(output):
                    exit_status = champion_gate.main()
            finally:
                sys.argv = argv
            verdict = json.loads(output.getvalue())
            self.assertEqual(verdict["decision"], "inconclusive")
            self.assertEqual(exit_status, 0)
            self.assertEqual(verdict["incumbent"]["victories"], 15)
            self.assertEqual(verdict["incumbent"]["episodes"], 77)


if __name__ == "__main__":
    unittest.main()
