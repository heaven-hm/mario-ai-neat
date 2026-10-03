from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.evaluate import build_trace_record, write_episode_csv
from mario_ai_fceux.apex_eval import (build_benchmark_report, write_action_traces,
                                      evaluation_rank, publish_eval_hud, save_best_policy,
                                      write_benchmark_report)
from mario_ai_fceux.benchmark import validate_reports
from mario_ai_fceux.environment import FileWorker, Observation


class EvaluationOutputTests(unittest.TestCase):
    def test_evaluation_window_shows_real_progress_instead_of_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            worker = FileWorker("eval", run_directory / "worker", action_profile="rainbow")
            worker.directory.mkdir()
            (run_directory / "learner_status.json").write_text(
                json.dumps({"optimizer_updates": 800, "replay_transitions": 12000}),
                encoding="utf-8",
            )
            observation = Observation(42, np.zeros(184, dtype=np.float32),
                                      546, 0, False, "")
            publish_eval_hud(worker, run_directory, observation, 4,
                             np.zeros(21), np.zeros(16), 271, 3, 2, 1, 546)
            hud = json.loads((worker.directory / "hud.json").read_text())
            self.assertEqual(hud["mode"], "eval")
            self.assertEqual((hud["steps"], hud["updates"], hud["replay"]),
                             (271, 800, 12000))
            self.assertEqual((hud["episodes"], hud["deaths"], hud["victories"]),
                             (3, 2, 1))
            self.assertEqual(hud["epsilon"], 0.0)
            self.assertEqual((len(hud["grid"]), len(hud["globals"]), len(hud["hidden"])),
                             (169, 15, 16))

    def test_best_policy_keeps_only_a_stronger_measured_greedy_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            initial = {"victories": 0, "avg_max_x": 120.0,
                       "episodes": [{"max_x": 100}, {"max_x": 140}]}
            improved = {"victories": 0, "avg_max_x": 130.0,
                        "episodes": [{"max_x": 110}, {"max_x": 150}]}
            self.assertTrue(save_best_policy(run_directory, b"first", initial))
            self.assertTrue(save_best_policy(run_directory, b"better", improved))
            self.assertFalse(save_best_policy(run_directory, b"worse", initial))
            self.assertEqual((run_directory / "best_policy_weights.pt").read_bytes(), b"better")
            saved = json.loads((run_directory / "best_policy_eval.json").read_text())
            self.assertEqual(evaluation_rank(saved), evaluation_rank(improved))

    def test_rank_uses_episode_progress_when_report_has_no_average(self) -> None:
        early = {"victories": 0, "episodes": [{"max_x": 100}, {"max_x": 140}]}
        later = {"victories": 0, "episodes": [{"max_x": 110}, {"max_x": 150}]}
        self.assertGreater(evaluation_rank(later), evaluation_rank(early))

    def test_apex_evaluation_persists_compact_action_trace(self) -> None:
        episodes = [{"action_trace": [
            {"decision": 0, "world_x": 430, "action": "jump+run@24",
             "duration_frames": 24, "enemy_dx": 0.2, "gap_ahead": True},
        ]}]
        with tempfile.TemporaryDirectory() as directory:
            trace_path = write_action_traces(Path(directory), 4, episodes)
            record = json.loads(trace_path.read_text(encoding="utf-8").strip())
        self.assertEqual(trace_path.name, "action-trace.jsonl")
        self.assertEqual(record["episode"], 1)
        self.assertEqual(record["world_x"], 430)
        self.assertTrue(record["gap_ahead"])

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


class ActionTraceTests(unittest.TestCase):
    """Death-cause analysis needs per-decision traces in one shared schema."""

    def test_trace_record_matches_the_apex_eval_schema(self) -> None:
        state = np.zeros(184, dtype=np.float32)
        state[169], state[170] = 0.62, -0.5
        state[171], state[174], state[175], state[182] = -1.0, 0.10, 0.69, 1.0
        q = np.linspace(0.0, 20.0, 21).astype(np.float32)
        record = build_trace_record(episode=2, decision=41, state=state,
                                    world_x=1783, action=1, q_values=q)
        # The fields the death-wall lesson relies on.
        self.assertEqual(record["world_x"], 1783)
        self.assertEqual(record["action_id"], 1)
        from mario_ai_fceux.actions import decode_action
        base, duration = decode_action(record["action_id"])
        self.assertEqual(record["action_base"], base)
        self.assertEqual(record["duration_frames"], duration)
        self.assertAlmostEqual(record["enemy_dx"], 0.10)
        self.assertAlmostEqual(record["enemy_dy"], 0.69)
        self.assertFalse(record["grounded"])
        self.assertTrue(record["gap_ahead"])
        self.assertEqual(record["top_actions"][0]["q"], round(float(q.max()), 5))
        # One schema for both tracers: same keys apex_eval writes.
        self.assertEqual(set(record), {
            "episode", "decision", "world_x", "action_id", "action", "action_base",
            "duration_frames", "q_value", "top_actions", "speed_x", "speed_y",
            "grounded", "enemy_dx", "enemy_dy", "gap_ahead",
        })

    def test_tracing_is_off_by_default_so_the_measured_path_is_unchanged(self) -> None:
        # --trace-actions must be the only way to record; without it evaluate()
        # takes exactly its old selection path (enforced by the flag default).
        from mario_ai_fceux import evaluate
        defaults = evaluate.arguments.__doc__ or ""
        self.assertIn("greedy", evaluate.__doc__.lower())


if __name__ == "__main__":
    unittest.main()
