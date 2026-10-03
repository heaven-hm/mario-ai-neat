from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.monitor import process_is_alive, read_run_monitor


class RunMonitorTests(unittest.TestCase):
    def test_live_fresh_supervisor_and_all_children_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "supervisor.json").write_text(json.dumps({
                "running": True, "heartbeat_unix": 100.0, "trainer_pid": 10,
                "actor_pids": [11, 12], "emulator_pids": [21, 22],
                "eval_pid": 13, "eval_emulator_pids": [23],
            }))
            report = read_run_monitor(root, now=110.0, kill=lambda pid, signal: None)
        self.assertTrue(report["running"])
        self.assertEqual(report["actor_alive"], 2)
        self.assertEqual(report["emulator_alive"], 2)
        self.assertEqual(report["eval_emulator_alive"], 1)

    def test_stale_heartbeat_is_not_treated_as_a_live_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "supervisor.json").write_text(json.dumps({
                "running": True, "heartbeat_unix": 10.0, "trainer_pid": 10,
                "actor_pids": [], "emulator_pids": [], "eval_pid": 0,
                "eval_emulator_pids": [],
            }))
            report = read_run_monitor(root, now=100.0, kill=lambda pid, signal: None)
        self.assertFalse(report["running"])
        self.assertFalse(report["trainer_alive"])

    def test_missing_pid_is_reported_dead(self) -> None:
        def missing_pid(_pid: int, _signal: int) -> None:
            raise ProcessLookupError

        self.assertFalse(process_is_alive(12, missing_pid))


if __name__ == "__main__":
    unittest.main()
