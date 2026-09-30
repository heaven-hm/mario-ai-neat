from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.environment import NoProgressTracker, prepare_worker_directory


class NoProgressTrackerTests(unittest.TestCase):
    def test_standing_jump_ends_after_four_game_seconds(self) -> None:
        tracker = NoProgressTracker(limit_frames=240)
        self.assertFalse(tracker.update(40))
        for _ in range(19):
            self.assertFalse(tracker.update(40, 12))
        self.assertTrue(tracker.update(40, 12))

    def test_new_forward_progress_resets_stuck_clock(self) -> None:
        tracker = NoProgressTracker(limit_frames=24)
        self.assertFalse(tracker.update(40))
        self.assertFalse(tracker.update(40, 12))
        self.assertFalse(tracker.update(41, 12))
        self.assertFalse(tracker.update(41, 12))
        self.assertTrue(tracker.update(41, 12))
        tracker.reset()
        self.assertFalse(tracker.update(40, 24))


class WorkerDirectoryTests(unittest.TestCase):
    def test_prepare_clears_stale_protocol_status_before_bridge_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "bridge.lua"
            worker = root / "worker"
            template.write_text(
                'local path="__WORKER_DIRECTORY__"\n'
                'local world=__TARGET_WORLD_INDEX__\n'
                'local profile="__ACTION_PROFILE__"\n',
                encoding="utf-8",
            )
            worker.mkdir()
            stale_files = (
                "observation.json", "command.json", "bridge_started.json",
                "status.json", "hud.json",
            )
            for name in stale_files:
                (worker / name).write_text("stale", encoding="utf-8")

            bridge_path = prepare_worker_directory(template, worker, 2, "rainbow")

            self.assertTrue(bridge_path.is_file())
            self.assertIn('world=1', bridge_path.read_text(encoding="utf-8"))
            self.assertIn('profile="rainbow"', bridge_path.read_text(encoding="utf-8"))
            self.assertTrue(all(not (worker / name).exists() for name in stale_files))


if __name__ == "__main__":
    unittest.main()
