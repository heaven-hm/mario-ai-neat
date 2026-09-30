from __future__ import annotations

import sys
import tempfile
import unittest
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.environment import FileWorker, NoProgressTracker, Observation, prepare_worker_directory
from mario_ai_fceux.protocol import read_json
import numpy as np


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
    def test_campaign_commands_are_sequence_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = FileWorker("worker-00", directory, action_profile="rainbow")
            observation = Observation(23, np.zeros(184, dtype=np.float32), 40, 0, False, "",
                                      world=0, level=1)
            worker.advance_level(observation)
            self.assertEqual(read_json(worker.command_path),
                             {"sequence": 23, "action": 3, "advance": True})
            worker.restart_world(observation)
            self.assertEqual(read_json(worker.command_path),
                             {"sequence": 23, "action": 3, "campaign_reset": True})

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

    @unittest.skipUnless(shutil.which("luajit"), "LuaJIT is required")
    def test_smb1_flagpole_and_level_end_are_victories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = prepare_worker_directory(
                PROJECT_ROOT / "python/fceux_bridge/mario_ai_fceux_bridge.lua",
                Path(directory) / "worker", 1, "rainbow")
            script = r'''
                MARIO_AI_TEST_PHASE=true
                local bytes={[0x0770]=1,[0x000E]=8}
                memory={readbyte=function(address) return bytes[address] or 0 end}
                local bridge=assert(loadfile(arg[0]))()
                assert(bridge.phase()=="playing")
                bytes[0x000E]=4; assert(bridge.phase()=="victory")
                bytes[0x000E]=5; assert(bridge.phase()=="victory")
                bytes[0x000E]=6; assert(bridge.phase()=="death")
                bytes[0x000E]=8; bytes[0x010E]=0x3e; bytes[0x070f]=0xa0
                assert(bridge.phase()=="playing")
                bytes[0x0770]=2; assert(bridge.phase()=="waiting")
            '''
            subprocess.run([shutil.which("luajit"), "-e", script, str(bridge)],
                           check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
