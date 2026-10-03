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

    def test_area_number_transitions_reset_progress_tracker_without_stuck_reset(self) -> None:
        # Area-aware progress tracking for multi-area levels:
        # area_number transitions reset progress_tracker without stuck reset.
        tracker = NoProgressTracker(limit_frames=240)

        # 1. Simulate observation sequence with (world=0, level=1, area=0) for 300 frames at world_x=24.
        for seq in range(1, 26):  # 25 updates * 12 frames = 300 frames
            obs = Observation(
                sequence=seq, state=np.zeros(184, dtype=np.float32),
                world_x=24, power=0, terminal=False, reason="",
                world=0, level=1, area=0,
            )
            tracker.update(obs.world_x, action_frames=12, area=obs.area)

        # 2. Warp to outdoor area (world=0, level=1, area=1) at world_x=40.
        obs_area1 = Observation(
            sequence=26, state=np.zeros(184, dtype=np.float32),
            world_x=40, power=0, terminal=False, reason="",
            world=0, level=1, area=1,
        )
        # Assert: does NOT trigger stuck reset after 240 frames in area 1
        stuck = tracker.update(obs_area1.world_x, action_frames=12, area=obs_area1.area)
        self.assertFalse(stuck)
        # Assert: correctly detects true progress in area 2 (area=1)
        self.assertEqual(tracker.best_world_x, 40)
        self.assertEqual(tracker.frames_without_progress, 0)

        # 3. Simulate root-cause case: underground area reaches world_x=2000+,
        # then warps to outdoor area at world_x=24 and makes true progress to 40.
        tracker_root_cause = NoProgressTracker(limit_frames=240)
        obs_underground = Observation(
            sequence=1, state=np.zeros(184, dtype=np.float32),
            world_x=2000, power=0, terminal=False, reason="",
            world=0, level=1, area=0,
        )
        self.assertFalse(tracker_root_cause.update(obs_underground.world_x, action_frames=0, area=obs_underground.area))
        self.assertEqual(tracker_root_cause.best_world_x, 2000)

        # Warp to outdoor area at world_x=24
        obs_outdoor_entry = Observation(
            sequence=2, state=np.zeros(184, dtype=np.float32),
            world_x=24, power=0, terminal=False, reason="",
            world=0, level=1, area=1,
        )
        # Transition resets best_world_x and frames_without_progress
        self.assertFalse(tracker_root_cause.update(obs_outdoor_entry.world_x, action_frames=12, area=obs_outdoor_entry.area))
        self.assertEqual(tracker_root_cause.best_world_x, 24)
        self.assertEqual(tracker_root_cause.frames_without_progress, 0)

        # Feed observations in area 1 with no X progress beyond 24 for 18 updates (216 frames)
        for _ in range(18):
            self.assertFalse(tracker_root_cause.update(obs_outdoor_entry.world_x, action_frames=12, area=obs_outdoor_entry.area))

        # Mario makes true progress in area 2 (area=1) by moving to world_x=40
        obs_outdoor_progress = Observation(
            sequence=3, state=np.zeros(184, dtype=np.float32),
            world_x=40, power=0, terminal=False, reason="",
            world=0, level=1, area=1,
        )
        self.assertFalse(tracker_root_cause.update(obs_outdoor_progress.world_x, action_frames=12, area=obs_outdoor_progress.area))
        # Assert: correctly detects true progress in area 2
        self.assertEqual(tracker_root_cause.best_world_x, 40)
        self.assertEqual(tracker_root_cause.frames_without_progress, 0)


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

    def test_frontier_commands_are_sequence_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = FileWorker("worker-00", directory, action_profile="rainbow")
            observation = Observation(23, np.zeros(184, dtype=np.float32), 320, 0, False, "")
            worker.send_action(observation, 8, checkpoint_frontier=True)
            self.assertEqual(read_json(worker.command_path), {
                "sequence": 23, "action": 8, "duration_frames": 24,
                "reset": False, "checkpoint": True,
            })
            worker.reset(observation, restore_frontier=True)
            self.assertEqual(read_json(worker.command_path), {
                "sequence": 23, "action": 3, "reset": True,
                "restore_frontier": True,
            })

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

    @unittest.skipUnless(shutil.which("luajit"), "LuaJIT is required")
    def test_command_fields_stay_bound_to_their_names(self) -> None:
        """A field appended to the protocol must never shift another field.

        The dispatch once unpacked readCommand's returns positionally, so the
        loop's checkpoint flag received the raw action code and restoreFrontier
        received the checkpoint flag. Named binding makes that impossible and
        this test pins each field independently.
        """
        with tempfile.TemporaryDirectory() as directory:
            worker_directory = Path(directory) / "worker"
            bridge = prepare_worker_directory(
                PROJECT_ROOT / "python/fceux_bridge/mario_ai_fceux_bridge.lua",
                worker_directory, 1, "rainbow")
            script = r'''
                MARIO_AI_TEST_PHASE=true
                memory={readbyte=function() return 0 end}
                local bridge=assert(loadfile(arg[0]))()
                local write=io.open(arg[0]:gsub("mario_ai_fceux_bridge%.lua$","").."command.json","w")
                write:write('{"sequence":7,"action":8,"duration_frames":24}')
                write:flush(); write:close()
                local command=bridge.readCommand(7)
                assert(command.base==3, "base should be action//3+1")
                assert(command.action==8, "action must stay 8")
                assert(command.durationFrames==24, "duration must stay 24")
                assert(command.checkpoint==false, "checkpoint must default false")
                assert(command.restoreFrontier==false, "restoreFrontier must default false")
                assert(command.reset==false and command.hold==false and command.advance==false)
                assert(command.campaignReset==false and command.restartWithCheats==nil)
                local handle=io.open(arg[0]:gsub("mario_ai_fceux_bridge%.lua$","").."command.json","w")
                handle:write('{"sequence":7,"action":3,"reset":true,"restore_frontier":true}')
                handle:flush(); handle:close()
                local resetCommand=bridge.readCommand(7)
                assert(resetCommand.reset==true, "reset must bind to reset")
                assert(resetCommand.restoreFrontier==true, "restore_frontier must bind to restoreFrontier")
                assert(resetCommand.checkpoint==false, "checkpoint stays independent of reset")
            '''
            subprocess.run([shutil.which("luajit"), "-e", script, str(bridge)],
                           check=True, capture_output=True, text=True)

    @unittest.skipUnless(shutil.which("luajit"), "LuaJIT is required")
    def test_bridge_parses_frontier_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker_directory = Path(directory) / "worker"
            bridge = prepare_worker_directory(
                PROJECT_ROOT / "python/fceux_bridge/mario_ai_fceux_bridge.lua",
                worker_directory, 1, "rainbow")
            (worker_directory / "command.json").write_text(
                '{"sequence":7,"action":8,"duration_frames":24,"reset":true,'
                '"checkpoint":true,"restore_frontier":true}', encoding="utf-8")
            script = r'''
                MARIO_AI_TEST_PHASE=true
                memory={readbyte=function() return 0 end}
                local bridge=assert(loadfile(arg[0]))()
                local command=bridge.readCommand(7)
                assert(command.durationFrames==24 and command.reset and command.action==8
                  and command.checkpoint and command.restoreFrontier)
            '''
            subprocess.run([shutil.which("luajit"), "-e", script, str(bridge)],
                           check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()

    @unittest.skipUnless(shutil.which("luajit"), "LuaJIT is required")
    def test_bridge_unpacks_readcommand_table_correctly(self) -> None:
        """readCommand returns named table; unpacking must extract fields, not assign table itself."""
        with tempfile.TemporaryDirectory() as directory:
            worker_directory = Path(directory) / "worker"
            bridge = prepare_worker_directory(
                PROJECT_ROOT / "python/fceux_bridge/mario_ai_fceux_bridge.lua",
                worker_directory, 1, "rainbow")
            test_cases = [(0, 6), (0, 12), (0, 24), (9, 6), (9, 12), (9, 24), (20, 24)]
            for action, duration in test_cases:
                (bridge.parent / "command.json").write_text(
                    f'{{"sequence":7,"action":{action},"duration_frames":{duration}}}',
                    encoding="utf-8")
                script = r'''
                    MARIO_AI_TEST_PHASE=true
                    memory={readbyte=function() return 0 end}
                    local bridge=assert(loadfile(arg[0]))()
                    local command = bridge.readCommand(7)
                    assert(command ~= nil, "readCommand returned nil")
                    assert(type(command) == "table", "readCommand should return a table")
                    assert(type(command.action) == "number", "action must be number not "..type(command.action))
                    assert(type(command.durationFrames) == "number", "durationFrames must be number not "..type(command.durationFrames))
                    local action, durationFrames = command.action, command.durationFrames
                    local baseIndex = math.floor(action / 3)
                    assert(baseIndex >= 0 and baseIndex <= 6, "baseIndex out of range")
                '''
                subprocess.run([shutil.which("luajit"), "-e", script, str(bridge.parent / "mario_ai_fceux_bridge.lua")],
                               check=True, capture_output=True, text=True)
