"""FCEUX file-protocol environment and multi-worker process launcher."""

from __future__ import annotations

from dataclasses import dataclass
import configparser
import shutil
import subprocess
import time
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .actions import ACTION_NAMES, LEGACY_DURATION_FRAMES, decode_action
from .protocol import atomic_write_json, read_json


START_PROTOCOL = "SMB1 clean selected-world start with course-start retries"
NO_PROGRESS_LIMIT_FRAMES = 240


@dataclass(frozen=True)
class Observation:
    sequence: int
    state: np.ndarray
    world_x: int
    power: int
    terminal: bool
    reason: str
    # Zero-based SMB1 identifiers; defaults keep older fixed-level callers
    # compatible while campaign-aware actors use the bridge values.
    # Area-aware progress tracking for multi-area levels
    world: int = 0
    level: int = 0
    area: int = 0


class NoProgressTracker:
    """End attempts that spend four game seconds without gaining ground."""

    def __init__(self, limit_frames: int = NO_PROGRESS_LIMIT_FRAMES) -> None:
        self.limit_frames = limit_frames
        self.best_world_x: int | None = None
        self.frames_without_progress = 0
        # Area-aware progress tracking for multi-area levels
        self.previous_area: int | None = None

    def reset(self) -> None:
        self.best_world_x = None
        self.frames_without_progress = 0
        # Area-aware progress tracking for multi-area levels
        self.previous_area = None

    def update(self, world_x: int | Observation, action_frames: int = 0,
               area: int | None = None) -> bool:
        # Area-aware progress tracking for multi-area levels
        if isinstance(world_x, Observation):
            if area is None:
                area = world_x.area
            world_x = world_x.world_x

        if area is not None and self.previous_area is not None and area != self.previous_area:
            self.best_world_x = None
            self.frames_without_progress = 0

        if self.best_world_x is None or world_x > self.best_world_x:
            self.best_world_x = world_x
            self.frames_without_progress = 0
        else:
            self.frames_without_progress += max(0, action_frames)

        self.previous_area = area
        return self.frames_without_progress >= self.limit_frames


class FileWorker:
    """One FCEUX Lua bridge worker. It does no training in Lua."""

    def __init__(self, worker_id: str, directory: str | Path, observation_size: int = 184,
                 action_profile: str = "legacy"):
        self.worker_id = worker_id
        self.directory = Path(directory)
        self.observation_path = self.directory / "observation.json"
        self.command_path = self.directory / "command.json"
        self.observation_size = observation_size
        if action_profile not in ("legacy", "rainbow"):
            raise ValueError("action_profile must be 'legacy' or 'rainbow'")
        self.action_profile = action_profile
        self.last_sequence = -1
        self.previous: Observation | None = None
        self._bridge_restarted = False

    def next_observation(self) -> Observation | None:
        message = read_json(self.observation_path)
        if not message or not isinstance(message.get("sequence"), int):
            return None
        sequence = int(message["sequence"])
        if sequence <= self.last_sequence:
            # A campaign boundary can restart only this worker's FCEUX process
            # to switch its private cheat configuration.  Its Lua sequence
            # then restarts at 1; accept that as a new episode source.
            if sequence < self.last_sequence:
                self.last_sequence = -1
                self.previous = None
                self._bridge_restarted = True
            else:
                return None
        features = message.get("features")
        if not isinstance(features, list) or len(features) != self.observation_size:
            raise ValueError(f"worker {self.worker_id} emitted an invalid feature vector")
        observation = Observation(
            sequence=sequence,
            state=np.asarray(features, dtype=np.float32),
            world_x=int(message.get("world_x", 0)),
            power=int(message.get("power", 0)),
            # SMB1 stores both values zero-based.  Keep that representation in
            # the protocol so campaign routing can identify levels 0..3.
            world=int(message.get("world", 0)),
            level=int(message.get("level", 0)),
            # Area-aware progress tracking for multi-area levels
            area=int(message.get("area", 0)),
            terminal=bool(message.get("terminal", False)),
            reason=str(message.get("reason", "")),
        )
        self.last_sequence = sequence
        return observation

    def send_action(self, observation: Observation, action: int,
                    checkpoint_frontier: bool = False) -> None:
        if self.action_profile == "rainbow":
            _, duration_frames = decode_action(action)
        else:
            if action < 0 or action >= 6:
                raise ValueError(f"invalid legacy SMB1 action: {action}")
            duration_frames = LEGACY_DURATION_FRAMES
        atomic_write_json(self.command_path, {"sequence": observation.sequence,
                                              "action": int(action),
                                              "duration_frames": duration_frames,
                                              "reset": False,
                                              "checkpoint": bool(checkpoint_frontier)})

    def reset(self, observation: Observation, restore_frontier: bool = False) -> None:
        """Restart a failed attempt, optionally from the latest safe frontier."""
        atomic_write_json(self.command_path, {
            "sequence": observation.sequence, "action": 3, "reset": True,
            "restore_frontier": bool(restore_frontier),
        })

    def advance_level(self, observation: Observation) -> None:
        """Let SMB1 load the next level and save its new start checkpoint."""
        atomic_write_json(self.command_path, {"sequence": observation.sequence, "action": 3,
                                              "advance": True})

    def restart_world(self, observation: Observation) -> None:
        """Restore this worker's saved World-N-1 state after World-N-4."""
        atomic_write_json(self.command_path, {"sequence": observation.sequence, "action": 3,
                                              "campaign_reset": True})

    def restart_world_with_cheat_mode(self, observation: Observation, cheats_enabled: bool) -> None:
        """Request an FCEUX restart with the supplied private cheat mode."""
        atomic_write_json(self.command_path, {
            "sequence": observation.sequence, "action": 3, "campaign_reset": True,
            "restart_with_cheats": bool(cheats_enabled),
        })

    def consume_bridge_restart(self) -> bool:
        restarted = self._bridge_restarted
        self._bridge_restarted = False
        return restarted

    def hold(self, observation: Observation) -> None:
        """Keep an episode-boundary PPO actor on its saved start state."""
        atomic_write_json(self.command_path, {"sequence": observation.sequence, "action": 3,
                                              "reset": False, "hold": True})


def prepare_worker_directory(template: Path, worker_directory: Path, target_world: int,
                             action_profile: str = "legacy") -> Path:
    """Create an isolated bridge file and protocol directory for one emulator."""
    if not 1 <= target_world <= 8:
        raise ValueError("SMB1 target worlds must be in the range 1..8")
    if action_profile not in ("legacy", "rainbow"):
        raise ValueError("action_profile must be 'legacy' or 'rainbow'")
    worker_directory.mkdir(parents=True, exist_ok=True)
    # Remove protocol state from a previous process. In particular, stale
    # bridge_started/status files can falsely make a new emulator look ready.
    for name in ("observation.json", "command.json", "bridge_started.json",
                 "status.json", "hud.json", "mode_request.json"):
        try:
            (worker_directory / name).unlink()
        except FileNotFoundError:
            pass
    bridge_path = worker_directory / "mario_ai_fceux_bridge.lua"
    worker_literal = str(worker_directory.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    bridge_source = (template.read_text(encoding="utf-8")
                     .replace("__WORKER_DIRECTORY__", worker_literal)
                     .replace("__TARGET_WORLD_INDEX__", str(target_world - 1))
                     .replace("__ACTION_PROFILE__", action_profile))
    bridge_path.write_text(bridge_source, encoding="utf-8")
    return bridge_path


def launch_fceux_workers(fceux: str, rom: Path, bridge_template: Path, run_directory: Path,
                         count: int, target_worlds: Sequence[int] | None = None,
                         extra_args: Sequence[str] = (),
                         action_profile: str = "legacy",
                         cheats_enabled: Sequence[bool] | None = None,
                         window_layout: Path | None = None,
                         cheat_file: Path | None = None,
                         worker_indexes: Sequence[int] | None = None) -> list[subprocess.Popen[bytes]]:
    """Launch isolated FCEUX processes. FCEUX officially supports `-lua` and `-nothrottle`."""
    executable = shutil.which(fceux) or fceux
    if not Path(rom).is_file():
        raise FileNotFoundError(f"SMB1 ROM was not found: {rom}")
    if count < 1:
        raise ValueError("worker count must be positive")
    assigned_worlds = tuple(target_worlds or (1,) * count)
    if len(assigned_worlds) != count:
        raise ValueError("target_worlds must contain one world for each worker")
    indexes = tuple(worker_indexes or range(count))
    if len(indexes) != count or len(set(indexes)) != count or any(index < 0 for index in indexes):
        raise ValueError("worker_indexes must contain unique non-negative worker indexes")
    if cheats_enabled is None:
        assigned_cheats = (False,) * count
    else:
        assigned_cheats = tuple(bool(value) for value in cheats_enabled)
        if len(assigned_cheats) != count:
            raise ValueError("cheats_enabled must contain one value for each worker")
    layout: configparser.ConfigParser | None = None
    if window_layout is not None and Path(window_layout).is_file():
        layout = configparser.ConfigParser()
        layout.read(window_layout)
    # FCEUX 2.6 uses --loadlua; earlier builds document -lua.  Detect the
    # installed binary rather than assuming the older spelling.
    try:
        help_result = subprocess.run([executable, "--help"], capture_output=True, text=True,
                                     timeout=5, check=False)
        help_output = help_result.stdout + help_result.stderr
    except (OSError, subprocess.SubprocessError):
        help_output = ""
    lua_option = "--loadlua" if "--loadlua" in help_output else "-lua"
    processes: list[subprocess.Popen[bytes]] = []
    for local_index, index in enumerate(indexes):
        bridge = prepare_worker_directory(bridge_template, run_directory / f"worker-{index:02d}",
                                          assigned_worlds[local_index], action_profile)
        # FCEUX otherwise auto-loads ~/.fceux/cheats/SuperMarioBros.cht even
        # with --gamegenie 0. Give each process its own empty cheat directory
        # and explicitly disable the cheat engine in its private config.
        config_directory = bridge.parent / "fceux-config"
        config_directory.mkdir(exist_ok=True)
        cheats_on = assigned_cheats[local_index]
        (config_directory / "fceux.cfg").write_text(
            "SDL.CheatsDisableAutoLS = %d\nSDL.CheatsDisabled = %d\nSDL.GameGenie = 0\n"
            % (0 if cheats_on else 1, 0 if cheats_on else 1), encoding="utf-8")
        if cheats_on and cheat_file is not None and Path(cheat_file).is_file():
            cheat_directory = config_directory / "cheats"
            cheat_directory.mkdir(exist_ok=True)
            shutil.copyfile(cheat_file, cheat_directory / f"{rom.stem}.cht")
        environment = os.environ.copy()
        environment["FCEUX_CONFIG_DIR"] = str(config_directory.resolve())
        # FCEUX is launched from the worker directory, so the bridge must be
        # absolute; otherwise a relative run directory is resolved twice.
        # Disable Game Genie for every training and evaluation instance.
        geometry = None
        if layout is not None and layout.has_section(f"worker-{index:02d}"):
            section = layout[f"worker-{index:02d}"]
            try:
                x, y = int(section["x"]), int(section["y"])
                width, height = int(section["width"]), max(200, int(section["height"]) - 28)
                geometry = f"{width}x{height}+{x}+{y}"
            except (KeyError, ValueError):
                geometry = None
        fceux_arguments = (["--gamegenie", "0"] if not cheats_on else [])
        if geometry:
            # One-times scaling is required for four 256px NES viewports to
            # fit across a 1680px display; FCEUX otherwise enforces a 512px
            # minimum at its default 2x scale.
            fceux_arguments.extend(["--xscale", "1", "--yscale", "1",
                                     "-qwindowgeometry", geometry])
        fceux_arguments.extend([lua_option, str(bridge.resolve()), *extra_args, str(rom.resolve())])
        # Launch the executable directly on every OS.  On macOS `open -na`
        # returns the launcher PID rather than the emulator PID, which prevents
        # reliable health checks and cleanup of the worker process.
        command = [executable, *fceux_arguments]
        try:
            processes.append(subprocess.Popen(command, cwd=bridge.parent, env=environment))
            # Let each Qt/SDL process finish attaching to the desktop before
            # the next window starts; simultaneous FCEUX initialization can
            # leave an instance without a live Lua bridge on macOS.
            time.sleep(0.2)
        except BaseException:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
    return processes


def wait_for_workers(workers: Sequence[FileWorker], timeout_seconds: float = 30.0) -> list[Observation]:
    deadline = time.monotonic() + timeout_seconds
    remaining = {worker.worker_id: worker for worker in workers}
    observations: list[Observation] = []
    while remaining and time.monotonic() < deadline:
        for worker_id, worker in list(remaining.items()):
            observation = worker.next_observation()
            if observation:
                observations.append(observation)
                remaining.pop(worker_id)
        if remaining:
            time.sleep(0.01)
    if remaining:
        names = ", ".join(sorted(remaining))
        raise TimeoutError(f"FCEUX workers did not publish observations: {names}")
    return observations
