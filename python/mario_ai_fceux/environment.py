"""FCEUX file-protocol environment and multi-worker process launcher."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .protocol import atomic_write_json, read_json


ACTION_NAMES = ("run", "jump_run", "retreat", "brake", "jump_place", "walk")


@dataclass(frozen=True)
class Observation:
    sequence: int
    state: np.ndarray
    world_x: int
    power: int
    terminal: bool
    reason: str


class FileWorker:
    """One FCEUX Lua bridge worker. It does no training in Lua."""

    def __init__(self, worker_id: str, directory: str | Path, observation_size: int = 184):
        self.worker_id = worker_id
        self.directory = Path(directory)
        self.observation_path = self.directory / "observation.json"
        self.command_path = self.directory / "command.json"
        self.observation_size = observation_size
        self.last_sequence = -1
        self.previous: Observation | None = None

    def next_observation(self) -> Observation | None:
        message = read_json(self.observation_path)
        if not message or not isinstance(message.get("sequence"), int):
            return None
        sequence = int(message["sequence"])
        if sequence <= self.last_sequence:
            return None
        features = message.get("features")
        if not isinstance(features, list) or len(features) != self.observation_size:
            raise ValueError(f"worker {self.worker_id} emitted an invalid feature vector")
        observation = Observation(
            sequence=sequence,
            state=np.asarray(features, dtype=np.float32),
            world_x=int(message.get("world_x", 0)),
            power=int(message.get("power", 0)),
            terminal=bool(message.get("terminal", False)),
            reason=str(message.get("reason", "")),
        )
        self.last_sequence = sequence
        return observation

    def send_action(self, observation: Observation, action: int) -> None:
        if action < 0 or action >= len(ACTION_NAMES):
            raise ValueError(f"invalid SMB1 action: {action}")
        atomic_write_json(self.command_path, {"sequence": observation.sequence, "action": int(action), "reset": False})

    def reset(self, observation: Observation) -> None:
        atomic_write_json(self.command_path, {"sequence": observation.sequence, "action": 3, "reset": True})


def prepare_worker_directory(template: Path, worker_directory: Path, target_world: int) -> Path:
    """Create an isolated bridge file and protocol directory for one emulator."""
    if not 1 <= target_world <= 8:
        raise ValueError("SMB1 target worlds must be in the range 1..8")
    worker_directory.mkdir(parents=True, exist_ok=True)
    for name in ("observation.json", "command.json"):
        try:
            (worker_directory / name).unlink()
        except FileNotFoundError:
            pass
    bridge_path = worker_directory / "mario_ai_fceux_bridge.lua"
    worker_literal = str(worker_directory.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    bridge_source = (template.read_text(encoding="utf-8")
                     .replace("__WORKER_DIRECTORY__", worker_literal)
                     .replace("__TARGET_WORLD_INDEX__", str(target_world - 1)))
    bridge_path.write_text(bridge_source, encoding="utf-8")
    return bridge_path


def launch_fceux_workers(fceux: str, rom: Path, bridge_template: Path, run_directory: Path,
                         count: int, target_worlds: Sequence[int] | None = None,
                         extra_args: Sequence[str] = ()) -> list[subprocess.Popen[bytes]]:
    """Launch isolated FCEUX processes. FCEUX officially supports `-lua` and `-nothrottle`."""
    executable = shutil.which(fceux) or fceux
    if not Path(rom).is_file():
        raise FileNotFoundError(f"SMB1 ROM was not found: {rom}")
    if count < 1:
        raise ValueError("worker count must be positive")
    assigned_worlds = tuple(target_worlds or (1,) * count)
    if len(assigned_worlds) != count:
        raise ValueError("target_worlds must contain one world for each worker")
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
    for index in range(count):
        bridge = prepare_worker_directory(bridge_template, run_directory / f"worker-{index:02d}",
                                          assigned_worlds[index])
        # FCEUX is launched from the worker directory, so the bridge must be
        # absolute; otherwise a relative run directory is resolved twice.
        fceux_arguments = [lua_option, str(bridge.resolve()), *extra_args, str(rom.resolve())]
        # Launch the executable directly on every OS.  On macOS `open -na`
        # returns the launcher PID rather than the emulator PID, which prevents
        # reliable health checks and cleanup of the worker process.
        command = [executable, *fceux_arguments]
        try:
            processes.append(subprocess.Popen(command, cwd=bridge.parent))
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
