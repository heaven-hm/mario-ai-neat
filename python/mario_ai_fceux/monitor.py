"""Read the trainer's heartbeat without shelling out to process-list tools."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable


HEARTBEAT_STALE_SECONDS = 30


def process_is_alive(process_id: int, kill: Callable[[int, int], None] = os.kill) -> bool:
    if process_id <= 0:
        return False
    try:
        kill(process_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def read_run_monitor(run_directory: Path, now: float | None = None,
                     kill: Callable[[int, int], None] = os.kill) -> dict[str, object]:
    """Validate supervisor heartbeat and child process liveness from JSON."""
    try:
        status = json.loads((run_directory / "supervisor.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"running": False, "trainer_alive": False, "actor_alive": 0,
                "emulator_alive": 0, "eval_alive": False, "eval_emulator_alive": 0,
                "heartbeat_age_seconds": None}
    current_time = time.time() if now is None else now
    try:
        heartbeat_age = max(0.0, current_time - float(status["heartbeat_unix"]))
        trainer_pid = int(status["trainer_pid"])
    except (KeyError, TypeError, ValueError):
        heartbeat_age = None
        trainer_pid = -1
    trainer_alive = heartbeat_age is not None and heartbeat_age <= HEARTBEAT_STALE_SECONDS \
        and process_is_alive(trainer_pid, kill)
    actor_alive = sum(process_is_alive(int(pid), kill) for pid in status.get("actor_pids", []))
    emulator_alive = sum(process_is_alive(int(pid), kill) for pid in status.get("emulator_pids", []))
    eval_alive = process_is_alive(int(status.get("eval_pid", -1)), kill)
    eval_emulator_alive = sum(process_is_alive(int(pid), kill)
                              for pid in status.get("eval_emulator_pids", []))
    return {
        "running": bool(status.get("running")) and trainer_alive,
        "trainer_alive": trainer_alive,
        "actor_alive": actor_alive,
        "emulator_alive": emulator_alive,
        "eval_alive": eval_alive,
        "eval_emulator_alive": eval_emulator_alive,
        "heartbeat_age_seconds": None if heartbeat_age is None else round(heartbeat_age, 1),
    }
