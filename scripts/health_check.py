#!/usr/bin/env python3
"""Write a local health report for the Lua NEAT and Python Rainbow trainers."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = PROJECT_ROOT / "runs" / "world-1-1-rainbow"
STATUS_DIRECTORY = RUN_DIRECTORY / "health"
WORKER_COUNT = 8
HUD_STALE_SECONDS = 180
MINIMUM_FREE_BYTES = 5 * 1024**3


def process_commands() -> list[str]:
    completed = subprocess.run(["ps", "-ax", "-o", "command="], check=True,
                               capture_output=True, text=True)
    return completed.stdout.splitlines()


def file_age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def main() -> None:
    commands = process_commands()
    trainer_count = sum("mario_ai_fceux.train" in command for command in commands)
    worker_commands = [command for command in commands
                       if "mario_ai_fceux_bridge.lua" in command and "world-1-1-rainbow" in command]
    lua_fceux_count = sum("fceux" in command and "world-1-1-rainbow" not in command
                          for command in commands)
    hud_ages = [file_age_seconds(RUN_DIRECTORY / f"worker-{index:02d}" / "hud.json")
                for index in range(WORKER_COUNT)]
    fresh_huds = sum(age is not None and age <= HUD_STALE_SECONDS for age in hud_ages)
    disk = shutil.disk_usage(PROJECT_ROOT)
    repairs: list[str] = []
    if trainer_count != 1:
        repairs.append(f"expected one Python trainer; found {trainer_count}")
    if len(worker_commands) != WORKER_COUNT:
        repairs.append(f"expected {WORKER_COUNT} Python FCEUX workers; found {len(worker_commands)}")
    if fresh_huds != WORKER_COUNT:
        repairs.append(f"expected {WORKER_COUNT} fresh worker HUD files; found {fresh_huds}")
    if disk.free < MINIMUM_FREE_BYTES:
        repairs.append("disk free space is below 5 GB")
    lua_log_age = file_age_seconds(PROJECT_ROOT / "mario_ai_neat.log")
    lua_state = "not_detected"
    if lua_fceux_count:
        lua_state = "running" if lua_log_age is not None and lua_log_age <= 900 else "needs_attention"
    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "healthy": not repairs,
        "repair_required": repairs,
        "python_rainbow": {
            "trainer_count": trainer_count,
            "worker_count": len(worker_commands),
            "fresh_hud_count": fresh_huds,
            "hud_ages_seconds": hud_ages,
        },
        "lua_neat": {
            "fceux_process_count": lua_fceux_count,
            "log_age_seconds": lua_log_age,
            "state": lua_state,
        },
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total},
    }
    STATUS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    latest = STATUS_DIRECTORY / "latest.json"
    temporary = latest.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(latest)
    with (STATUS_DIRECTORY / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
