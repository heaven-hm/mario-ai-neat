#!/usr/bin/env python3
"""Cron-friendly health and progress summary for Mario AI training."""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path


MARKER = "mario_ai_fceux.apex_train"
HUD_STALE_SECONDS = 180
MINIMUM_FREE_BYTES = 5 * 1024**3


def process_commands() -> list[str]:
    result = subprocess.run(["ps", "-ax", "-o", "command="], check=True,
                          capture_output=True, text=True)
    return result.stdout.splitlines()


def file_age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / "full-rainbow")
    options = parser.parse_args()
    run_directory = options.run_dir.resolve()
    health_directory = run_directory / "health"
    repository_root = Path(__file__).resolve().parents[1]
    try:
        metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    expected_workers = int(metadata.get("training_actors", 8))
    commands = process_commands()
    def belongs_to_run(command: str) -> bool:
        if MARKER not in command:
            return False
        # Training is often launched with a repository-relative --run-dir,
        # while FCEUX bridge paths are absolute. Normalize both forms.
        if str(run_directory) in command:
            return True
        try:
            tokens = shlex.split(command)
            run_index = tokens.index("--run-dir")
            process_run = Path(tokens[run_index + 1])
            if not process_run.is_absolute():
                process_run = repository_root / process_run
            return process_run.resolve() == run_directory
        except (ValueError, IndexError):
            return False

    trainers = sum(belongs_to_run(command) for command in commands)
    worker_prefix = str(run_directory / "worker-")
    worker_commands = [command for command in commands
                       if "mario_ai_fceux_bridge.lua" in command and worker_prefix in command]
    observation_ages = [file_age_seconds(run_directory / f"worker-{index:02d}" / "observation.json")
                        for index in range(expected_workers)]
    fresh_workers = sum(age is not None and age <= HUD_STALE_SECONDS for age in observation_ages)
    disk = shutil.disk_usage(repository_root)
    repairs: list[str] = []
    if trainers != 1:
        repairs.append(f"expected one Ape-X trainer for this run; found {trainers}")
    if len(worker_commands) != expected_workers:
        repairs.append(f"expected {expected_workers} FCEUX actors; found {len(worker_commands)}")
    if fresh_workers != expected_workers:
        repairs.append(f"expected {expected_workers} fresh observations; found {fresh_workers}")
    if disk.free < MINIMUM_FREE_BYTES:
        repairs.append("disk free space is below 5 GB")

    trainer_health_path = health_directory / "latest.json"
    trainer_age = file_age_seconds(trainer_health_path)
    try:
        trainer_health = json.loads(trainer_health_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        trainer_health = {}
    # The in-process report is periodic. Prefer the learner's live 10-second
    # status and per-actor HUD files so cron does not report stale counters.
    learner_path = run_directory / "learner_status.json"
    try:
        live_learner = json.loads(learner_path.read_text(encoding="utf-8"))
        if learner_path.stat().st_mtime >= trainer_health_path.stat().st_mtime:
            trainer_health["learner"] = live_learner
    except (OSError, json.JSONDecodeError):
        pass
    live_actors = []
    for index in range(expected_workers):
        try:
            hud = json.loads((run_directory / f"worker-{index:02d}" / "hud.json")
                             .read_text(encoding="utf-8"))
            live_actors.append({
                "actor": index,
                "steps": int(hud.get("steps", 0)),
                "episodes": int(hud.get("episodes", 0)),
                "deaths": int(hud.get("deaths", 0)),
                "victories": int(hud.get("victories", 0)),
                "best_episode_x": int(hud.get("best_x", 0)),
            })
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    if live_actors:
        trainer_health["actors"] = live_actors
    current_learner = trainer_health.get("learner", {})
    trainer_health["measured_progress"] = {
        "python_transitions": current_learner.get("transitions_received", 0),
        "python_optimizer_updates": current_learner.get("optimizer_updates", 0),
        "python_replay_size": current_learner.get("replay_transitions", 0),
        "python_best_episode_x": max(
            (int(actor.get("best_episode_x", 0)) for actor in trainer_health.get("actors", [])),
            default=0,
        ),
        "python_victories": sum(
            int(actor.get("victories", 0)) for actor in trainer_health.get("actors", [])
        ),
    }
    eval_path = run_directory / "eval_latest.json"
    try:
        trainer_health["evaluation"] = json.loads(eval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    if trainer_age is None or trainer_age > 12 * 60:
        repairs.append("trainer health snapshot is missing or older than 12 minutes")

    lua_log = repository_root / "mario_ai_neat.log"
    lua_database = repository_root / "mario_ai_neat.db"
    lua_age = file_age_seconds(lua_log)
    lua_summary = {"generation": None, "best_fitness": None, "latest_max_x": None}
    try:
        fields = lua_database.read_text(encoding="utf-8", errors="replace").splitlines()[0].split(",")
        if len(fields) == 6 and fields[0] == "MARIO_AI_NEAT_V1":
            lua_summary.update({"generation": int(fields[1]), "best_fitness": float(fields[3])})
    except (OSError, ValueError, IndexError):
        pass

    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "healthy": not repairs,
        "repair_required": repairs,
        "trainer_report_age_seconds": None if trainer_age is None else round(trainer_age, 1),
        "python_rainbow": {
            "trainer_count": trainers,
            "worker_count": len(worker_commands),
            "fresh_observation_count": fresh_workers,
            "observation_ages_seconds": observation_ages,
        },
        "learner": trainer_health.get("learner", {}),
        "actors": trainer_health.get("actors", []),
        "evaluation": trainer_health.get("evaluation"),
        "measured_progress": trainer_health.get("measured_progress", {}),
        "lua_neat": {
            "log_age_seconds": None if lua_age is None else round(lua_age, 1),
            "state": "active" if lua_age is not None and lua_age <= 600 else "stale_or_not_detected",
            **lua_summary,
        },
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total},
    }
    health_directory.mkdir(parents=True, exist_ok=True)
    (health_directory / "cron_latest.json").write_text(json.dumps(report, indent=2) + "\n",
                                                       encoding="utf-8")
    with (health_directory / "cron_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, separators=(",", ":")) + "\n")

    learner = report["learner"]
    measured = report["measured_progress"]
    evaluation = report["evaluation"] or {}
    rows = [
        "# Scheduled Mario AI health check", "",
        f"Checked: {report['checked_at']}",
        f"Healthy: {'yes' if report['healthy'] else 'no'}", "",
        "| System | Transitions / generation | Updates / fitness | Replay / latest X | Eval win rate |",
        "| --- | ---: | ---: | ---: | ---: |",
        (f"| Python Ape-X Rainbow | {learner.get('transitions_received', '?')} | "
         f"{learner.get('optimizer_updates', '?')} | {learner.get('replay_transitions', '?')} | "
         f"{evaluation.get('win_rate', '?')} |"),
        (f"| Lua NEAT | generation {lua_summary.get('generation', '?')} | "
         f"fitness {lua_summary.get('best_fitness', '?')} | "
         f"X {lua_summary.get('latest_max_x', '?')} | — |"),
        "",
        f"Python progress: {measured}", "",
        "## Repair items", "",
    ]
    rows.extend(f"- {item}" for item in repairs) if repairs else rows.append("- None")
    (health_directory / "cron_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if repairs else 0


if __name__ == "__main__":
    raise SystemExit(main())
