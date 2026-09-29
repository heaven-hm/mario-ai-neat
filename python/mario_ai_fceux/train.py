"""Command-line trainer that runs multiple FCEUX workers and one Python learner."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import signal
import time
from pathlib import Path

import numpy as np

from .agent import AgentConfig
from .environment import FileWorker, launch_fceux_workers
from .learner import learner_main
from .protocol import atomic_write_json


HEALTH_CHECK_INTERVAL_SECONDS = 10 * 60
WORKER_STALE_SECONDS = 3 * 60
MINIMUM_FREE_BYTES = 5 * 1024**3


def shaped_reward(previous, current) -> float:
    """Reward forward SMB1 progress and make deaths materially undesirable."""
    reward = max(-2.0, min(2.0, (current.world_x - previous.world_x) / 16.0))
    reward += max(-0.2, min(0.2, (current.power - previous.power) * 0.1))
    if current.terminal:
        reward += 20.0 if current.reason == "victory" else -5.0
    return reward


def read_lua_neat_summary(database_path: Path, log_path: Path) -> dict[str, int | float | None]:
    """Read stable, public progress fields without modifying Lua's checkpoint."""
    summary: dict[str, int | float | None] = {
        "generation": None, "innovation": None, "best_fitness": None,
        "population": None, "latest_max_x": None, "latest_fitness": None,
    }
    try:
        header = database_path.open(encoding="utf-8").readline().strip().split(",")
        if len(header) == 6 and header[0] == "MARIO_AI_NEAT_V1":
            summary.update({
                "generation": int(header[1]), "innovation": int(header[2]),
                "best_fitness": float(header[3]), "population": int(header[4]),
            })
    except (OSError, ValueError):
        pass
    try:
        # Logs are append-only. The last episode line is a factual recent run,
        # not a claim that the AI has permanently mastered that situation.
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            if "episode end |" not in line:
                continue
            fields = line.split("|")
            for field in fields:
                key, _, value = field.strip().partition("=")
                if key == "max_x":
                    summary["latest_max_x"] = int(value)
                elif key == "fitness":
                    summary["latest_fitness"] = float(value)
            break
    except (OSError, ValueError):
        pass
    return summary


def write_learning_table(health_directory: Path, report: dict[str, object]) -> None:
    """Write a human-readable report alongside the machine-readable health JSON."""
    learning = report["learning"]
    python = learning["python_dqn"]
    lua = learning["lua_neat"]
    rows = [
        "# Ten-minute learning report",
        "",
        f"Generated: {report['checked_at']}",
        "",
        "| AI path | Measured state now | Change since previous report | Evidence-based discovery |",
        "| --- | --- | --- | --- |",
        ("| Python Rainbow DQN | "
         f"{python['steps']} decisions; {python['optimizer_updates']} updates; "
         f"replay {python['replay_transitions']}; best X {python['best_x']} | "
         f"{python['change']} | {python['discovery']} |"),
        ("| Lua NEAT + contextual Q | "
         f"generation {lua['generation']}; best fitness {lua['best_fitness']}; "
         f"latest X {lua['latest_max_x']} | "
         f"{lua['change']} | {lua['discovery']} |"),
        "",
        "A discovery reports only saved progress, a new best distance, or a victory. "
        "It does not label a maneuver as mastered until repeatable evaluation supports that claim.",
        "",
    ]
    (health_directory / "learning_report.md").write_text("\n".join(rows), encoding="utf-8")


def write_health_report(run_directory: Path, repository_root: Path,
                        last_observation_at: dict[str, float], expected_workers: int,
                        python_training: dict[str, int | float | None]) -> dict[str, object]:
    """Record a trainer-owned health snapshot without a separate OS service.

    The trainer already receives each worker's observation, so this is more
    dependable on macOS than an external scheduled process that may be denied
    access to the user's project folder. A stale observation means the worker
    needs attention; this report never kills or restarts a live FCEUX process.
    """
    monotonic_now = time.monotonic()
    unix_now = time.time()
    worker_ages = {
        worker_id: round(max(0.0, monotonic_now - observed_at), 1)
        for worker_id, observed_at in sorted(last_observation_at.items())
    }
    fresh_workers = sum(age <= WORKER_STALE_SECONDS for age in worker_ages.values())
    repairs: list[str] = []
    if fresh_workers != expected_workers:
        stale_workers = [worker_id for worker_id, age in worker_ages.items()
                         if age > WORKER_STALE_SECONDS]
        repairs.append("stale Python worker observations: " + ", ".join(stale_workers))
    disk = shutil.disk_usage(repository_root)
    if disk.free < MINIMUM_FREE_BYTES:
        repairs.append("disk free space is below 5 GB")
    lua_log = repository_root / "mario_ai_neat.log"
    lua_log_age = (round(max(0.0, unix_now - lua_log.stat().st_mtime), 1)
                   if lua_log.exists() else None)
    lua_state = "running" if lua_log_age is not None and lua_log_age <= HEALTH_CHECK_INTERVAL_SECONDS else "not_detected_or_stale"
    health_directory = run_directory / "health"
    previous_report: dict[str, object] = {}
    try:
        previous_report = json.loads((health_directory / "latest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    lua_summary = read_lua_neat_summary(repository_root / "mario_ai_neat.db", lua_log)
    previous_learning = previous_report.get("learning", {}) if isinstance(previous_report, dict) else {}
    previous_python = previous_learning.get("python_dqn", {}) if isinstance(previous_learning, dict) else {}
    previous_lua = previous_learning.get("lua_neat", {}) if isinstance(previous_learning, dict) else {}
    previous_best_x = int(previous_python.get("best_x", python_training["best_x"]))
    previous_victories = int(previous_python.get("victories", python_training["victories"]))
    previous_lua_fitness = float(previous_lua.get("best_fitness", lua_summary["best_fitness"] or 0))
    previous_lua_generation = int(previous_lua.get("generation", lua_summary["generation"] or 0))
    python_best_x = int(python_training["best_x"] or 0)
    python_victories = int(python_training["victories"] or 0)
    lua_best_fitness = float(lua_summary["best_fitness"] or 0)
    lua_generation = int(lua_summary["generation"] or 0)
    python_change = f"+{python_best_x - previous_best_x} best X; +{python_victories - previous_victories} victories"
    lua_change = f"+{lua_generation - previous_lua_generation} generations; +{lua_best_fitness - previous_lua_fitness:.2f} best fitness"
    python_discovery = (f"new victory recorded ({python_victories} total)" if python_victories > previous_victories
                        else f"new best progress to X={python_best_x}" if python_best_x > previous_best_x
                        else "no new best progress in this interval")
    lua_discovery = (f"new record fitness {lua_best_fitness:.2f}" if lua_best_fitness > previous_lua_fitness
                     else f"advanced to generation {lua_generation}" if lua_generation > previous_lua_generation
                     else "no new saved record in this interval")
    report: dict[str, object] = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "healthy": not repairs,
        "repair_required": repairs,
        "python_rainbow": {
            "trainer_state": "running",
            "expected_workers": expected_workers,
            "fresh_workers": fresh_workers,
            "worker_observation_ages_seconds": worker_ages,
        },
        "lua_neat": {"log_age_seconds": lua_log_age, "state": lua_state},
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total},
        "learning": {
            "python_dqn": {**python_training, "change": python_change, "discovery": python_discovery},
            "lua_neat": {**lua_summary, "change": lua_change, "discovery": lua_discovery},
        },
    }
    health_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(health_directory / "latest.json", report)
    with (health_directory / "history.jsonl").open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(report, separators=(",", ":")) + "\n")
    write_learning_table(health_directory, report)
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a full Rainbow DQN SMB1 controller through FCEUX Lua workers.")
    parser.add_argument("--rom", type=Path, required=True, help="Path to a legally obtained SMB1 NES ROM.")
    parser.add_argument("--fceux", default="fceux", help="FCEUX executable path or command.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/python-rainbow"))
    parser.add_argument("--workers", type=int, default=4, help="Parallel FCEUX processes.")
    parser.add_argument("--worlds", default="1", help="Comma-separated SMB1 worlds, one per worker; each starts at level 1.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total action decisions to collect.")
    parser.add_argument("--resume", action="store_true", help="Load model.pt if it exists in the run directory.")
    parser.add_argument("--device", default=None, help="PyTorch device: mps, cuda, or cpu.")
    parser.add_argument("--seed", type=int, default=7, help="Seed saved in model and replay checkpoints.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    try:
        requested_worlds = tuple(int(value.strip()) for value in arguments.worlds.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("--worlds must contain integers in the range 1..8") from error
    if len(requested_worlds) == 1:
        requested_worlds *= arguments.workers
    if len(requested_worlds) != arguments.workers or any(world < 1 or world > 8 for world in requested_worlds):
        raise ValueError("--worlds must provide one value from 1..8 per worker")
    repository_root = Path(__file__).resolve().parents[2]
    bridge_template = repository_root / "python" / "fceux_bridge" / "mario_ai_fceux_bridge.lua"
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    configuration = AgentConfig(seed=arguments.seed)
    metadata = {"algorithm": "Rainbow DQN (C51 + NoisyNet + Double + Dueling + PER + n-step)",
                "workers": arguments.workers, "worlds": requested_worlds, "observation_size": 184,
                "actions": 6, "rom": str(arguments.rom), "seed": arguments.seed,
                "replay": "in-memory global PER; replay.npz checkpoint snapshot"}
    (arguments.run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    learner_inbox = context.Queue(maxsize=20_000)
    learner_outbox = context.Queue(maxsize=1_000)
    learner = context.Process(target=learner_main,
                              args=(learner_inbox, learner_outbox, str(arguments.run_dir), vars(configuration),
                                    arguments.resume, arguments.device), daemon=True)
    learner.start()
    processes = launch_fceux_workers(arguments.fceux, arguments.rom, bridge_template,
                                     arguments.run_dir, arguments.workers, requested_worlds)
    workers = [FileWorker(f"worker-{index:02d}", arguments.run_dir / f"worker-{index:02d}")
               for index in range(arguments.workers)]
    metrics_path = arguments.run_dir / "training_metrics.json"
    try:
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        saved_metrics = {}
    episodes = int(saved_metrics.get("episodes", 0))
    deaths = int(saved_metrics.get("deaths", 0))
    victories = int(saved_metrics.get("victories", 0))
    best_world_x = int(saved_metrics.get("best_x", 0))
    learner_status: dict[str, int | float | None] = {"steps": 0, "optimizer_updates": 0,
                                                      "replay_transitions": 0, "epsilon": 0.0,
                                                      "latest_loss": None}
    collected_steps = 0
    request_id = 0
    last_observation_at = {worker.worker_id: time.monotonic() for worker in workers}
    next_health_check_at = time.monotonic()  # Produce one snapshot at launch, then every 10 minutes.
    active = True

    def stop(*_: object) -> None:
        nonlocal active
        active = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while active and collected_steps < arguments.steps:
            ready: list[tuple[FileWorker, object]] = []
            for worker in workers:
                observation = worker.next_observation()
                if observation is None:
                    continue
                last_observation_at[worker.worker_id] = time.monotonic()
                ready.append((worker, observation))

            if time.monotonic() >= next_health_check_at:
                learner_inbox.put(("status",))
                try:
                    kind, learner_status = learner_outbox.get(timeout=5)
                    if kind != "status":
                        raise RuntimeError("learner returned an unexpected health response")
                except Exception as error:
                    learner_status = {**learner_status, "latest_loss": None, "learner_error": str(error)}
                write_health_report(arguments.run_dir, repository_root, last_observation_at, arguments.workers, {
                    "steps": learner_status["steps"],
                    "optimizer_updates": learner_status["optimizer_updates"],
                    "replay_transitions": learner_status["replay_transitions"],
                    "episodes": episodes,
                    "deaths": deaths,
                    "victories": victories,
                    "best_x": best_world_x,
                    "epsilon": learner_status["epsilon"],
                    "latest_loss": learner_status["latest_loss"],
                })
                next_health_check_at = time.monotonic() + HEALTH_CHECK_INTERVAL_SECONDS

            # Replay updates mutate one shared model, so they must stay
            # ordered.  The FCEUX instances themselves continue in parallel.
            actionable: list[tuple[FileWorker, object]] = []
            for worker, observation in ready:
                best_world_x = max(best_world_x, observation.world_x)
                if worker.previous is not None:
                    learner_inbox.put(("transition", worker.worker_id, worker.previous.state, worker.previous_action,
                                        shaped_reward(worker.previous, observation), observation.state,
                                        observation.terminal))
                    collected_steps += 1
                if observation.terminal:
                    episodes += 1
                    if observation.reason == "victory":
                        victories += 1
                    else:
                        deaths += 1
                    worker.reset(observation)
                    worker.previous = None
                    continue
                actionable.append((worker, observation))

            # The collector only exchanges observations/actions. A separate
            # learner process owns replay, model inference, and optimization.
            if actionable:
                states = np.stack([observation.state for _, observation in actionable])
                request_id += 1
                learner_inbox.put(("act", request_id, states, True))
                kind, response_id, actions, action_matrix, encoder_matrix, learner_status = learner_outbox.get(timeout=10)
                if kind != "act" or response_id != request_id:
                    raise RuntimeError("learner action response did not match collector request")
            else:
                action_matrix = np.empty((0, configuration.action_count), dtype=np.float32)
                encoder_matrix = np.empty((0, 16), dtype=np.float32)
                actions = np.empty((0,), dtype=np.int64)
            for index, (worker, observation) in enumerate(actionable):
                action_values = action_matrix[index]
                encoder_summary = encoder_matrix[index]
                action = int(actions[index])
                worker.send_action(observation, action)
                # FCEUX reads this optional HUD message.  It contains only
                # inspectable learner telemetry; controller input still comes
                # from command.json.
                if observation.sequence % 4 == 0:
                    metrics = {"episodes": episodes, "deaths": deaths, "victories": victories,
                               "best_x": best_world_x}
                    atomic_write_json(metrics_path, metrics)
                    atomic_write_json(worker.directory / "hud.json", {
                        "sequence": observation.sequence,
                        "steps": learner_status["steps"],
                        "updates": learner_status["optimizer_updates"],
                        "replay": learner_status["replay_transitions"],
                        "epsilon": 0.0,
                        **metrics,
                        "action": action,
                        "values": [round(float(value), 3) for value in action_values],
                        "grid": [int(value) for value in observation.state[:169]],
                        "globals": [round(float(value), 3) for value in observation.state[169:]],
                        "hidden": [round(float(value), 3) for value in encoder_summary],
                        "loss": learner_status["latest_loss"],
                    })
                worker.previous = observation
                worker.previous_action = action
            time.sleep(0.001)
    finally:
        learner_inbox.put(("stop",))
        learner.join(timeout=90)
        if learner.is_alive():
            learner.kill()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()


if __name__ == "__main__":
    main()
