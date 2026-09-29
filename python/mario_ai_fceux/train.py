"""Command-line trainer that runs multiple FCEUX workers and one Python learner."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from .agent import AgentConfig, RainbowLiteAgent
from .environment import FileWorker, launch_fceux_workers
from .protocol import atomic_write_json
from .replay import ReplayDatabase


def shaped_reward(previous, current) -> float:
    """Reward forward SMB1 progress and make deaths materially undesirable."""
    reward = max(-2.0, min(2.0, (current.world_x - previous.world_x) / 16.0))
    reward += max(-0.2, min(0.2, (current.power - previous.power) * 0.1))
    if current.terminal:
        reward += 20.0 if current.reason == "victory" else -5.0
    return reward


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Rainbow-lite SMB1 controller through FCEUX Lua workers.")
    parser.add_argument("--rom", type=Path, required=True, help="Path to a legally obtained SMB1 NES ROM.")
    parser.add_argument("--fceux", default="fceux", help="FCEUX executable path or command.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/python-rainbow"))
    parser.add_argument("--workers", type=int, default=4, help="Parallel FCEUX processes.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total action decisions to collect.")
    parser.add_argument("--resume", action="store_true", help="Load model.pt if it exists in the run directory.")
    parser.add_argument("--device", default=None, help="PyTorch device: mps, cuda, or cpu.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repository_root = Path(__file__).resolve().parents[2]
    bridge_template = repository_root / "python" / "fceux_bridge" / "mario_ai_fceux_bridge.lua"
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    replay = ReplayDatabase(arguments.run_dir / "replay.sqlite3", observation_size=184)
    agent = RainbowLiteAgent(replay, AgentConfig(), device=arguments.device)
    checkpoint = arguments.run_dir / "model.pt"
    if arguments.resume and checkpoint.exists():
        agent.load(checkpoint)
    metadata = {"algorithm": "Rainbow-lite Double DQN", "workers": arguments.workers,
                "observation_size": 184, "actions": 6, "rom": str(arguments.rom)}
    (arguments.run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    processes = launch_fceux_workers(arguments.fceux, arguments.rom, bridge_template,
                                     arguments.run_dir, arguments.workers)
    workers = [FileWorker(f"worker-{index:02d}", arguments.run_dir / f"worker-{index:02d}")
               for index in range(arguments.workers)]
    latest_loss: float | None = None
    metrics_path = arguments.run_dir / "training_metrics.json"
    try:
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        saved_metrics = {}
    episodes = int(saved_metrics.get("episodes", 0))
    deaths = int(saved_metrics.get("deaths", 0))
    victories = int(saved_metrics.get("victories", 0))
    best_world_x = int(saved_metrics.get("best_x", 0))
    active = True

    def stop(*_: object) -> None:
        nonlocal active
        active = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while active and agent.steps < arguments.steps:
            ready: list[tuple[FileWorker, object]] = []
            for worker in workers:
                observation = worker.next_observation()
                if observation is None:
                    continue
                ready.append((worker, observation))

            # Replay updates mutate one shared model, so they must stay
            # ordered.  The FCEUX instances themselves continue in parallel.
            actionable: list[tuple[FileWorker, object]] = []
            for worker, observation in ready:
                best_world_x = max(best_world_x, observation.world_x)
                if worker.previous is not None:
                    agent.observe(worker.worker_id, worker.previous.state, worker.previous_action,
                                  shaped_reward(worker.previous, observation), observation.state, observation.terminal)
                    learned_loss = agent.learn()
                    if learned_loss is not None:
                        latest_loss = learned_loss
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

            # Every ready worker gets one shared batched inference call.  This
            # avoids serial per-window neural-network evaluation.
            if actionable:
                states = np.stack([observation.state for _, observation in actionable])
                action_matrix, encoder_matrix = agent.inspect(states)
                actions = agent.choose_actions(action_matrix, states, explore=True)
            else:
                action_matrix = np.empty((0, agent.config.action_count), dtype=np.float32)
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
                        "steps": agent.steps,
                        "updates": agent.optimizer_steps,
                        "replay": len(replay),
                        "epsilon": round(agent.epsilon, 4),
                        **metrics,
                        "action": action,
                        "values": [round(float(value), 3) for value in action_values],
                        "grid": [int(value) for value in observation.state[:169]],
                        "globals": [round(float(value), 3) for value in observation.state[169:]],
                        "hidden": [round(float(value), 3) for value in encoder_summary],
                        "loss": round(latest_loss, 4) if latest_loss is not None else None,
                    })
                worker.previous = observation
                worker.previous_action = action
            if agent.steps and agent.steps % 2_000 == 0:
                agent.save(checkpoint)
            time.sleep(0.001)
    finally:
        agent.save(checkpoint)
        replay.close()
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
