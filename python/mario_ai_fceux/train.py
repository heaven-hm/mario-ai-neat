"""Command-line trainer that runs multiple FCEUX workers and one Python learner."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from .agent import AgentConfig, RainbowLiteAgent
from .environment import FileWorker, launch_fceux_workers
from .replay import ReplayDatabase


def shaped_reward(previous, current) -> float:
    """Dense but bounded reward from real SMB1 progress, power, and terminal result."""
    reward = max(-1.0, min(1.0, (current.world_x - previous.world_x) / 24.0))
    reward += max(-0.2, min(0.2, (current.power - previous.power) * 0.1))
    if current.terminal:
        reward += 2.0 if current.reason == "victory" else -1.0
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
    bridge_template = repository_root / "fceux_bridge" / "mario_ai_fceux_bridge.lua"
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
    active = True

    def stop(*_: object) -> None:
        nonlocal active
        active = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while active and agent.steps < arguments.steps:
            for worker in workers:
                observation = worker.next_observation()
                if observation is None:
                    continue
                if worker.previous is not None:
                    agent.observe(worker.worker_id, worker.previous.state, worker.previous_action,
                                  shaped_reward(worker.previous, observation), observation.state, observation.terminal)
                    agent.learn()
                if observation.terminal:
                    worker.reset(observation)
                    worker.previous = None
                    continue
                action = int(agent.select_actions(observation.state[None, :], explore=True)[0])
                worker.send_action(observation, action)
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
