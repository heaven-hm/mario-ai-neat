"""Greedy, no-learning FCEUX evaluation for a saved Rainbow checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .agent import AgentConfig, RainbowAgent
from .environment import FileWorker, launch_fceux_workers
from .replay import PrioritizedReplayBuffer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Rainbow checkpoint without exploration or learning.")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--fceux", default="fceux")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-seconds", type=int, default=1_800)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    options = arguments()
    if not 1 <= options.world <= 8:
        raise ValueError("--world must be in 1..8")
    checkpoint = options.run_dir / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    configuration = AgentConfig(**payload["config"])
    agent = RainbowAgent(PrioritizedReplayBuffer(configuration.observation_size, capacity=1, seed=configuration.seed),
                         configuration, options.device)
    agent.load(checkpoint)
    repository_root = Path(__file__).resolve().parents[2]
    evaluation_directory = options.run_dir / "evaluations" / time.strftime("%Y%m%d-%H%M%S")
    evaluation_directory.mkdir(parents=True)
    bridge = repository_root / "python" / "fceux_bridge" / "mario_ai_fceux_bridge.lua"
    processes = launch_fceux_workers(options.fceux, options.rom, bridge, evaluation_directory, 1, (options.world,))
    worker = FileWorker("evaluation", evaluation_directory / "worker-00")
    episodes: list[dict[str, object]] = []
    maximum_x = 0
    deadline = time.monotonic() + options.max_seconds
    try:
        while len(episodes) < options.episodes and time.monotonic() < deadline:
            observation = worker.next_observation()
            if observation is None:
                time.sleep(0.002)
                continue
            maximum_x = max(maximum_x, observation.world_x)
            if observation.terminal:
                episodes.append({"episode": len(episodes) + 1, "reason": observation.reason,
                                 "max_x": maximum_x, "terminal_x": observation.world_x})
                worker.reset(observation)
                maximum_x = 0
                continue
            action = int(agent.select_actions(np.asarray([observation.state]), explore=False)[0])
            worker.send_action(observation, action)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
    victories = sum(episode["reason"] == "victory" for episode in episodes)
    report = {"algorithm": payload.get("algorithm"), "checkpoint": str(checkpoint), "world": options.world,
              "episodes_requested": options.episodes, "episodes_finished": len(episodes), "victories": victories,
              "completion_rate": victories / len(episodes) if episodes else 0.0, "episodes": episodes}
    (evaluation_directory / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (evaluation_directory / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("episode", "reason", "max_x", "terminal_x"))
        writer.writeheader()
        writer.writerows(episodes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
