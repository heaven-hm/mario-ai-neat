"""Greedy, no-learning FCEUX evaluation for a saved Rainbow checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .agent import AgentConfig, RainbowAgent
from .environment import FileWorker, launch_fceux_workers
from .replay import PrioritizedReplayBuffer

ACTION_REPEAT_FRAMES = 12
EPISODE_CSV_FIELDS = ("episode", "reason", "max_x", "terminal_x",
                      "action_decisions", "elapsed_seconds")


def write_episode_csv(path: Path, episodes: list[dict[str, object]]) -> None:
    """Write every evaluation field without rejecting rich episode records."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(episodes)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Rainbow checkpoint without exploration or learning.")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--fceux", default="fceux")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-seconds", type=int, default=1_800)
    parser.add_argument("--evaluation-seed", type=int, default=2026,
                        help="Seed recorded for each policy's identical clean-start evaluation protocol.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    options = arguments()
    if not 1 <= options.world <= 8:
        raise ValueError("--world must be in 1..8")
    random.seed(options.evaluation_seed)
    np.random.seed(options.evaluation_seed)
    torch.manual_seed(options.evaluation_seed)
    checkpoint = options.run_dir / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configuration = AgentConfig(**payload["config"])
    agent = RainbowAgent(PrioritizedReplayBuffer(configuration.observation_size, capacity=1, seed=configuration.seed),
                         configuration, options.device)
    agent.load(checkpoint, validate_replay=False, restore_rng=False)
    repository_root = Path(__file__).resolve().parents[2]
    evaluation_directory = options.run_dir / "evaluations" / time.strftime("%Y%m%d-%H%M%S")
    evaluation_directory.mkdir(parents=True)
    bridge = repository_root / "python" / "fceux_bridge" / "mario_ai_fceux_bridge.lua"
    processes = launch_fceux_workers(options.fceux, options.rom, bridge, evaluation_directory, 1, (options.world,))
    worker = FileWorker("evaluation", evaluation_directory / "worker-00")
    episodes: list[dict[str, object]] = []
    maximum_x = 0
    episode_decisions = 0
    episode_started_at = time.monotonic()
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
                                 "max_x": maximum_x, "terminal_x": observation.world_x,
                                 "action_decisions": episode_decisions,
                                 "elapsed_seconds": round(time.monotonic() - episode_started_at, 3)})
                worker.reset(observation)
                maximum_x = 0
                episode_decisions = 0
                episode_started_at = time.monotonic()
                continue
            action = int(agent.select_actions(np.asarray([observation.state]), explore=False)[0])
            worker.send_action(observation, action)
            episode_decisions += 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except Exception:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
    victories = sum(episode["reason"] == "victory" for episode in episodes)
    rom_digest = hashlib.sha256(options.rom.read_bytes()).hexdigest()
    fceux_path = Path(shutil.which(options.fceux) or options.fceux).resolve()
    fceux_digest = hashlib.sha256(fceux_path.read_bytes()).hexdigest()
    report = {"algorithm": payload.get("algorithm"), "checkpoint": str(checkpoint), "world": options.world,
              "evaluation_mode": "greedy_no_learning",
              "evaluation_seed": options.evaluation_seed,
              "rom_sha256": rom_digest, "fceux_executable": str(fceux_path),
              "fceux_sha256": fceux_digest,
              "action_repeat_frames": ACTION_REPEAT_FRAMES,
              "start_protocol": "SMB1 title start and FCEUX training-slot restore",
              "episodes_requested": options.episodes, "episodes_finished": len(episodes), "victories": victories,
              "completion_rate": victories / len(episodes) if episodes else 0.0, "episodes": episodes}
    (evaluation_directory / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_episode_csv(evaluation_directory / "episodes.csv", episodes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
