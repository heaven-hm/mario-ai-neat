"""Train plain DDQN or PPO baselines through the shared SMB1/FCEUX bridge.

Example:
    python -m mario_ai_fceux.baseline_train --algorithm ddqn --rom SuperMarioBros.nes \
      --fceux /path/to/fceux --workers 8 --worlds 1,2,3,4,5,6,7,8 --run-dir runs/ddqn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import BaselineConfig, DDQNAgent, PPOAgent
from .environment import START_PROTOCOL, FileWorker, launch_fceux_workers
from .train import shaped_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train plain DDQN or PPO in FCEUX on SMB1.")
    parser.add_argument("--algorithm", choices=("ddqn", "ppo"), required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--fceux", default="fceux")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worlds", default="1")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--ppo-rollout-steps", type=int, default=2_048,
                        help="PPO applies updates only after every worker completed an episode.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _world_assignments(raw_worlds: str, worker_count: int) -> tuple[int, ...]:
    try:
        worlds = tuple(int(value.strip()) for value in raw_worlds.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("--worlds must be comma-separated integers in 1..8") from error
    if len(worlds) == 1:
        worlds *= worker_count
    if len(worlds) != worker_count or any(world < 1 or world > 8 for world in worlds):
        raise ValueError("--worlds must contain one SMB1 world from 1..8 per worker")
    return worlds


def _source_revision(repository_root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    options = parse_arguments()
    if options.workers < 1 or options.steps < 1 or options.checkpoint_every < 1:
        raise ValueError("workers, steps, and checkpoint interval must be positive")
    worlds = _world_assignments(options.worlds, options.workers)
    options.run_dir.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[2]
    fceux_binary = Path(shutil.which(options.fceux) or options.fceux).resolve()
    algorithm_class = DDQNAgent if options.algorithm == "ddqn" else PPOAgent
    configuration = BaselineConfig(seed=options.seed)
    agent = algorithm_class(configuration, options.device)
    checkpoint = options.run_dir / "model.pt"
    if options.resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"cannot resume; missing checkpoint: {checkpoint}")
        agent.load(checkpoint)
        logger.info("Resumed %s from %s at %d decisions", options.algorithm, checkpoint, agent.steps)
    metadata = {
        "algorithm": agent.algorithm,
        "architecture": "single-learner-multi-FCEUX-baseline",
        "training_actors": options.workers,
        "worlds": worlds,
        "seed": options.seed,
        "evaluation_seed": options.seed,
        "evaluation_mode": "greedy_no_learning",
        "eval_world": worlds[0],
        "eval_level": 1,
        "eval_episodes": 10,
        "action_repeat_frames": 12,
        "start_protocol": START_PROTOCOL,
        "rom_sha256": hashlib.sha256(options.rom.read_bytes()).hexdigest(),
        "fceux_executable": str(fceux_binary),
        "fceux_sha256": hashlib.sha256(fceux_binary.read_bytes()).hexdigest(),
        "source_revision": _source_revision(repository_root),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "config": vars(configuration),
        "ppo_barrier": "one terminal episode per FCEUX worker before policy update" if options.algorithm == "ppo" else None,
    }
    (options.run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    bridge = repository_root / "python" / "fceux_bridge" / "mario_ai_fceux_bridge.lua"
    processes = launch_fceux_workers(options.fceux, options.rom, bridge, options.run_dir,
                                     options.workers, worlds)
    workers = [FileWorker(f"worker-{index:02d}", options.run_dir / f"worker-{index:02d}")
               for index in range(options.workers)]
    pending: dict[str, tuple[object, int, float | None, float | None]] = {}
    paused = {worker.worker_id: False for worker in workers}
    paused_observations: dict[str, object] = {}
    last_checkpoint_step = agent.steps
    episode_count = 0
    victory_count = 0
    active = True

    def stop(*_: object) -> None:
        nonlocal active
        active = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while active and agent.steps < options.steps:
            ready_to_act: dict[str, object] = {}
            observed_any = False
            for worker in workers:
                observation = worker.next_observation()
                if observation is None:
                    continue
                observed_any = True
                previous = pending.pop(worker.worker_id, None)
                if previous is not None:
                    previous_observation, previous_action, old_log_probability, old_value = previous
                    reward = shaped_reward(previous_observation, observation)
                    if options.algorithm == "ddqn":
                        agent.observe(previous_observation.state, previous_action, reward,
                                      observation.state, observation.terminal)
                        agent.learn()
                    else:
                        agent.observe(worker.worker_id, previous_observation.state, previous_action,
                                      reward, float(old_log_probability), float(old_value),
                                      observation.terminal)
                if observation.terminal:
                    episode_count += 1
                    victory_count += int(observation.reason == "victory")
                    worker.reset(observation)
                    if options.algorithm == "ppo":
                        paused[worker.worker_id] = True
                    continue
                if options.algorithm == "ppo" and paused[worker.worker_id]:
                    paused_observations[worker.worker_id] = observation
                    worker.hold(observation)
                else:
                    ready_to_act[worker.worker_id] = observation

            if options.algorithm == "ppo" and all(paused.values()) \
                    and len(paused_observations) == len(workers):
                loss = agent.learn()
                if loss is not None:
                    logger.info("PPO update=%d decisions=%d loss=%.5f episodes=%d wins=%d",
                                agent.updates, agent.steps, loss, episode_count, victory_count)
                # A hold command was already sent for each stored observation.
                # Wait for the next observation after each savestate reload
                # before issuing actions under the updated policy.
                paused_observations.clear()
                for worker_id in paused:
                    paused[worker_id] = False

            for worker in workers:
                observation = ready_to_act.get(worker.worker_id)
                if observation is None:
                    continue
                if options.algorithm == "ddqn":
                    action = int(agent.select_actions(np.asarray([observation.state]), explore=True)[0])
                    old_log_probability = old_value = None
                else:
                    action, old_log_probability, old_value = agent.act_with_statistics(observation.state)
                worker.send_action(observation, action)
                pending[worker.worker_id] = (observation, action, old_log_probability, old_value)

            if agent.steps - last_checkpoint_step >= options.checkpoint_every:
                # PPO only checkpoints after a policy update, never mid-rollout.
                if options.algorithm == "ddqn" or not agent.rollout:
                    agent.save(checkpoint)
                    last_checkpoint_step = agent.steps
                    logger.info("Saved %s at %d transitions", checkpoint, agent.steps)
            if not observed_any:
                time.sleep(0.002)
        agent.save(checkpoint)
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
    logger.info("Stopped %s: decisions=%d updates=%d episodes=%d victories=%d",
                options.algorithm, agent.steps, agent.updates, episode_count, victory_count)


if __name__ == "__main__":
    main()
