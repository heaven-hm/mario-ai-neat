"""Evaluation worker for Ape-X Rainbow.

Runs as a separate lightweight process (no learning, no training):
- Exploration is OFF (greedy evaluation).
- Weights are loaded from the learner's weight broadcast queue.
- Periodically runs N evaluation episodes and writes results to disk.

This gives unbiased performance metrics independent of the training epsilon.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from multiprocessing.queues import Queue
from pathlib import Path

import numpy as np
import torch

from .environment import FileWorker, Observation
from .model import RainbowNetwork
from .protocol import atomic_write_json
from .apex_actor import _load_weights_from_bytes

logger = logging.getLogger(__name__)


def eval_worker_main(
    worker: FileWorker,
    weight_queue: Queue,         # receives ("weights", bytes) from learner
    run_directory: str,
    observation_size: int = 184,
    action_count: int = 6,
    atom_count: int = 51,
    value_min: float = -20.0,
    value_max: float = 20.0,
    eval_every_seconds: float = 120.0,   # run an eval batch every N seconds
    episodes_per_eval: int = 5,
    max_episode_seconds: float = 300.0,
    device_str: str | None = None,
) -> None:
    """Evaluation process: greedy policy, frozen weights, periodic eval runs."""

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    device = torch.device(
        device_str or ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    )
    support = torch.linspace(value_min, value_max, atom_count, device=device)
    network = RainbowNetwork(observation_size, action_count, atom_count).to(device)
    network.eval()

    run_dir = Path(run_directory)
    try:
        run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        run_metadata = {}
    eval_log = run_dir / "eval_results.jsonl"

    weights_loaded = False
    last_eval_at = 0.0
    eval_count = 0

    def _load_latest_weights() -> bool:
        nonlocal weights_loaded
        latest: bytes | None = None
        while True:
            try:
                _, weight_bytes = weight_queue.get_nowait()
                latest = weight_bytes
            except Exception:
                break
        if latest is not None:
            try:
                _load_weights_from_bytes(network, latest, device, evaluation=True)
                weights_loaded = True
            except Exception as exc:
                logger.warning("Eval weight load failed: %s", exc)
        return weights_loaded

    def _greedy_action(state: np.ndarray) -> int:
        obs = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = network(obs, support)
        return int(q_values.argmax(dim=1).item())

    def _run_eval_episode() -> dict:
        """Run one full greedy episode; return episode stats."""
        episode_start = time.monotonic()
        max_x = 0
        decisions = 0
        result = {"reason": "timeout", "max_x": 0, "duration": 0.0,
                  "action_decisions": 0}
        last_observation: Observation | None = None
        while time.monotonic() - episode_start < max_episode_seconds:
            obs = worker.next_observation()
            if obs is None:
                time.sleep(0.002)
                continue
            last_observation = obs
            max_x = max(max_x, obs.world_x)
            if obs.terminal:
                result = {
                    "reason": obs.reason,
                    "max_x": max_x,
                    "duration": round(time.monotonic() - episode_start, 2),
                    "action_decisions": decisions,
                }
                worker.reset(obs)
                return result
            action = _greedy_action(obs.state)
            worker.send_action(obs, action)
            decisions += 1
        # A timed-out evaluation must not contaminate the next episode.
        if last_observation is not None:
            worker.reset(last_observation)
        result.update({"max_x": max_x, "action_decisions": decisions,
                       "duration": round(time.monotonic() - episode_start, 2)})
        return result

    logger.info("Eval worker started")

    while True:
        _load_latest_weights()

        now = time.monotonic()
        if weights_loaded and (now - last_eval_at) >= eval_every_seconds:
            eval_count += 1
            episodes = [_run_eval_episode() for _ in range(episodes_per_eval)]
            victories = sum(e["reason"] == "victory" for e in episodes)
            avg_x = sum(e["max_x"] for e in episodes) / max(1, len(episodes))
            win_rate = victories / max(1, len(episodes))
            record = {
                "eval": eval_count,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "episodes": len(episodes),
                "episodes_requested": episodes_per_eval,
                "episodes_finished": len(episodes),
                "victories": victories,
                "win_rate": round(win_rate, 4),
                "completion_rate": round(win_rate, 4),
                "avg_max_x": round(avg_x, 1),
                "world": run_metadata.get("eval_world"),
                "rom_sha256": run_metadata.get("rom_sha256"),
                "fceux_sha256": run_metadata.get("fceux_sha256"),
                "action_repeat_frames": run_metadata.get("action_repeat_frames"),
                "start_protocol": run_metadata.get("start_protocol"),
                "algorithm": "Rainbow C51 + NoisyNet + Double + Dueling + PER + n-step",
                "episode_details": episodes,
            }
            with eval_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            # Write a human-readable latest snapshot too.
            atomic_write_json(
                run_dir / "eval_latest.json",
                {k: v for k, v in record.items() if k != "episode_details"},
            )
            logger.info(
                "Eval #%d | episodes=%d victories=%d win_rate=%.2f avg_x=%.0f",
                eval_count, len(episodes), victories, win_rate, avg_x,
            )
            last_eval_at = time.monotonic()
        else:
            time.sleep(1.0)
