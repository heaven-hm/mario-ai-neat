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
import csv
from dataclasses import replace
from multiprocessing.queues import Queue
from pathlib import Path

import numpy as np
import torch

from .actions import ACTION_COUNT, ACTION_NAMES, decode_action, greedy_action
from .environment import FileWorker, NoProgressTracker, Observation
from .model import RainbowNetwork
from .protocol import atomic_write_json
from .apex_actor import _load_weights_from_bytes

logger = logging.getLogger(__name__)


def build_benchmark_report(episodes: list[dict], run_metadata: dict,
                           evaluation_number: int, requested_episodes: int | None = None,
                           timestamp: str | None = None) -> dict:
    """Convert one frozen-policy episode batch to the shared benchmark schema."""
    victories = sum(episode.get("reason") == "victory" for episode in episodes)
    finished = len(episodes)
    details = []
    for index, episode in enumerate(episodes, 1):
        details.append({
            "episode": index,
            "reason": episode.get("reason", "timeout"),
            "max_x": int(episode.get("max_x", 0)),
            "terminal_x": int(episode.get("terminal_x", episode.get("max_x", 0))),
            "action_decisions": int(episode.get("action_decisions", 0)),
            "elapsed_seconds": float(episode.get("elapsed_seconds", episode.get("duration", 0.0))),
        })
    return {
        "algorithm": run_metadata.get("algorithm", "Ape-X Rainbow"),
        "evaluation_number": evaluation_number,
        "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "world": run_metadata.get("eval_world"),
        "level": 1,
        "evaluation_mode": "greedy_no_learning",
        "evaluation_seed": int(run_metadata.get("evaluation_seed", run_metadata.get("seed", 0))),
        "rom_sha256": run_metadata.get("rom_sha256"),
        "fceux_sha256": run_metadata.get("fceux_sha256"),
        "fceux_executable": run_metadata.get("fceux_executable"),
        "action_repeat_frames": run_metadata.get("action_repeat_frames"),
        "start_protocol": run_metadata.get("start_protocol"),
        "episodes_requested": int(requested_episodes if requested_episodes is not None
                                  else run_metadata.get("eval_episodes", finished)),
        "episodes_finished": finished,
        "victories": victories,
        "completion_rate": victories / finished if finished else 0.0,
        "episodes": details,
        "source_revision": run_metadata.get("source_revision"),
    }


def write_benchmark_report(run_directory: Path, report: dict) -> Path:
    """Persist a machine-readable JSON report and matching per-episode CSV."""
    destination = run_directory / "evaluations" / f"eval-{report['evaluation_number']:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "results.json"
    atomic_write_json(result_path, report)
    csv_path = destination / "episodes.csv"
    fields = ("episode", "reason", "max_x", "terminal_x", "action_decisions", "elapsed_seconds")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["episodes"])
    return result_path


def write_action_traces(run_directory: Path, evaluation_number: int,
                        episodes: list[dict]) -> Path:
    """Save compact state/action traces so repeatable failures are diagnosable."""
    destination = run_directory / "evaluations" / f"eval-{evaluation_number:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    trace_path = destination / "action-trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for episode_index, episode in enumerate(episodes, 1):
            for decision in episode.get("action_trace", []):
                handle.write(json.dumps({"episode": episode_index, **decision},
                                        separators=(",", ":")) + "\n")
    return trace_path


def evaluation_rank(report: dict) -> tuple[int, float, int]:
    """Rank policies by wins first, then average and worst-case level progress."""
    episode_rows = report.get("episodes", [])
    minimum_x = min((int(row.get("max_x", 0)) for row in episode_rows), default=0)
    return (int(report.get("victories", 0)),
            float(report.get("avg_max_x", 0.0)), minimum_x)


def save_best_policy(run_directory: Path, weight_bytes: bytes, report: dict) -> bool:
    """Keep the best measured greedy policy apart from the learner checkpoint."""
    metadata_path = run_directory / "best_policy_eval.json"
    weights_path = run_directory / "best_policy_weights.pt"
    try:
        previous_report = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous_report = None
    if previous_report is not None and evaluation_rank(report) <= evaluation_rank(previous_report):
        return False

    temporary_weights = weights_path.with_suffix(weights_path.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    try:
        temporary_weights.write_bytes(weight_bytes)
        temporary_metadata.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary_weights.replace(weights_path)
        temporary_metadata.replace(metadata_path)
    finally:
        temporary_weights.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return True


def eval_worker_main(
    worker: FileWorker,
    weight_queue: Queue,         # receives ("weights", bytes) from learner
    run_directory: str,
    observation_size: int = 184,
    action_count: int = ACTION_COUNT,
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
    loaded_weight_bytes: bytes | None = None
    last_eval_at = 0.0
    prior_evaluations = [path for path in (run_dir / "evaluations").glob("eval-*")
                         if path.is_dir() and path.name.removeprefix("eval-").isdigit()]
    eval_count = max((int(path.name.removeprefix("eval-")) for path in prior_evaluations), default=0)

    def _load_latest_weights() -> bool:
        nonlocal weights_loaded, loaded_weight_bytes
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
                loaded_weight_bytes = latest
            except Exception as exc:
                logger.warning("Eval weight load failed: %s", exc)
        return weights_loaded

    def _greedy_action(state: np.ndarray) -> tuple[int, np.ndarray]:
        obs = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = network(obs, support)
        values = q_values[0].cpu().numpy()
        return greedy_action(values), values

    def _run_eval_episode() -> dict:
        """Run one full greedy episode; return episode stats."""
        episode_start = time.monotonic()
        max_x = 0
        decisions = 0
        result = {"reason": "timeout", "max_x": 0, "terminal_x": 0,
                  "elapsed_seconds": 0.0, "action_decisions": 0,
                  "action_trace": []}
        action_trace = result["action_trace"]
        last_observation: Observation | None = None
        progress_tracker = NoProgressTracker()
        previous_action_frames = 0
        while time.monotonic() - episode_start < max_episode_seconds:
            obs = worker.next_observation()
            if obs is None:
                time.sleep(0.002)
                continue
            last_observation = obs
            max_x = max(max_x, obs.world_x)
            if not obs.terminal and progress_tracker.update(obs.world_x, previous_action_frames):
                obs = replace(obs, terminal=True, reason="stuck")
            if obs.terminal:
                result = {
                    "reason": obs.reason,
                    "max_x": max_x,
                    "terminal_x": obs.world_x,
                    "elapsed_seconds": round(time.monotonic() - episode_start, 3),
                    "action_decisions": decisions,
                    "action_trace": action_trace,
                }
                worker.reset(obs)
                return result
            action, q_values = _greedy_action(obs.state)
            action_base, duration_frames = decode_action(action)
            action_trace.append({
                "decision": decisions,
                "world_x": obs.world_x,
                "action_id": action,
                "action": ACTION_NAMES[action],
                "action_base": action_base,
                "duration_frames": duration_frames,
                "q_value": round(float(q_values[action]), 5),
                "top_actions": [
                    {"name": ACTION_NAMES[index], "q": round(float(q_values[index]), 5)}
                    for index in np.argsort(q_values)[-3:][::-1]
                ],
                "speed_x": round(float(obs.state[169]), 4),
                "speed_y": round(float(obs.state[170]), 4),
                "grounded": bool(obs.state[171] > 0),
                "enemy_dx": round(float(obs.state[174]), 4),
                "enemy_dy": round(float(obs.state[175]), 4),
                "gap_ahead": bool(obs.state[182] > 0),
            })
            worker.send_action(obs, action)
            previous_action_frames = duration_frames
            decisions += 1
        # A timed-out evaluation must not contaminate the next episode.
        if last_observation is not None:
            worker.reset(last_observation)
        result.update({"max_x": max_x, "terminal_x": max_x, "action_decisions": decisions,
                       "elapsed_seconds": round(time.monotonic() - episode_start, 3),
                       "action_trace": action_trace})
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
            benchmark_report = build_benchmark_report(episodes, run_metadata, eval_count,
                                                     requested_episodes=episodes_per_eval)
            write_benchmark_report(run_dir, benchmark_report)
            trace_path = write_action_traces(run_dir, eval_count, episodes)
            best_policy = bool(loaded_weight_bytes and save_best_policy(
                run_dir, loaded_weight_bytes, benchmark_report,
            ))
            record = {**benchmark_report, "eval": eval_count, "episodes": len(episodes),
                      "win_rate": round(win_rate, 4), "avg_max_x": round(avg_x, 1),
                      "best_policy_updated": best_policy,
                      "episode_details": benchmark_report["episodes"]}
            with eval_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            # Write a human-readable latest snapshot too.
            atomic_write_json(
                run_dir / "eval_latest.json",
                {k: v for k, v in record.items() if k not in ("episode_details", "episodes")},
            )
            logger.info(
                "Eval #%d | episodes=%d victories=%d win_rate=%.2f avg_x=%.0f trace=%s",
                eval_count, len(episodes), victories, win_rate, avg_x, trace_path,
            )
            last_eval_at = time.monotonic()
        else:
            time.sleep(1.0)
