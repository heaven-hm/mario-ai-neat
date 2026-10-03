"""Greedy, no-learning FCEUX evaluation for Rainbow, DDQN, or PPO."""

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

from .actions import ACTION_NAMES, decode_action
from .agent import AgentConfig, RainbowAgent
from .baselines import load_baseline_policy
from .environment import START_PROTOCOL, FileWorker, launch_fceux_workers
from .replay import PrioritizedReplayBuffer

ACTION_REPEAT_FRAMES = 12
EPISODE_CSV_FIELDS = ("episode", "reason", "max_x", "terminal_x",
                      "action_decisions", "elapsed_seconds")
# Feature layout of the bridge's 184-dim state. Field names match the trace
# apex_eval writes, so one analysis reads both.
STATE_SPEED_X = 169
STATE_SPEED_Y = 170
STATE_GROUNDED = 171
STATE_ENEMY_DX = 174
STATE_ENEMY_DY = 175
STATE_GAP_AHEAD = 182


def build_trace_record(episode: int, decision: int, state: np.ndarray,
                       world_x: int, action: int, q_values: np.ndarray) -> dict[str, object]:
    """One per-decision trace row, shaped like apex_eval's action-trace entries.

    Death x positions alone cannot distinguish a pit death from an enemy death;
    these fields settle it (see the project lesson on the 1-1 death wall).
    """
    action_base, duration_frames = decode_action(action)
    return {
        "episode": episode,
        "decision": decision,
        "world_x": int(world_x),
        "action_id": int(action),
        "action": ACTION_NAMES[action],
        "action_base": action_base,
        "duration_frames": duration_frames,
        "q_value": round(float(q_values[action]), 5),
        "top_actions": [
            {"name": ACTION_NAMES[index], "q": round(float(q_values[index]), 5)}
            for index in np.argsort(q_values)[-3:][::-1]
        ],
        "speed_x": round(float(state[STATE_SPEED_X]), 4),
        "speed_y": round(float(state[STATE_SPEED_Y]), 4),
        "grounded": bool(state[STATE_GROUNDED] > 0),
        "enemy_dx": round(float(state[STATE_ENEMY_DX]), 4),
        "enemy_dy": round(float(state[STATE_ENEMY_DY]), 4),
        "gap_ahead": bool(state[STATE_GAP_AHEAD] > 0),
    }


def write_episode_csv(path: Path, episodes: list[dict[str, object]]) -> None:
    """Write every evaluation field without rejecting rich episode records."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(episodes)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved SMB1 policy without exploration or learning.")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--fceux", default="fceux")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-seconds", type=int, default=1_800)
    parser.add_argument("--evaluation-seed", type=int, default=2026,
                        help="Seed recorded for each policy's identical clean-start evaluation protocol.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--trace-actions", action="store_true",
                        help="Write action-trace.jsonl with the per-decision state, action and "
                             "Q values used for death-cause analysis. Off by default so the "
                             "measured selection path is unchanged.")
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
    algorithm = str(payload.get("algorithm", ""))
    if algorithm.startswith("Rainbow") or algorithm.startswith("Ape-X Rainbow"):
        action_profile = "rainbow"
        configuration = AgentConfig(**payload["config"])
        agent = RainbowAgent(
            PrioritizedReplayBuffer(configuration.observation_size, capacity=1, seed=configuration.seed),
            configuration, options.device)
        agent.load(checkpoint, validate_replay=False, restore_rng=False)
    else:
        action_profile = "legacy"
        algorithm, agent = load_baseline_policy(checkpoint, options.device)
    repository_root = Path(__file__).resolve().parents[2]
    evaluation_directory = options.run_dir / "evaluations" / time.strftime("%Y%m%d-%H%M%S")
    evaluation_directory.mkdir(parents=True)
    bridge = repository_root / "python" / "fceux_bridge" / "mario_ai_fceux_bridge.lua"
    processes = launch_fceux_workers(
        options.fceux, options.rom, bridge, evaluation_directory, 1, (options.world,),
        action_profile=action_profile,
        extra_args=("--xscale", "1", "--yscale", "1", "-qwindowgeometry", "512x469+851+205"),
    )
    worker = FileWorker("evaluation", evaluation_directory / "worker-00", action_profile=action_profile)
    episodes: list[dict[str, object]] = []
    maximum_x = 0
    episode_decisions = 0
    episode_started_at = time.monotonic()
    deadline = time.monotonic() + options.max_seconds
    trace_records: list[dict[str, object]] = []
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
            if options.trace_actions:
                # The action comes from the deployed selection path; inspect()
                # only supplies the Q row recorded beside it.
                action = int(agent.select_actions(np.asarray([observation.state]), explore=False)[0])
                q_row, _ = agent.inspect(np.asarray([observation.state]))
                q_row = q_row[0]
                trace_records.append(build_trace_record(
                    len(episodes) + 1, episode_decisions, observation.state,
                    observation.world_x, action, q_row))
            else:
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
    try:
        run_metadata = json.loads((options.run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        run_metadata = {}
    report = {"algorithm": algorithm, "checkpoint": str(checkpoint), "world": options.world, "level": 1,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "source_revision": run_metadata.get("source_revision"),
              "evaluation_mode": "greedy_no_learning",
              "selection_policy": "agent.select_actions (greedy_action tie-break when action_count==21; NO safe_start, NO pit_edge_commit) -- verified against agent.py 2026-10-03, the argmax branch is legacy-6-action only",
              "evaluation_seed": options.evaluation_seed,
              "rom_sha256": rom_digest, "fceux_executable": str(fceux_path),
              "fceux_sha256": fceux_digest,
              "action_repeat_frames": ACTION_REPEAT_FRAMES,
              "start_protocol": START_PROTOCOL,
              "episodes_requested": options.episodes, "episodes_finished": len(episodes), "victories": victories,
              "completion_rate": victories / len(episodes) if episodes else 0.0, "episodes": episodes}
    (evaluation_directory / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_episode_csv(evaluation_directory / "episodes.csv", episodes)
    if options.trace_actions:
        with (evaluation_directory / "action-trace.jsonl").open("w", encoding="utf-8") as handle:
            for record in trace_records:
                handle.write(json.dumps(record) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
