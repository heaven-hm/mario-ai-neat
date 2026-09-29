"""Ape-X Rainbow coordinator: launches actors, learner, and eval worker.

Architecture:
    8 FCEUX Actors ──► multiprocessing.Queue (experience) ──► GPU Learner
                                                                    │
                                                         (weights broadcast)
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    ▼                                               ▼
             Actor 1..8 weight queues                   Eval worker weight queue

Key design decisions vs. old train.py:
    - Actors and learner run in *separate processes* (truly async).
    - Actors push *batches* (default 32) to reduce IPC overhead.
    - Learner owns ALL replay and optimizer state — no shared mutable objects.
    - SQLite is NOT touched in the training hot path.
    - Weights are broadcast every N optimizer steps, not every frame.
    - Queue has a hard capacity limit for backpressure (default 512 batches).
    - Exploration: actor-specific epsilon-greedy, without a Mario action prior.
    - Separate eval worker runs greedy episodes every 2 minutes.

Usage:
    python -m mario_ai_fceux.apex_train \\
        --rom SuperMarioBros.nes \\
        --workers 8 \\
        --device mps \\
        --run-dir runs/apex-rainbow \\
        [--resume] \\
        [--replay-capacity 500000]
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import multiprocessing
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .agent import AgentConfig
from .apex_actor import ActorConfig, actor_main, _apex_epsilon
from .apex_eval import eval_worker_main
from .apex_learner import apex_learner_main
from .environment import START_PROTOCOL, FileWorker, launch_fceux_workers
from .protocol import atomic_write_json
from .train import read_lua_neat_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


HEALTH_CHECK_INTERVAL = 10 * 60   # seconds
WORKER_STALE_SECONDS = 3 * 60


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ape-X Rainbow for Super Mario Bros — distributed async RL."
    )
    parser.add_argument("--rom", type=Path, required=True,
                        help="Path to a legally obtained SMB1 NES ROM.")
    parser.add_argument("--fceux", default="fceux",
                        help="FCEUX executable path or command.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/apex-rainbow"))
    parser.add_argument("--workers", type=int, default=8,
                        help="Training actor count (Ape-X recommendation: 8).")
    parser.add_argument("--worlds", default="1",
                        help="Comma-separated SMB1 worlds (one per worker).")
    parser.add_argument("--steps", type=int, default=5_000_000,
                        help="Total training steps target.")
    parser.add_argument("--resume", action="store_true",
                        help="Load model.pt and replay.npz if they exist.")
    parser.add_argument("--device", default=None,
                        help="PyTorch device: mps, cuda, or cpu.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replay-capacity", type=int, default=500_000,
                        help="PER buffer size (250k–1M recommended).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Learner mini-batch size.")
    parser.add_argument("--actor-batch-size", type=int, default=32,
                        help="Transitions per actor queue push.")
    parser.add_argument("--weight-sync-every", type=int, default=500,
                        help="Broadcast weights every N learner optimizer steps.")
    parser.add_argument("--actor-weight-sync-every", type=int, default=400,
                        help="Actors reload weights every N collected steps.")
    parser.add_argument("--queue-capacity", type=int, default=512,
                        help="Max batches in experience queue (backpressure).")
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--learn-per-batch", type=int, default=4,
                        help="Optimizer updates per received experience batch.")
    parser.add_argument("--checkpoint-every", type=int, default=10_000,
                        help="Checkpoint every N optimizer steps.")
    parser.add_argument("--eval-every", type=float, default=120.0,
                        help="Seconds between greedy evaluation runs.")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Episodes per evaluation run.")
    parser.add_argument("--eval-world", type=int, default=1,
                        help="World for the greedy evaluation actor.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Health reporting (statistics only — no hot-path DB writes)
# ---------------------------------------------------------------------------

def write_health_report(
    run_dir: Path,
    repo_root: Path,
    learner_status: dict,
    actor_metrics: list[dict],
    eval_record: dict | None,
    expected_workers: int,
) -> None:
    """Write a health JSON + markdown snapshot every 10 minutes."""
    health_dir = run_dir / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "architecture": "Ape-X Rainbow (C51 + NoisyNet + Double + Dueling + PER + n-step)",
        "healthy": True,
        "repair_required": [],
        "learner": learner_status,
        "actors": actor_metrics,
        "expected_workers": expected_workers,
        "evaluation": eval_record,
        "disk": {
            "free_bytes": shutil.disk_usage(repo_root).free,
            "total_bytes": shutil.disk_usage(repo_root).total,
        },
    }
    lua_log = repo_root / "mario_ai_neat.log"
    lua_database = repo_root / "mario_ai_neat.db"
    lua_age = (round(max(0.0, time.time() - lua_log.stat().st_mtime), 1)
               if lua_log.exists() else None)
    lua_summary = read_lua_neat_summary(lua_database, lua_log)
    report["lua_neat"] = {
        "state": "active" if lua_age is not None and lua_age <= HEALTH_CHECK_INTERVAL else "stale_or_not_detected",
        "log_age_seconds": lua_age,
        **lua_summary,
    }
    actor_best_x = max((int(item.get("best_episode_x", 0)) for item in actor_metrics), default=0)
    actor_victories = sum(int(item.get("victories", 0)) for item in actor_metrics)
    report["measured_progress"] = {
        "python_transitions": learner_status.get("transitions_received", 0),
        "python_optimizer_updates": learner_status.get("optimizer_updates", 0),
        "python_replay_size": learner_status.get("replay_transitions", 0),
        "python_best_episode_x": actor_best_x,
        "python_victories": actor_victories,
    }
    atomic_write_json(health_dir / "latest.json", report)
    with (health_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report, separators=(",", ":")) + "\n")
    # Markdown summary.
    rows = [
        "# Ape-X Rainbow — Health Report",
        "",
        f"Generated: {report['checked_at']}",
        "",
        "## Learner",
        f"- Steps: {learner_status.get('steps', '?')}",
        f"- Optimizer updates: {learner_status.get('optimizer_updates', '?')}",
        f"- Replay size: {learner_status.get('replay_transitions', '?')}",
        f"- Latest loss: {learner_status.get('latest_loss', '?')}",
        "",
        "## Actors",
    ]
    for m in actor_metrics:
        rows.append(
            f"- Actor {m.get('actor', '?')}: ε={m.get('epsilon', '?')} "
            f"steps={m.get('steps', 0)} episodes={m.get('episodes', 0)} "
            f"victories={m.get('victories', 0)}"
        )
    if eval_record:
        rows += [
            "",
            "## Evaluation (greedy, frozen weights)",
            f"- Win rate: {eval_record.get('win_rate', '?')}",
            f"- Avg X: {eval_record.get('avg_max_x', '?')}",
            f"- Episodes: {eval_record.get('episodes', '?')}",
        ]
    rows += [
        "",
        "## Learning comparison",
        "| System | Latest evidence |",
        "| --- | --- |",
        (f"| Python Ape-X Rainbow | {learner_status.get('transitions_received', 0)} transitions; "
         f"{learner_status.get('optimizer_updates', 0)} updates; replay "
         f"{learner_status.get('replay_transitions', 0)}; best actor X {actor_best_x}; "
         f"victories {actor_victories} |"),
        (f"| Lua NEAT | generation {lua_summary.get('generation')}; "
         f"best fitness {lua_summary.get('best_fitness')}; "
         f"latest X {lua_summary.get('latest_max_x')}; log {report['lua_neat']['state']} |"),
        "",
        "Movement skill is not called learned until clean evaluation repeats it reliably.",
    ]
    (health_dir / "report.md").write_text("\n".join(rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    try:
        requested_worlds = tuple(int(v.strip()) for v in args.worlds.split(",") if v.strip())
    except ValueError as exc:
        raise ValueError("--worlds must be integers in 1..8") from exc
    if len(requested_worlds) == 1:
        requested_worlds *= args.workers
    if len(requested_worlds) != args.workers or any(w < 1 or w > 8 for w in requested_worlds):
        raise ValueError("--worlds must provide one value in 1..8 per training worker")

    repository_root = Path(__file__).resolve().parents[2]
    bridge_template = repository_root / "python" / "fceux_bridge" / "mario_ai_fceux_bridge.lua"
    args.run_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "unknown"

    # Metadata snapshot (statistics use — not training hot path).
    metadata = {
        "algorithm": "Ape-X Rainbow (C51 + NoisyNet + Double DQN + Dueling + PER + N-step)",
        "architecture": "distributed_async",
        "training_actors": args.workers,
        "eval_actor": 1,
        "worlds": requested_worlds,
        "replay_capacity": args.replay_capacity,
        "exploration": "Ape-X actor epsilon-greedy plus NoisyNet; no Mario action prior",
        "actor_epsilons": [round(_apex_epsilon(i, args.workers), 5) for i in range(args.workers)],
        "actor_seeds": [args.seed + i * 1000 for i in range(args.workers)],
        "weight_sync_every_optimizer_steps": args.weight_sync_every,
        "n_step": args.n_step,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "evaluation_seed": args.seed,
        "eval_world": args.eval_world,
        "eval_episodes": args.eval_episodes,
        "action_repeat_frames": 12,
        "start_protocol": START_PROTOCOL,
        "determinism": "seeded components; asynchronous queue interleaving is not bitwise reproducible",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "source_revision": source_revision,
        "rom_sha256": hashlib.sha256(args.rom.read_bytes()).hexdigest(),
    }
    fceux_executable = shutil.which(args.fceux) or args.fceux
    metadata["fceux_executable"] = str(Path(fceux_executable).resolve())
    try:
        metadata["fceux_sha256"] = hashlib.sha256(Path(fceux_executable).resolve().read_bytes()).hexdigest()
    except OSError:
        metadata["fceux_sha256"] = None
    (args.run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    atomic_write_json(args.run_dir / "learner_status.json", {
        "steps": 0, "transitions_received": 0, "optimizer_updates": 0,
        "replay_transitions": 0, "epsilon": 0.0, "latest_loss": None,
    })
    logger.info("Run config written: %s", args.run_dir / "run.json")

    context = multiprocessing.get_context("spawn")

    # One shared experience queue with backpressure.
    experience_queue: multiprocessing.Queue = context.Queue(maxsize=args.queue_capacity)

    # One weight queue per actor (+ eval worker).
    total_weight_queues = args.workers + 1  # +1 for eval
    weight_queues: list[multiprocessing.Queue] = [
        context.Queue(maxsize=2) for _ in range(total_weight_queues)
    ]
    eval_weight_queue = weight_queues[-1]

    # Status channel (learner health queries).
    status_inbox: multiprocessing.Queue = context.Queue(maxsize=10)
    status_outbox: multiprocessing.Queue = context.Queue(maxsize=10)

    # ---- Build AgentConfig (learner-side).
    agent_config = AgentConfig(
        observation_size=184,
        action_count=6,
        gamma=0.99,
        learning_rate=6.25e-5,
        batch_size=args.batch_size,
        learning_starts=10_000,
        target_sync_steps=2_000,
        n_step=args.n_step,
        atom_count=51,
        value_min=-100.0,
        value_max=100.0,
        per_beta_start=0.4,
        per_beta_steps=1_000_000,
        seed=args.seed,
    )
    learner_config = {**vars(agent_config), "replay_capacity": args.replay_capacity}

    # ---- Build ActorConfig (shared template; index injected at launch).
    actor_config = ActorConfig(
        observation_size=184,
        action_count=6,
        gamma=0.99,
        n_step=args.n_step,
        batch_size=args.actor_batch_size,
        atom_count=51,
        value_min=-100.0,
        value_max=100.0,
        weight_sync_every=args.actor_weight_sync_every,
        seed=args.seed,
    )

    # Track direct emulator processes immediately so a failed later startup
    # stage cannot leave orphan FCEUX windows behind.
    fceux_processes = []
    eval_fceux = []

    def _stop_emulators() -> None:
        for process in fceux_processes + eval_fceux:
            if process.poll() is None:
                process.terminate()
        for process in fceux_processes + eval_fceux:
            try:
                process.wait(timeout=5)
            except Exception:
                if process.poll() is None:
                    process.kill()

    atexit.register(_stop_emulators)

    # ---- Launch FCEUX processes (training actors).
    fceux_processes.extend(launch_fceux_workers(
        args.fceux, args.rom, bridge_template,
        args.run_dir, args.workers, requested_worlds,
    ))
    training_workers = [
        FileWorker(f"actor-{i:02d}", args.run_dir / f"worker-{i:02d}")
        for i in range(args.workers)
    ]

    # ---- Launch FCEUX process for the eval worker.
    eval_run_dir = args.run_dir / "eval"
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    eval_fceux.extend(launch_fceux_workers(
        args.fceux, args.rom, bridge_template,
        eval_run_dir, 1, (args.eval_world,),
    ))
    eval_file_worker = FileWorker("eval", eval_run_dir / "worker-00")

    # ---- Launch learner process.
    learner_process = context.Process(
        target=apex_learner_main,
        args=(experience_queue, weight_queues, status_inbox, status_outbox,
              str(args.run_dir), learner_config, args.resume, args.device,
              args.weight_sync_every, args.checkpoint_every, args.learn_per_batch),
        daemon=True,
    )
    learner_process.start()
    logger.info("Learner process started (PID %d)", learner_process.pid)

    # ---- Launch actor processes.
    actor_processes = []
    for i, worker in enumerate(training_workers):
        p = context.Process(
            target=actor_main,
            args=(i, args.workers, worker, experience_queue, weight_queues[i],
                  str(args.run_dir), vars(actor_config), args.device),
            daemon=True,
        )
        p.start()
        actor_processes.append(p)
        logger.info("Actor %d started (PID %d, ε=%.4f)", i, p.pid,
                    _apex_epsilon(i, args.workers))

    # ---- Launch eval worker process.
    eval_process = context.Process(
        target=eval_worker_main,
        args=(eval_file_worker, eval_weight_queue, str(args.run_dir),
              184, 6, 51, -100.0, 100.0,
              args.eval_every, args.eval_episodes, 300.0, args.device),
        daemon=True,
    )
    eval_process.start()
    logger.info("Eval worker started (PID %d)", eval_process.pid)

    # ---- Coordinator loop (health monitoring only — no training logic here).
    active = True

    def stop(*_: object) -> None:
        nonlocal active
        active = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    next_health_at = time.monotonic()
    next_status_at = time.monotonic()
    learner_status: dict = {"steps": 0, "optimizer_updates": 0,
                            "replay_transitions": 0, "epsilon": 0.0, "latest_loss": None}

    try:
        while active:
            now = time.monotonic()
            if now >= next_status_at:
                if not learner_process.is_alive():
                    raise RuntimeError("Rainbow learner exited; see the learner log before resuming")
                dead_actors = [index for index, process in enumerate(actor_processes)
                               if not process.is_alive()]
                if dead_actors:
                    raise RuntimeError(f"Ape-X actor process(es) exited: {dead_actors}")
                dead_emulators = [index for index, process in enumerate(fceux_processes)
                                  if process.poll() is not None]
                if dead_emulators:
                    raise RuntimeError(f"FCEUX training emulator(s) exited: {dead_emulators}")
                if not eval_process.is_alive():
                    logger.warning("Evaluation process exited; training continues without current eval data")
                try:
                    status_inbox.put_nowait(("status",))
                    _, learner_status = status_outbox.get(timeout=3)
                    atomic_write_json(args.run_dir / "learner_status.json", learner_status)
                except Exception as exc:
                    logger.warning("Learner status query failed: %s", exc)
                if learner_status.get("transitions_received", 0) >= args.steps:
                    logger.info("Reached requested experience target of %d transitions", args.steps)
                    active = False
                next_status_at = now + 10.0
            if now >= next_health_at:
                # Read actor metrics from per-actor JSON files.
                actor_metrics = []
                for i in range(args.workers):
                    metrics_path = args.run_dir / f"actor_{i:02d}_metrics.json"
                    try:
                        import json as _json
                        actor_metrics.append(_json.loads(metrics_path.read_text()))
                    except Exception:
                        actor_metrics.append({"actor": i, "steps": 0})

                # Read latest eval result.
                eval_record: dict | None = None
                eval_latest = args.run_dir / "eval_latest.json"
                try:
                    import json as _json
                    eval_record = _json.loads(eval_latest.read_text())
                except Exception:
                    pass

                write_health_report(
                    args.run_dir, repository_root,
                    learner_status, actor_metrics, eval_record, args.workers,
                )
                logger.info(
                    "Health | learner steps=%d optimizer=%d replay=%d loss=%s",
                    learner_status.get("steps", 0),
                    learner_status.get("optimizer_updates", 0),
                    learner_status.get("replay_transitions", 0),
                    f"{learner_status.get('latest_loss', 0):.5f}"
                    if learner_status.get("latest_loss") else "warming up",
                )
                next_health_at = now + HEALTH_CHECK_INTERVAL

            time.sleep(5.0)

    finally:
        logger.info("Shutting down...")
        status_inbox.put(("stop",))
        learner_process.join(timeout=90)
        if learner_process.is_alive():
            learner_process.kill()
        for p in actor_processes + [eval_process]:
            if p.is_alive():
                p.terminate()
        for p in actor_processes + [eval_process]:
            try:
                p.join(timeout=5)
            except Exception:
                p.kill()
        for proc in fceux_processes + eval_fceux:
            if proc.poll() is None:
                proc.terminate()
        for proc in fceux_processes + eval_fceux:
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        logger.info("All processes stopped.")


if __name__ == "__main__":
    main()
