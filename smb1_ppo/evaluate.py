"""Deterministic, no-cheat evaluation of a trained SMB1 PPO policy.

    python -m smb1_ppo.evaluate --model runs/smb1-ppo-w1-1/best_model.zip --episodes 20

The evaluation is greedy and on demand. It never opens an emulator window, and
it writes machine-readable evidence: ``results.json``, ``episodes.csv``,
``action_trace.csv``, and ``action_trace.jsonl``.

Two honest caveats are recorded in every report:

* Super Mario Bros. 1-1 restores the same backup state on every reset and the
  NES emulator is deterministic, so greedy episodes repeat one trajectory.
  Twenty of them show the policy is stable, not that it generalises. The report
  therefore counts how many distinct action traces it actually observed.
* A synthetic stand-in run is never promoted, because it is not World 1-1.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from . import checkpoints, cli
from .checkpoints import DEFAULT_RUN_DIRECTORY, guard_run_directory
from .env import ENVIRONMENT_SMB1, EnvConfig, make_vec_env, resolve_device
from .episodes import (
    PROMOTION_EPISODES,
    REPORT_FIELDS,
    TRACE_FIELDS,
    EpisodeRecord,
    summarize,
)
from .stats import RewardStatsVecEnv

DEFAULT_EPISODES = PROMOTION_EPISODES
MAX_DECISION_BUDGET_FACTOR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m smb1_ppo.evaluate",
        description="Run deterministic no-cheat World 1-1 episodes and report measured results.",
    )
    cli.add_environment_arguments(parser)
    cli.add_reward_arguments(parser)
    cli.add_device_arguments(parser)
    parser.add_argument("--model", type=Path, required=True, help="path to a saved PPO .zip")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=f"run directory holding run.json and receiving evaluations/ (default {DEFAULT_RUN_DIRECTORY})",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="explicit output directory for this evaluation"
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel evaluation environments")
    parser.add_argument("--no-action-trace", dest="action_trace", action="store_false", default=True)
    parser.add_argument(
        "--sample-actions",
        dest="deterministic",
        action="store_false",
        default=True,
        help="sample actions instead of acting greedily; the result can never be promoted",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def build_evaluation_environment(
    config: EnvConfig, workers: int = 1, record_trace: bool = True
) -> RewardStatsVecEnv:
    """Wrap a vectorized environment so finished episodes are retained."""
    environment = make_vec_env(config, workers=workers, record_trace=record_trace)
    return RewardStatsVecEnv(
        environment,
        config.level,
        frame_skip=config.frame_skip,
        record_trace=record_trace,
        keep_records=True,
    )


def collect_episodes(
    model,
    environment: RewardStatsVecEnv,
    config: EnvConfig,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> list[EpisodeRecord]:
    """Play until ``episodes`` have finished, returning every measured record."""
    observations = environment.reset()
    budget = config.max_episode_steps * episodes * MAX_DECISION_BUDGET_FACTOR + 1000
    decisions = 0
    while len(environment.stream_records()) < episodes:
        if decisions >= budget:
            raise RuntimeError(
                f"evaluation exceeded {budget} decisions without finishing {episodes} episodes; "
                "check --max-episode-steps and the model"
            )
        actions, _ = model.predict(observations, deterministic=deterministic)
        observations, _, _, _ = environment.step(actions)
        decisions += 1
    return environment.stream_records()[:episodes]


def resolve_run_directory(model: Path, requested: Path | None) -> Path:
    """Find the run directory that owns a model, unless one is given."""
    if requested is not None:
        return guard_run_directory(requested)
    parent = Path(model).expanduser().resolve().parent
    if (parent / checkpoints.RUN_METADATA).is_file():
        return guard_run_directory(parent)
    if (parent.parent / checkpoints.RUN_METADATA).is_file():
        return guard_run_directory(parent.parent)
    return guard_run_directory(DEFAULT_RUN_DIRECTORY)


def load_run_metadata(run_directory: Path) -> dict:
    return checkpoints.read_json(run_directory / checkpoints.RUN_METADATA) or {}


def check_environment_match(config: EnvConfig, metadata: dict) -> dict:
    """Refuse to evaluate with a different observation pipeline than training used."""
    recorded = (metadata.get("environment") or {}) if metadata else {}
    differences = {}
    for key in ("frame_skip", "frame_size", "frame_stack"):
        expected = recorded.get(key)
        actual = getattr(config, key)
        if expected is not None and int(expected) != int(actual):
            differences[key] = {"trained": expected, "requested": actual}
    if differences:
        details = ", ".join(
            f"{key}: trained {value['trained']} but {value['requested']} requested"
            for key, value in differences.items()
        )
        raise SystemExit(
            f"observation pipeline mismatch with {metadata.get('run_directory')}: {details}. "
            "Pass the same --frame-skip/--frame-size/--frame-stack the model was trained with."
        )
    reward = recorded.get("reward")
    reward_drift = {}
    if isinstance(reward, dict):
        for key, value in config.reward.__dict__.items():
            if key in reward and float(reward[key]) != float(value):
                reward_drift[key] = {"trained": reward[key], "requested": value}
    return {"checked": True, "reward_drift": reward_drift}


def distinct_traces(records: list[EpisodeRecord]) -> int:
    """Count distinct action traces, the real content of a deterministic run."""
    return len({tuple(int(step["action"]) for step in record.trace) for record in records})


def write_episode_csv(path: Path, records: list[EpisodeRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_FIELDS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.row())


def write_action_traces(directory: Path, records: list[EpisodeRecord]) -> None:
    with (directory / "action_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", *TRACE_FIELDS])
        writer.writeheader()
        for record in records:
            for step in record.trace:
                writer.writerow({"episode": record.episode, **step})
    with (directory / "action_trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "episode": record.episode,
                        "won": record.won,
                        "reason": record.reason,
                        "actions": [int(step["action"]) for step in record.trace],
                    }
                )
                + "\n"
            )


def build_report(
    records: list[EpisodeRecord],
    config: EnvConfig,
    model_path: Path,
    metadata: dict,
    deterministic: bool,
    match: dict,
    environment_config: dict,
    extra: dict | None = None,
) -> dict:
    promotable = config.environment == ENVIRONMENT_SMB1 and deterministic
    report = summarize(records, config.level.label, promotable=promotable)
    report.update(
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": config.environment,
            "deterministic": deterministic,
            "model": str(model_path),
            "run_directory": str(config.run_directory) if hasattr(config, "run_directory") else None,
            "trained_run": metadata.get("run_directory"),
            "deaths": sum(1 for record in records if not record.won),
            "distinct_action_traces": distinct_traces(records) if records and records[0].trace else None,
            "environment_config": environment_config,
            "environment_match": match,
            "cheats": "none: stock environment, stock ROM mode 'vanilla'",
        }
    )
    if report["promotion"]["promoted"] is False and not promotable:
        report["promotion"]["blocked_by"] = (
            "a synthetic stand-in environment is not World 1-1"
            if config.environment != ENVIRONMENT_SMB1
            else "actions were sampled, not greedy"
        )
    if extra:
        report.update(extra)
    return report


def main() -> None:
    options = build_parser().parse_args()
    if options.episodes < 1:
        raise ValueError("--episodes must be positive")
    model_path = options.model.expanduser()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    config = cli.env_config_from(options)
    environment_config = cli.environment_summary(config)
    run_directory = resolve_run_directory(model_path, options.run_dir)
    metadata = load_run_metadata(run_directory)
    match = check_environment_match(config, metadata)
    if match["reward_drift"] and not options.quiet:
        print(
            "warning: reward weights differ from training "
            f"({match['reward_drift']}); the reported shaped reward will not match training "
            "for the same trajectory"
        )
    device = resolve_device(options.device)
    directory = (
        guard_run_directory(options.out)
        if options.out
        else run_directory / "evaluations" / time.strftime("%Y%m%d-%H%M%S")
    )
    directory.mkdir(parents=True, exist_ok=True)

    from .checkpoints import load_model

    environment = build_evaluation_environment(
        config, workers=options.workers, record_trace=options.action_trace
    )
    model = load_model(model_path, env=environment, device=device)
    records = collect_episodes(
        model, environment, config, options.episodes, options.seed, options.deterministic
    )
    environment.close()

    report = build_report(
        records, config, model_path, metadata, options.deterministic, match, environment_config
    )
    report["run_directory"] = str(run_directory)
    report["evaluation_directory"] = str(directory)
    (directory / "results.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_episode_csv(directory / "episodes.csv", records)
    if options.action_trace:
        write_action_traces(directory, records)

    if not options.quiet:
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "episodes",
                        "wins",
                        "win_rate",
                        "x_position",
                        "duration_seconds",
                        "death_reasons",
                        "death_causes",
                        "power_ups",
                        "distinct_action_traces",
                        "promotion",
                    )
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote {directory / 'results.json'}")
        print(f"wrote {directory / 'episodes.csv'}")
        if options.action_trace:
            print(f"wrote {directory / 'action_trace.csv'}")
            print(f"wrote {directory / 'action_trace.jsonl'}")


if __name__ == "__main__":
    main()
