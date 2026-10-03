"""Checkpoint, best-model, and run-directory bookkeeping.

Two rules this module enforces:

* A run may never write into the directories the live Ape-X Rainbow experiment
  owns, so a PPO experiment cannot disturb a training run that is in progress.
* ``best_model.zip`` is only ever replaced by a *measured* deterministic
  evaluation that beats the previous best, so the name means something.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from .rom import REPOSITORY_ROOT

PROTECTED_RUN_DIRECTORIES = (
    "runs/w1-bootcamp",
    "runs/full-rainbow-input-fixed",
    "runs/full-rainbow",
)
DEFAULT_RUN_DIRECTORY = Path("runs/smb1-ppo-w1-1")

CHECKPOINT_DIRECTORY = "checkpoints"
LATEST_CHECKPOINT = "latest.zip"
BEST_MODEL = "best_model.zip"
TRAINING_STATE = "training_state.json"
BEST_EVALUATION = "best_model_eval.json"
RUN_METADATA = "run.json"

_CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d+)_steps\.zip$")


class RunDirectoryError(RuntimeError):
    """Raised when a run directory would collide with another experiment."""


def guard_run_directory(path: Path | str) -> Path:
    """Return the resolved run directory, refusing protected locations."""
    resolved = Path(path).expanduser().resolve()
    for protected in PROTECTED_RUN_DIRECTORIES:
        target = (REPOSITORY_ROOT / protected).resolve()
        if resolved == target or target in resolved.parents:
            raise RunDirectoryError(
                f"{path} is inside the Ape-X Rainbow experiment's {protected}. "
                "This PPO baseline must not write there; use runs/smb1-ppo-w1-1 instead."
            )
    return resolved


def checkpoint_directory(run_directory: Path | str) -> Path:
    return guard_run_directory(run_directory) / CHECKPOINT_DIRECTORY


def checkpoint_filename(steps: int) -> str:
    return f"checkpoint_{int(steps):012d}_steps.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def checkpoint_metadata(path: Path, steps: int, config: dict, evaluation: dict | None = None) -> dict:
    """Describe a checkpoint well enough to resume or audit it later."""
    return {
        "checkpoint": Path(path).name,
        "step": int(steps),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": _sha256(Path(path)),
        "size_bytes": Path(path).stat().st_size,
        "algorithm": "PPO",
        "config": config,
        "deterministic_evaluation": evaluation,
    }


def save_checkpoint(
    model,
    run_directory: Path | str,
    steps: int,
    config: dict,
    evaluation: dict | None = None,
    label: str | None = None,
) -> Path:
    """Save a checkpoint plus the metadata needed to resume from it."""
    directory = checkpoint_directory(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (label or checkpoint_filename(steps))
    model.save(path)
    write_json(
        path.with_suffix(path.suffix + ".json"),
        checkpoint_metadata(path, steps, config, evaluation),
    )
    return path


def list_checkpoints(run_directory: Path | str) -> list[Path]:
    """Return periodic checkpoints newest first."""
    directory = checkpoint_directory(run_directory)
    if not directory.is_dir():
        return []
    found = []
    for candidate in directory.glob("checkpoint_*_steps.zip"):
        match = _CHECKPOINT_PATTERN.search(candidate.name)
        if match:
            found.append((int(match.group(1)), candidate))
    return [path for _, path in sorted(found, reverse=True)]


def prune_checkpoints(run_directory: Path | str, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` periodic checkpoints."""
    removed = []
    for path in list_checkpoints(run_directory)[keep:]:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".json").unlink(missing_ok=True)
        removed.append(path)
    return removed


def write_training_state(run_directory: Path | str, state: dict) -> Path:
    path = guard_run_directory(run_directory) / TRAINING_STATE
    write_json(path, state)
    return path


def read_training_state(run_directory: Path | str) -> dict:
    return read_json(guard_run_directory(run_directory) / TRAINING_STATE) or {}


def write_run_metadata(run_directory: Path | str, payload: dict) -> Path:
    path = guard_run_directory(run_directory) / RUN_METADATA
    write_json(path, payload)
    return path


def read_best_model_evaluation(run_directory: Path | str) -> dict | None:
    return read_json(guard_run_directory(run_directory) / BEST_EVALUATION)


def evaluation_rank(evaluation: dict | None) -> tuple[float, float]:
    """Rank an evaluation by win rate first, then mean furthest x reached."""
    if not evaluation:
        return (-1.0, -1.0)
    positions = evaluation.get("x_position") or {}
    return (float(evaluation.get("win_rate", 0.0)), float(positions.get("mean", 0.0)))


def mark_best_model(
    model,
    run_directory: Path | str,
    evaluation: dict,
    config: dict | None = None,
) -> Path | None:
    """Replace ``best_model.zip`` only when a greedy evaluation improves on it."""
    run_directory = guard_run_directory(run_directory)
    previous = read_best_model_evaluation(run_directory)
    if evaluation_rank(evaluation) <= evaluation_rank(previous):
        return None
    path = run_directory / BEST_MODEL
    model.save(path)
    payload = dict(evaluation)
    payload["step"] = getattr(model, "num_timesteps", None)
    payload["selected_by"] = "deterministic greedy evaluation only"
    payload["previous"] = previous
    payload["config"] = config
    write_json(run_directory / BEST_EVALUATION, payload)
    return path


def discard_stale_best_model(run_directory: Path | str) -> bool:
    """Remove a best-model left behind by an earlier run in the same directory.

    A fresh start (no --resume) is a new training run, so its first measured
    evaluation must rank against nothing rather than against a policy the
    operator may have left behind; otherwise a stale best_model.zip keeps
    winning the comparison and gets reported as the new run's best.
    """
    run_directory = guard_run_directory(run_directory)
    removed = False
    for name in (BEST_MODEL, BEST_EVALUATION):
        candidate = run_directory / name
        if candidate.is_file():
            candidate.unlink()
            removed = True
    return removed


def resolve_resume(run_directory: Path | str, request: str | None) -> Path:
    """Resolve ``--resume`` into a concrete checkpoint path."""
    run_directory = guard_run_directory(run_directory)
    if request and request not in ("latest", "best", "auto"):
        candidate = Path(request).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"checkpoint not found: {candidate}")
        return candidate
    candidates = []
    if request != "best":
        candidates.append(run_directory / CHECKPOINT_DIRECTORY / LATEST_CHECKPOINT)
    candidates.append(run_directory / BEST_MODEL)
    candidates.extend(list_checkpoints(run_directory))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no checkpoint to resume from in {run_directory}; run training without --resume first"
    )


def load_model(path: Path | str, env=None, device: str = "auto"):
    """Load a saved PPO policy, optionally binding it to an environment."""
    from stable_baselines3 import PPO

    return PPO.load(Path(path), env=env, device=device)
