"""Train the SMB1 PPO baseline.

Example, World 1-1 fresh from the Mac:

    python -m smb1_ppo.train --rom roms/super-mario-bros.nes \\
        --world 1 --stage 1 --workers 8 --total-timesteps 2000000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from . import checkpoints, cli
from .checkpoints import (
    DEFAULT_RUN_DIRECTORY,
    guard_run_directory,
    prune_checkpoints,
    read_best_model_evaluation,
    read_training_state,
    save_checkpoint,
    write_run_metadata,
    write_training_state,
)
from .env import (
    EnvConfig,
    describe_environment,
    device_caveat,
    make_vec_env,
    resolve_device,
)
from .evaluate import build_evaluation_environment, collect_episodes
from .stats import RewardStatsVecEnv

TENSORBOARD_DIRECTORY = "tensorboard"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m smb1_ppo.train",
        description="Train a PPO baseline for Super Mario Bros. 1 with no cheats.",
    )
    cli.add_environment_arguments(parser)
    cli.add_reward_arguments(parser)
    cli.add_device_arguments(parser)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIRECTORY)
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=8, help="parallel environments")
    parser.add_argument("--n-steps", type=int, default=256, help="rollout steps per worker per update")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--features-dim", type=int, default=512)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="0 disables checkpoints")
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=100_000, help="0 disables periodic evaluation")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--no-eval", dest="evaluate", action="store_false", default=True)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="resume from latest, best, or a checkpoint path",
    )
    parser.add_argument(
        "--progress-bar", dest="progress_bar", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


class PeriodicCheckpointCallback(BaseCallback):
    """Save resumable checkpoints on an env-step schedule."""

    def __init__(
        self, run_directory: Path, frequency: int, keep: int, config: dict, verbose: int = 0
    ) -> None:
        super().__init__(verbose)
        self.run_directory = run_directory
        self.frequency = frequency
        self.keep = keep
        self.config = config
        self.last_step = 0

    def _on_step(self) -> bool:
        if self.frequency <= 0:
            return True
        steps = int(self.num_timesteps)
        if steps - self.last_step < self.frequency:
            return True
        self.last_step = steps
        save_checkpoint(self.model, self.run_directory, steps, self.config)
        save_checkpoint(
            self.model,
            self.run_directory,
            steps,
            self.config,
            label=checkpoints.LATEST_CHECKPOINT,
        )
        prune_checkpoints(self.run_directory, self.keep)
        write_training_state(
            self.run_directory,
            {"step": steps, "checkpoint": checkpoints.LATEST_CHECKPOINT, "saved_at": time.time()},
        )
        return True


class DeterministicEvaluationCallback(BaseCallback):
    """Measure greedy World 1-1 episodes and promote a best model on improvement."""

    def __init__(
        self,
        config: EnvConfig,
        run_directory: Path,
        frequency: int,
        episodes: int,
        seed: int,
        environment_config: dict,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.config = config
        self.run_directory = run_directory
        self.frequency = frequency
        self.episodes = episodes
        self.seed = seed
        self.environment_config = environment_config
        self.last_step = 0
        self._environment: RewardStatsVecEnv | None = None
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.frequency <= 0 or self.episodes <= 0:
            return True
        steps = int(self.num_timesteps)
        if steps - self.last_step < self.frequency:
            return True
        self.last_step = steps
        self.history.append(self.evaluate(steps))
        return True

    def evaluate(self, steps: int) -> dict:
        if self._environment is None:
            self._environment = build_evaluation_environment(self.config)
        records = collect_episodes(
            self.model,
            self._environment,
            self.config,
            episodes=self.episodes,
            seed=self.seed,
            deterministic=True,
        )
        from .episodes import summarize

        report = summarize(records, self.config.level.label, promotable=not self.config.is_synthetic)
        report["step"] = steps
        report["deterministic"] = True
        report["environment"] = self.config.environment
        report["windows"] = "during-training evaluation, fewer episodes than the promotion run"
        directory = guard_run_directory(self.run_directory) / "evaluations"
        directory.mkdir(parents=True, exist_ok=True)
        checkpoints.write_json(directory / f"step-{steps:012d}.json", report)
        promoted = checkpoints.mark_best_model(self.model, self.run_directory, report, self.config_dict())
        report["promoted_best_model"] = bool(promoted)
        print(
            f"[eval] step={steps} episodes={report['episodes']} wins={report['wins']} "
            f"win_rate={report['win_rate']:.3f} mean_max_x={report['x_position']['mean']:.1f}"
            + ("  -> best model updated" if promoted else "")
        )
        return report

    def config_dict(self) -> dict:
        return self.environment_config

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None


def main() -> None:
    options = build_parser().parse_args()
    if options.total_timesteps < 1:
        raise ValueError("--total-timesteps must be positive")
    if options.eval_episodes < 1:
        raise ValueError("--eval-episodes must be positive")
    run_directory = guard_run_directory(options.run_dir)
    run_directory.mkdir(parents=True, exist_ok=True)
    config = cli.env_config_from(options)
    device = resolve_device(options.device)
    caveat = device_caveat(device)
    if caveat:
        print(f"warning: {caveat}")

    environment_summary = cli.environment_summary(config)
    training_config = {
        "workers": options.workers,
        "n_steps": options.n_steps,
        "batch_size": options.batch_size,
        "n_epochs": options.n_epochs,
        "learning_rate": options.learning_rate,
        "gamma": options.gamma,
        "gae_lambda": options.gae_lambda,
        "clip_range": options.clip_range,
        "ent_coef": options.ent_coef,
        "vf_coef": options.vf_coef,
        "max_grad_norm": options.max_grad_norm,
        "features_dim": options.features_dim,
        "device": device,
        "seed": options.seed,
    }

    environment = make_vec_env(config, workers=options.workers)
    stats_env = RewardStatsVecEnv(environment, config.level, frame_skip=config.frame_skip)

    from stable_baselines3 import PPO

    policy_kwargs = {
        # The observation pipeline already normalizes to [0, 1]; this disables
        # SB3's own image preprocessing and its uint8 bounds assumption.
        "normalize_images": False,
        "features_extractor_kwargs": {"features_dim": options.features_dim},
    }
    resume_path = None
    if options.resume:
        resume_path = checkpoints.resolve_resume(run_directory, options.resume)
        model = checkpoints.load_model(resume_path, env=stats_env, device=device)
        model.set_random_seed(options.seed)
        restart = False
    else:
        model = PPO(
            "CnnPolicy",
            stats_env,
            n_steps=options.n_steps,
            batch_size=options.batch_size,
            n_epochs=options.n_epochs,
            learning_rate=options.learning_rate,
            gamma=options.gamma,
            gae_lambda=options.gae_lambda,
            clip_range=options.clip_range,
            ent_coef=options.ent_coef,
            vf_coef=options.vf_coef,
            max_grad_norm=options.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(options.tensorboard_log or (run_directory / TENSORBOARD_DIRECTORY)),
            seed=options.seed,
            verbose=1,
        )
        restart = True

    state = read_training_state(run_directory) if resume_path else {}
    checkpoint_callback = PeriodicCheckpointCallback(
        run_directory, options.checkpoint_freq, options.keep_checkpoints, training_config
    )
    evaluation_callback = DeterministicEvaluationCallback(
        config,
        run_directory,
        options.eval_freq if options.evaluate else 0,
        options.eval_episodes,
        options.seed,
        cli.environment_summary(config),
    )

    write_run_metadata(
        run_directory,
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_directory": str(run_directory),
            "total_timesteps": options.total_timesteps,
            "resumed_from": str(resume_path) if resume_path else None,
            "previous_step": state.get("step"),
            "environment": environment_summary,
            "training": training_config,
            "checkpoint_frequency": options.checkpoint_freq,
            "evaluation_frequency": options.eval_freq if options.evaluate else 0,
            "evaluation_episodes": options.eval_episodes,
            "best_model_evaluation": read_best_model_evaluation(run_directory),
            "libraries": describe_environment(),
        },
    )
    print(
        f"training PPO on SMB1 {config.level.label} ({config.environment}) "
        f"workers={options.workers} steps={options.total_timesteps} device={device} "
        f"run_dir={run_directory}"
    )
    if resume_path:
        print(f"resumed from {resume_path}")

    model.learn(
        total_timesteps=options.total_timesteps,
        callback=[checkpoint_callback, evaluation_callback],
        reset_num_timesteps=restart,
        progress_bar=options.progress_bar,
    )

    final_step = int(model.num_timesteps)
    save_checkpoint(model, run_directory, final_step, training_config)
    save_checkpoint(model, run_directory, final_step, training_config, label=checkpoints.LATEST_CHECKPOINT)
    write_training_state(
        run_directory,
        {"step": final_step, "checkpoint": checkpoints.LATEST_CHECKPOINT, "finished": True},
    )
    if options.evaluate and not (
        evaluation_callback.history and evaluation_callback.history[-1]["step"] == final_step
    ):
        evaluation_callback.evaluate(final_step)
    evaluation_callback.close()
    stats_env.close()

    best = read_best_model_evaluation(run_directory)
    print(f"training finished at step {final_step}")
    if best:
        print(
            "best model from deterministic evaluation: "
            f"wins={best.get('wins')}/{best.get('episodes')} "
            f"win_rate={best.get('win_rate')} step={best.get('step')}"
        )
    print(f"checkpoints: {checkpoints.checkpoint_directory(run_directory)}")
    print(f"evaluate with: python -m smb1_ppo.evaluate --model {run_directory / 'best_model.zip'}")


if __name__ == "__main__":
    main()
