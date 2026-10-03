"""Reward and episode statistics for the vectorized training loop.

Stable-Baselines3 reports the shaped return and nothing else. This module
intercepts the rollout, decomposes every step's reward by cause, and counts
finished episodes so a training run's TensorBoard makes the same claims the
evaluation report does: how many World 1-1 episodes finished, at what x
position, and why they ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper

from .env import LevelSpec
from .episodes import REASON_DEATH, REASON_FLAG, EpisodeRecord, EpisodeTracker
from .rewards import EVENT_TRUNCATED

REWARD_TERM_NAMES = (
    "forward",
    "mushroom",
    "fire_flower",
    "power_loss",
    "death",
    "timeout",
    "flag",
    "time",
)


@dataclass
class RolloutStats:
    """Aggregates drained from one rollout or evaluation window."""

    steps: int = 0
    episodes: int = 0
    wins: int = 0
    deaths: int = 0
    mean_max_x: float = 0.0
    mean_length: float = 0.0
    reasons: dict[str, int] = field(default_factory=dict)
    reward_terms: dict[str, float] = field(default_factory=dict)
    raw_reward_per_step: float = 0.0
    shaped_reward_per_step: float = 0.0
    episodes_total: int = 0
    wins_total: int = 0
    deaths_total: int = 0
    mean_max_x_total: float = 0.0
    mean_length_total: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0

    @property
    def win_rate_total(self) -> float:
        return self.wins_total / self.episodes_total if self.episodes_total else 0.0


class RewardStatsVecEnv(VecEnvWrapper):
    """Track reward components and finished episodes across a vectorized env."""

    def __init__(
        self,
        venv,
        level: LevelSpec,
        frame_skip: int = 1,
        record_trace: bool = False,
        keep_records: bool = False,
    ) -> None:
        super().__init__(venv)
        self.level = level
        self.frame_skip = frame_skip
        self.keep_records = keep_records
        self.records: list[EpisodeRecord] = []
        self.trackers = [
            EpisodeTracker(
                level.world,
                level.stage,
                index=1 + rank,
                record_trace=record_trace,
                frame_skip=frame_skip,
            )
            for rank in range(venv.num_envs)
        ]
        self._actions: np.ndarray | None = None
        self._begin_window()
        self.episodes_total = 0
        self.wins_total = 0
        self.deaths_total = 0
        self.max_x_total = 0.0
        self.length_total = 0

    # MARK: VecEnv plumbing

    def step_async(self, actions) -> None:
        self._actions = np.asarray(actions).reshape(-1)
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        for index, info in enumerate(infos):
            self._accumulate(info, float(rewards[index]))
            done = bool(dones[index])
            event = str(info.get("mario_event", ""))
            truncated = done and event == EVENT_TRUNCATED
            terminated = done and not truncated
            action = None
            if self._actions is not None and index < len(self._actions):
                action = int(self._actions[index])
            finished = self.trackers[index].observe(
                info, terminated, truncated, action=action, step_reward=float(rewards[index])
            )
            if finished is not None:
                self._record(finished)
        self.window_steps += len(infos)
        return observations, rewards, dones, infos

    def reset(self):
        observations = self.venv.reset()
        for tracker in self.trackers:
            tracker.reset()
        return observations

    # MARK: Aggregation

    def drain(self) -> RolloutStats:
        """Return the window's statistics and start a fresh window."""
        stats = RolloutStats(
            steps=self.window_steps,
            episodes=len(self.window_episodes),
            wins=sum(1 for record in self.window_episodes if record.won),
            deaths=sum(1 for record in self.window_episodes if record.reason == REASON_DEATH),
            mean_max_x=self._mean([record.max_x for record in self.window_episodes]),
            mean_length=self._mean([record.steps for record in self.window_episodes]),
            reasons=self._reasons(),
            reward_terms={
                name: self.window_terms.get(name, 0.0) / max(1, self.window_steps)
                for name in REWARD_TERM_NAMES
            },
            raw_reward_per_step=self.window_raw / max(1, self.window_steps),
            shaped_reward_per_step=self.window_shaped / max(1, self.window_steps),
            episodes_total=self.episodes_total,
            wins_total=self.wins_total,
            deaths_total=self.deaths_total,
            mean_max_x_total=self.max_x_total / self.episodes_total if self.episodes_total else 0.0,
            mean_length_total=self.length_total / self.episodes_total if self.episodes_total else 0.0,
        )
        self._begin_window()
        return stats

    def _begin_window(self) -> None:
        self.window_steps = 0
        self.window_terms: dict[str, float] = {}
        self.window_raw = 0.0
        self.window_shaped = 0.0
        self.window_episodes: list[EpisodeRecord] = []

    def _accumulate(self, info: dict, reward: float) -> None:
        terms = info.get("reward_terms")
        if isinstance(terms, dict):
            for name, value in terms.items():
                try:
                    self.window_terms[name] = self.window_terms.get(name, 0.0) + float(value)
                except (TypeError, ValueError):
                    continue
        try:
            self.window_raw += float(info.get("raw_reward", 0.0))
        except (TypeError, ValueError):
            pass
        self.window_shaped += reward

    def _record(self, record: EpisodeRecord) -> None:
        self.window_episodes.append(record)
        if self.keep_records:
            self.records.append(record)
        self.episodes_total += 1
        self.wins_total += 1 if record.won else 0
        self.deaths_total += 1 if record.reason == REASON_DEATH else 0
        self.max_x_total += record.max_x
        self.length_total += record.steps

    def stream_records(self) -> list[EpisodeRecord]:
        """Return the episodes finished so far, newest included."""
        return list(self.records)

    def _reasons(self) -> dict[str, int]:
        reasons: dict[str, int] = {}
        for record in self.window_episodes:
            reasons[record.reason] = reasons.get(record.reason, 0) + 1
        return reasons

    @staticmethod
    def _mean(values: list) -> float:
        return float(np.mean(values)) if values else 0.0


class EpisodeOutcomeCallback(BaseCallback):
    """Write measured episode outcomes and reward decomposition to TensorBoard."""

    def __init__(self, stats_env: RewardStatsVecEnv, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.stats_env = stats_env

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        stats = self.stats_env.drain()
        self.logger.record("rollout/shaped_reward_per_step", stats.shaped_reward_per_step)
        self.logger.record("rollout/raw_reward_per_step", stats.raw_reward_per_step)
        for name in REWARD_TERM_NAMES:
            self.logger.record(f"reward_terms/{name}_per_step", stats.reward_terms[name])
        self.logger.record("episodes/finished", stats.episodes)
        self.logger.record("episodes/win_rate_window", stats.win_rate)
        self.logger.record("episodes/win_rate_total", stats.win_rate_total)
        self.logger.record("episodes/wins_total", stats.wins_total)
        self.logger.record("episodes/deaths_total", stats.deaths_total)
        self.logger.record("episodes/mean_max_x_window", stats.mean_max_x)
        self.logger.record("episodes/mean_max_x_total", stats.mean_max_x_total)
        self.logger.record("episodes/mean_length_window", stats.mean_length)
        self.logger.record("episodes/mean_length_total", stats.mean_length_total)
        for reason in (REASON_FLAG, REASON_DEATH, "timeout", "truncated", "incomplete"):
            self.logger.record(f"episodes/reason_{reason}_window", stats.reasons.get(reason, 0))
        return True
