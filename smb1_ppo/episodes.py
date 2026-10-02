"""Episode accounting shared by training and evaluation.

One state machine consumes ``(info, terminated, truncated)`` and produces a
measured record per episode. Training uses it to log what actually happened
(wins, deaths, completions) instead of only a reward curve, and evaluation uses
the same code to produce the promotion report, so the two can never disagree.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field

from .actions import ACTION_COUNT, ACTION_NAMES, action_counts
from .rewards import (
    CAUSE_UNCLASSIFIED,
    EVENT_DEATH,
    EVENT_FLAG,
    EVENT_TIMEOUT,
    EVENT_TRUNCATED,
)

REASON_FLAG = "flag"
REASON_TIMEOUT = "timeout"
REASON_DEATH = "death"
REASON_TRUNCATED = "truncated"
REASON_INCOMPLETE = "incomplete"

# Promotion to "World 1-1 solved without cheats" requires this exact evidence.
PROMOTION_EPISODES = 20
PROMOTION_WIN_RATE = 1.0

TRACE_FIELDS = (
    "step",
    "action",
    "action_name",
    "x_pos",
    "y_pos",
    "time",
    "status",
    "score",
    "raw_reward",
    "shaped_reward",
    "event",
)

REPORT_FIELDS = (
    "episode",
    "world",
    "stage",
    "won",
    "reason",
    "death_cause",
    "steps",
    "frames",
    "duration_seconds",
    "max_x",
    "terminal_x",
    "final_status",
    "final_time",
    "final_score",
    "deaths",
    "mushrooms",
    "fire_flowers",
    "power_losses",
    "shaped_reward",
    "raw_reward",
    "action_counts",
) + tuple(f"action_{name}" for name in ACTION_NAMES)


@dataclass
class EpisodeRecord:
    """One measured SMB1 episode."""

    episode: int
    world: int
    stage: int
    won: bool = False
    reason: str = REASON_INCOMPLETE
    death_cause: str = CAUSE_UNCLASSIFIED
    steps: int = 0
    frames: int = 0
    duration_seconds: float = 0.0
    max_x: int = 0
    terminal_x: int = 0
    final_status: str = ""
    final_time: int | None = None
    final_score: int | None = None
    deaths: int = 0
    mushrooms: int = 0
    fire_flowers: int = 0
    power_losses: int = 0
    shaped_reward: float = 0.0
    raw_reward: float = 0.0
    action_counts: list[int] = field(default_factory=lambda: [0] * ACTION_COUNT)
    trace: list[dict] = field(default_factory=list)
    started_at: float = 0.0

    def row(self) -> dict[str, object]:
        """Flatten the record into a CSV row without the step trace."""
        payload = asdict(self)
        payload.pop("trace")
        payload.pop("started_at")
        payload["action_counts"] = " ".join(str(count) for count in self.action_counts)
        for index, name in enumerate(ACTION_NAMES):
            payload[f"action_{name}"] = self.action_counts[index]
        return {key: payload[key] for key in REPORT_FIELDS}


class EpisodeTracker:
    """Accumulate one episode at a time into an :class:`EpisodeRecord`."""

    def __init__(
        self,
        world: int,
        stage: int,
        index: int = 1,
        record_trace: bool = True,
        frame_skip: int = 1,
    ) -> None:
        self.world = world
        self.stage = stage
        self.index = index
        self.record_trace = record_trace
        self.frame_skip = max(1, frame_skip)
        self.episodes: list[EpisodeRecord] = []
        self.actions: list[int] = []
        self._current = self._blank()

    def _blank(self) -> EpisodeRecord:
        record = EpisodeRecord(episode=self.index, world=self.world, stage=self.stage)
        record.started_at = time.perf_counter()
        return record

    def reset(self) -> None:
        """Begin a new episode."""
        self._current = self._blank()

    def observe(
        self,
        info: dict,
        terminated: bool,
        truncated: bool,
        action: int | None = None,
        step_reward: float = 0.0,
    ) -> EpisodeRecord | None:
        """Fold one step in, returning the finished record on the final step."""
        record = self._current
        terms = info.get("reward_terms") or {}
        record.steps += 1
        record.frames += self.frame_skip
        record.max_x = max(record.max_x, self._int(info, "x_pos", record.max_x))
        record.terminal_x = self._int(info, "x_pos", record.terminal_x)
        record.shaped_reward += float(step_reward)
        record.raw_reward += float(info.get("raw_reward", 0.0))
        record.final_status = str(info.get("status", record.final_status))
        record.final_time = self._int(info, "time", record.final_time)
        record.final_score = self._int(info, "score", record.final_score)
        record.mushrooms += 1 if terms.get("mushroom") else 0
        record.fire_flowers += 1 if terms.get("fire_flower") else 0
        record.power_losses += 1 if terms.get("power_loss") else 0
        event = str(info.get("mario_event", ""))
        if event == EVENT_DEATH:
            record.deaths += 1
            record.death_cause = str(info.get("death_cause", CAUSE_UNCLASSIFIED))
        if action is not None:
            self.actions.append(int(action))
        if self.record_trace:
            record.trace.append(self._trace_row(info, action, step_reward))
        if not (terminated or truncated):
            return None
        return self.finish(terminated=terminated, info=info)

    def finish(self, terminated: bool, info: dict | None = None) -> EpisodeRecord:
        """Close the current episode and append it to the log."""
        record = self._current
        info = info or {}
        event = str(info.get("mario_event", EVENT_TRUNCATED if not terminated else EVENT_DEATH))
        if event == EVENT_FLAG or bool(info.get("flag_get", False)):
            record.won = True
            record.reason = REASON_FLAG
        elif event == EVENT_TIMEOUT:
            record.reason = REASON_TIMEOUT
        elif event == EVENT_DEATH or terminated:
            record.reason = REASON_DEATH
        elif event == EVENT_TRUNCATED:
            record.reason = REASON_TRUNCATED
        else:
            record.reason = REASON_INCOMPLETE
        record.duration_seconds = max(0.0, time.perf_counter() - record.started_at)
        record.action_counts = action_counts(self.actions)
        self.episodes.append(record)
        self.actions = []
        self.index += 1
        self._current = self._blank()
        return record

    def _trace_row(self, info: dict, action: int | None, step_reward: float) -> dict:
        return {
            "step": self._current.steps,
            "action": -1 if action is None else int(action),
            "action_name": "unset" if action is None else ACTION_NAMES[int(action)],
            "x_pos": self._int(info, "x_pos", 0),
            "y_pos": self._int(info, "y_pos", 0),
            "time": self._int(info, "time", 0),
            "status": str(info.get("status", "")),
            "score": self._int(info, "score", 0),
            "raw_reward": round(float(info.get("raw_reward", 0.0)), 6),
            "shaped_reward": round(float(step_reward), 6),
            "event": str(info.get("mario_event", "")),
        }

    @staticmethod
    def _int(info: dict, key: str, fallback):
        value = info.get(key)
        if value is None:
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback


def summarize(records: list[EpisodeRecord], level_label: str, promotable: bool = True) -> dict[str, object]:
    """Build the aggregate report the promotion decision is made from."""
    episodes = len(records)
    wins = [record for record in records if record.won]
    positions = [record.max_x for record in records] or [0]
    durations = [record.duration_seconds for record in records] or [0.0]
    reasons: dict[str, int] = {}
    causes: dict[str, int] = {}
    actions = [0] * ACTION_COUNT
    for record in records:
        reasons[record.reason] = reasons.get(record.reason, 0) + 1
        if record.reason == REASON_DEATH:
            causes[record.death_cause] = causes.get(record.death_cause, 0) + 1
        for index in range(ACTION_COUNT):
            actions[index] += record.action_counts[index]
    win_rate = len(wins) / episodes if episodes else 0.0
    return {
        "level": level_label,
        "episodes": episodes,
        "wins": len(wins),
        "losses": episodes - len(wins),
        "win_rate": win_rate,
        "x_position": {
            "mean": statistics.fmean(positions),
            "median": statistics.median(positions),
            "max": max(positions),
            "min": min(positions),
        },
        "duration_seconds": {
            "mean": statistics.fmean(durations),
            "median": statistics.median(durations),
            "max": max(durations),
        },
        "steps": {
            "mean": statistics.fmean([record.steps for record in records] or [0]),
            "max": max([record.steps for record in records] or [0]),
        },
        "death_reasons": reasons,
        "death_causes": causes,
        "power_ups": {
            "mushrooms": sum(record.mushrooms for record in records),
            "fire_flowers": sum(record.fire_flowers for record in records),
            "power_losses": sum(record.power_losses for record in records),
        },
        "action_counts": dict(zip(ACTION_NAMES, actions, strict=True)),
        "reward": {
            "shaped_mean": statistics.fmean([record.shaped_reward for record in records] or [0.0]),
            "raw_mean": statistics.fmean([record.raw_reward for record in records] or [0.0]),
        },
        "promotion": promotion_verdict(episodes, win_rate, promotable),
    }


def promotion_verdict(episodes: int, win_rate: float, promotable: bool = True) -> dict[str, object]:
    """State the promotion rule and whether the measurement satisfies it."""
    promoted = promotable and episodes >= PROMOTION_EPISODES and win_rate >= PROMOTION_WIN_RATE
    if not promotable:
        verdict = "not promoted: these episodes were not a real World 1-1 measurement"
    elif promoted:
        verdict = "promoted: the measured greedy evaluation meets the criterion"
    else:
        verdict = "not promoted: the measured greedy evaluation does not meet the criterion"
    return {
        "criterion": (
            f"{PROMOTION_EPISODES} deterministic no-cheat World 1-1 episodes with "
            f"a {PROMOTION_WIN_RATE:.0%} win rate"
        ),
        "required_episodes": PROMOTION_EPISODES,
        "required_win_rate": PROMOTION_WIN_RATE,
        "measured_episodes": episodes,
        "measured_win_rate": win_rate,
        "promoted": promoted,
        "verdict": verdict,
    }
