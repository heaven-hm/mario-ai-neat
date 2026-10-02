"""Mario-specific reward shaping built only from observable game events.

Every term is derived from what the environment actually reports:

* ``info["x_pos"]``   Mario's world x position, for forward progress.
* ``info["status"]``  ``small`` / ``tall`` / ``fireball``, which is the only
  honest power-up signal available. A rise to ``tall`` is a mushroom, a rise to
  ``fireball`` is a fire flower, and a fall is a loss of power.
* ``info["life"]``    a decrease means Mario lost a life.
* ``info["time"]``    reaching zero is an in-game time-out.
* ``info["flag_get"]`` the environment's own flag-capture flag.

Two anti-hacking rules are structural rather than tunable:

* Forward progress pays only for reaching a new furthest x within the episode,
  so oscillating back and forth, walking backwards, or re-crossing ground
  already covered earns nothing.
* Total forward shaping is capped per episode, so no amount of small jitter can
  out-earn finishing the level.

Nothing here rewards coins, score, or standing still, and no term is emitted for
a signal the environment does not provide: a missing field turns its component
off and is reported through ``signals`` instead of being replaced by a guess.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium

STATUS_RANK = {"small": 0, "tall": 1, "fireball": 2}

# The three events an episode can end on, as classified from public signals.
EVENT_FLAG = "flag"
EVENT_TIMEOUT = "timeout"
EVENT_DEATH = "death"
EVENT_TRUNCATED = "truncated"
EVENT_NONE = "none"

CAUSE_PIT = "pit"
CAUSE_HAZARD = "hazard"
CAUSE_UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the shaped reward. Ordering is asserted in the tests."""

    forward_scale: float = 0.01
    forward_step_cap: float = 0.25
    forward_episode_cap: float = 30.0
    mushroom_reward: float = 8.0
    fire_flower_reward: float = 15.0
    power_loss_penalty: float = -8.0
    death_penalty: float = -15.0
    timeout_penalty: float = -15.0
    time_penalty: float = -0.02
    flag_reward: float = 50.0

    def validate(self) -> None:
        if self.forward_scale < 0:
            raise ValueError("forward_scale must not be negative")
        if self.forward_step_cap <= 0 or self.forward_episode_cap <= 0:
            raise ValueError("forward caps must be positive")
        if self.flag_reward <= 0:
            raise ValueError("flag_reward must be positive, completing 1-1 is the goal")
        if self.death_penalty > 0 or self.timeout_penalty > 0:
            raise ValueError("death and time-out penalties must not be positive")
        if self.flag_reward <= max(self.mushroom_reward, self.fire_flower_reward):
            raise ValueError("flag capture must be worth more than any power-up")
        if self.time_penalty > 0:
            raise ValueError("the per-step time cost must not be positive")


@dataclass
class RewardTerms:
    """The shaped reward decomposed by cause, for logging and tests."""

    forward: float = 0.0
    mushroom: float = 0.0
    fire_flower: float = 0.0
    power_loss: float = 0.0
    death: float = 0.0
    timeout: float = 0.0
    flag: float = 0.0
    time: float = 0.0
    episode: int = 0

    @property
    def total(self) -> float:
        return (
            self.forward
            + self.mushroom
            + self.fire_flower
            + self.power_loss
            + self.death
            + self.timeout
            + self.flag
            + self.time
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MarioSignals:
    """Which reward inputs the environment actually provides."""

    x_pos: bool = False
    status: bool = False
    life: bool = False
    time: bool = False
    flag_get: bool = False
    viewport: bool = False

    MISSING_MEANS = {
        "x_pos": "no forward-progress reward",
        "status": "no power-up or power-loss reward",
        "life": "no life-loss death reward",
        "time": "no in-game time-out reward",
        "flag_get": "no level-completion reward",
        "viewport": "death cause is reported as unclassified",
    }

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(name for name in self.MISSING_MEANS if not getattr(self, name))

    def consequences(self) -> dict[str, str]:
        return {name: self.MISSING_MEANS[name] for name in self.missing}

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["missing"] = list(self.missing)
        payload["consequences"] = self.consequences()
        return payload


def detect_signals(info: dict, viewport: int | None) -> MarioSignals:
    """Record which signals exist, so absent ones can be reported honestly."""
    return MarioSignals(
        x_pos="x_pos" in info,
        status="status" in info,
        life="life" in info,
        time="time" in info,
        flag_get="flag_get" in info,
        viewport=viewport is not None,
    )


def classify_death_cause(viewport: int | None) -> str:
    """Name a death cause only when the environment exposes one.

    ``gym_super_mario_bros`` ends a stage episode on flag capture, on the dying
    animation, or on the dead state, and its own dying test is "state 0x0b *or*
    the vertical viewport dropped below the screen". A viewport above 1 is
    therefore a fall into a hole; any other end that is not a flag capture and
    not the in-game clock is something on screen that killed Mario. The emulator
    does not expose which hazard, so that is as far as this can honestly go.
    """
    if viewport is None:
        return CAUSE_UNCLASSIFIED
    if viewport > 1:
        return CAUSE_PIT
    return CAUSE_HAZARD


class MarioRewardWrapper(gymnasium.Wrapper):
    """Replace the environment reward with documented SMB1 event shaping.

    The environment's own reward is preserved in ``info["raw_reward"]`` on
    every step, and the shaped value and its components are exposed through
    ``info["shaped_reward"]`` and ``info["reward_terms"]`` so both can be logged
    side by side.
    """

    def __init__(self, env, config: RewardConfig | None = None):
        super().__init__(env)
        self.config = config or RewardConfig()
        self.config.validate()
        self.signals: MarioSignals | None = None
        self.event = EVENT_NONE
        self.death_cause = CAUSE_UNCLASSIFIED
        self._clear_episode()

    # MARK: Gymnasium API

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._clear_episode()
        if info:
            self._previous = dict(info)
            self._observe(info)
        return observation, info

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(raw_reward)
        terms = self._shape(info, terminated, truncated)
        self._raw_total += float(raw_reward)
        self._shaped_total += terms.total
        info["shaped_reward"] = terms.total
        info["reward_terms"] = terms.as_dict()
        info["mario_event"] = self.event
        info["death_cause"] = self.death_cause
        return observation, terms.total, terminated, truncated, info

    # MARK: Diagnostics

    @property
    def raw_reward_total(self) -> float:
        return self._raw_total

    @property
    def shaped_reward_total(self) -> float:
        return self._shaped_total

    @property
    def power_ups(self) -> tuple[int, int, int]:
        return self._mushrooms, self._fire_flowers, self._power_losses

    # MARK: Reward computation

    def _clear_episode(self) -> None:
        self._previous: dict | None = None
        self._best_x: int | None = None
        self._forward_earned = 0.0
        self._mushrooms = 0
        self._fire_flowers = 0
        self._power_losses = 0
        self._raw_total = 0.0
        self._shaped_total = 0.0
        self.event = EVENT_NONE
        self.death_cause = CAUSE_UNCLASSIFIED

    def _observe(self, info: dict) -> None:
        viewport = self._viewport()
        detected = detect_signals(info, viewport)
        self.signals = detected
        if self._best_x is None and "x_pos" in info:
            self._best_x = int(info["x_pos"])

    def _viewport(self) -> int | None:
        """Read the probe the environment adapter exposes, if it has one."""
        try:
            viewport = self.env.unwrapped.viewport
        except Exception:
            return None
        return None if viewport is None else int(viewport)

    def _shape(self, info: dict, terminated: bool, truncated: bool) -> RewardTerms:
        terms = RewardTerms()
        if self._previous is None:
            # A legacy reset may not describe the episode's first state; take the
            # baseline from this step and pay nothing for it.
            self._previous = dict(info)
            self._observe(info)
            return terms

        previous = self._previous
        self._observe(info)

        flag = bool(info.get("flag_get", False))
        clock = self._int(info, "time")
        previous_life = self._int(previous, "life")
        life = self._int(info, "life")
        life_lost = previous_life is not None and life is not None and life < previous_life
        timed_out = clock is not None and clock <= 0
        died = not flag and (terminated or life_lost)

        forward = self._forward(info)
        if forward:
            terms.forward = forward

        previous_rank = STATUS_RANK.get(str(previous.get("status", "")))
        rank = STATUS_RANK.get(str(info.get("status", "")))
        if rank is not None and previous_rank is not None:
            if rank > previous_rank:
                if rank == STATUS_RANK["fireball"]:
                    terms.fire_flower = self.config.fire_flower_reward
                    self._fire_flowers += 1
                else:
                    terms.mushroom = self.config.mushroom_reward
                    self._mushrooms += 1
            elif rank < previous_rank and not died:
                # A death already costs the death penalty and drops power with
                # it; charging both would double-count one event.
                terms.power_loss = self.config.power_loss_penalty
                self._power_losses += 1

        terms.time = self.config.time_penalty

        if flag:
            terms.flag = self.config.flag_reward
            self.event = EVENT_FLAG
            self.death_cause = CAUSE_UNCLASSIFIED
        elif timed_out and terminated:
            terms.timeout = self.config.timeout_penalty
            self.event = EVENT_TIMEOUT
            self.death_cause = CAUSE_UNCLASSIFIED
        elif died:
            terms.death = self.config.death_penalty
            self.event = EVENT_DEATH
            self.death_cause = classify_death_cause(self._viewport())
        elif truncated:
            self.event = EVENT_TRUNCATED
            self.death_cause = CAUSE_UNCLASSIFIED
        else:
            self.event = EVENT_NONE
            self.death_cause = CAUSE_UNCLASSIFIED

        self._previous = dict(info)
        return terms

    def _forward(self, info: dict) -> float:
        """Pay only for reaching a new furthest x, up to the episode cap."""
        current = self._int(info, "x_pos")
        if current is None:
            return 0.0
        if self._best_x is None:
            self._best_x = current
            return 0.0
        gain = current - self._best_x
        if gain <= 0:
            return 0.0
        self._best_x = current
        remaining = self.config.forward_episode_cap - self._forward_earned
        if remaining <= 0:
            return 0.0
        reward = min(gain * self.config.forward_scale, self.config.forward_step_cap, remaining)
        self._forward_earned += reward
        return reward

    @staticmethod
    def _int(info: dict, key: str) -> int | None:
        value = info.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
