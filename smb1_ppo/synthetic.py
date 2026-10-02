"""A synthetic stand-in for the SMB1 environment contract.

This exists so the PPO pipeline, logging, checkpointing, and evaluation can be
tested and smoke-run on a machine that has no Super Mario Bros. ROM, which is
the normal situation in CI. It reproduces the parts of
``gym_super_mario_bros`` that this package depends on: a 240x256x3 uint8 frame,
the same ten ``info`` keys, the same raw-reward formula, and episodes that end
on flag capture, a death, or a time-out.

It is **not** a benchmark environment. It is never a substitute for the real
ROM, and the evaluation report refuses to mark a synthetic run as promoted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium
import numpy as np

SCREEN_HEIGHT = 240
SCREEN_WIDTH = 256
START_X = 40
LEVEL_X = 3150
SCREEN_X_OFFSET = 96
STATUS_ORDER = {"small": 0, "tall": 1, "fireball": 2}
POWER_UP_SCORE = 1000


@dataclass(frozen=True)
class SyntheticScript:
    """The scripted episode the synthetic environment plays out."""

    max_decisions: int = 2500
    step_x: int = 3
    time_units: int = 400
    decisions_per_time_unit: int = 2
    power_events: tuple[tuple[int, str], ...] = (
        (150, "tall"),
        (400, "fireball"),
        (650, "small"),
    )
    flag_x: int | None = LEVEL_X
    death_x: int | None = None
    pit_death: bool = True
    life_loss_on_death: bool = False
    drop_signals: tuple[str, ...] = field(default=())

    def with_death(self, death_x: int, pit: bool = True) -> SyntheticScript:
        return SyntheticScript(
            max_decisions=self.max_decisions,
            step_x=self.step_x,
            time_units=self.time_units,
            decisions_per_time_unit=self.decisions_per_time_unit,
            power_events=self.power_events,
            flag_x=None,
            death_x=death_x,
            pit_death=pit,
            life_loss_on_death=self.life_loss_on_death,
            drop_signals=self.drop_signals,
        )


def _background() -> np.ndarray:
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = 104
    frame[96:, :, 0] = 172
    frame[96:, :, 1] = 92
    frame[96:, :, 2] = 46
    frame[196:, :, 0] = 152
    frame[196:, :, 1] = 92
    frame[196:, :, 2] = 46
    return frame


class SyntheticSmb1Env(gymnasium.Env):
    """A deterministic, ROM-free environment with the SMB1 contract."""

    metadata = {"render_modes": []}
    observation_space = gymnasium.spaces.Box(
        low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
    )
    action_space = gymnasium.spaces.Discrete(7)

    def __init__(self, script: SyntheticScript | None = None):
        self.script = script or SyntheticScript()
        self._background = _background()
        self._x = START_X
        self._last_x = START_X
        self._last_time = self.script.time_units
        self._reset_state()

    # MARK: Gymnasium API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._frame(), self._info()

    def step(self, action):
        action = int(action)
        walking = action in (1, 2, 3, 4)
        self._decisions += 1
        self._last_x = self._x
        self._last_time = self._time
        self._x += self.script.step_x if walking else 0
        if self._decisions % self.script.decisions_per_time_unit == 0:
            self._time = max(0, self._time - 1)
        gained_power = False
        for threshold, status in self.script.power_events:
            if walking and self._x >= threshold and STATUS_ORDER[status] > STATUS_ORDER[self._status]:
                self._status = status
                gained_power = True
        if gained_power:
            self._score += POWER_UP_SCORE

        flag = self.script.flag_x is not None and self._x >= self.script.flag_x
        death = self.script.death_x is not None and self._x >= self.script.death_x
        timeout = self._time <= 0
        terminated = bool(flag or death or timeout)
        truncated = (not terminated) and self._decisions >= self.script.max_decisions
        if death:
            if self.script.life_loss_on_death:
                self._life = max(0, self._life - 1)
            self._viewport = 2 if self.script.pit_death else 1
        if flag:
            self._x = self.script.flag_x or self._x
        if terminated or truncated:
            self._dying = bool(death)
        return self._frame(), self._reward(), terminated, truncated, self._info()

    @property
    def viewport(self) -> int:
        """Mirror SMB1's vertical viewport byte, which is 2 after a pit fall."""
        return self._viewport

    # MARK: State

    def _reset_state(self) -> None:
        self._x = START_X
        self._last_x = START_X
        self._time = self.script.time_units
        self._last_time = self._time
        self._status = "small"
        self._life = 2
        self._score = 0
        self._decisions = 0
        self._dying = False
        self._viewport = 1

    def _reward(self) -> float:
        """Reproduce gym-super-mario-bros' own reward formula."""
        delta = self._x - self._last_x
        reward = 0.0 if delta < -5 or delta > 5 else float(delta)
        reward += min(0, self._time - self._last_time)
        if self._dying:
            reward -= 25.0
        return reward

    def _info(self) -> dict[str, object]:
        info: dict[str, object] = {
            "coins": 0,
            "flag_get": self.script.flag_x is not None and self._x >= self.script.flag_x,
            "life": self._life,
            "score": self._score,
            "stage": 1,
            "status": self._status,
            "time": self._time,
            "world": 1,
            "x_pos": self._x,
            "y_pos": 79,
        }
        for key in self.script.drop_signals:
            info.pop(key, None)
        return info

    def _frame(self) -> np.ndarray:
        frame = self._background.copy()
        offset = max(0, self._x - SCREEN_X_OFFSET)
        screen_x = int(np.clip(self._x - offset, 0, SCREEN_WIDTH - 16))
        frame[176:192, screen_x : screen_x + 16, 0] = 252
        frame[176:192, screen_x : screen_x + 16, 1] = 216
        frame[176:192, screen_x : screen_x + 16, 2] = 168
        flag_x = int(np.clip((self.script.flag_x or LEVEL_X) - offset, 0, SCREEN_WIDTH - 2))
        frame[80:192, flag_x : flag_x + 2, 0] = 60
        frame[80:192, flag_x : flag_x + 2, 1] = 236
        frame[80:192, flag_x : flag_x + 2, 2] = 60
        progress = int(np.clip(self._x / LEVEL_X * 64, 0, 64))
        frame[8:16, 8 : 8 + progress, :] = 255
        return frame
