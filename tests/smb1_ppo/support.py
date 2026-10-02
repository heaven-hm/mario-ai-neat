"""Test doubles implementing the environment contracts this package relies on."""

from __future__ import annotations

import gymnasium
import numpy as np

SMB1_INFO_KEYS = (
    "coins",
    "flag_get",
    "life",
    "score",
    "stage",
    "status",
    "time",
    "world",
    "x_pos",
    "y_pos",
)


class FrameEnv(gymnasium.Env):
    """Emit small, changing RGB frames so wrappers can be checked in isolation."""

    metadata = {"render_modes": []}
    observation_space = gymnasium.spaces.Box(0, 255, (8, 8, 3), np.uint8)
    action_space = gymnasium.spaces.Discrete(7)

    def __init__(self, frame_value: int = 10, done_at: int | None = None, reward: float = 1.0):
        self.frame_value = frame_value
        self.done_at = done_at
        self.reward = reward
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._frame(), {}

    def step(self, action):
        self.steps += 1
        terminated = self.done_at is not None and self.steps >= self.done_at
        return self._frame(), self.reward, terminated, False, {"x_pos": 100 + self.steps}

    def _frame(self) -> np.ndarray:
        frame = np.full((8, 8, 3), self.frame_value + self.steps, dtype=np.uint8)
        frame[0, 0] = [255, 255, 255]
        return frame


class LegacyGymEnv:
    """A legacy Gym-API environment: bare reset and a four-tuple step."""

    observation_space = None
    action_space = None

    def __init__(self, info: dict):
        self.info = info
        self.steps = 0
        self.closed = False

    def reset(self, seed=None):
        self.steps = 0
        self.seed_used = seed
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= 2
        info = dict(self.info)
        info["x_pos"] = self.info.get("x_pos", 0) + 10 * self.steps
        return np.zeros((2, 2, 3), dtype=np.uint8), 1.5, terminated, info

    def close(self):
        self.closed = True

    def render(self, mode="human"):
        return f"rendered:{mode}"


def legacy_spaces():
    """Legacy Gym spaces, the type gym-super-mario-bros actually exposes."""
    import gym

    return gym.spaces.Box(0, 255, (2, 2, 3), np.uint8), gym.spaces.Discrete(7)


class ScriptedEnv(gymnasium.Env):
    """Yield a scripted sequence of SMB1-style transitions.

    Each scripted step is ``(info, terminated, truncated, reward)``. The reward
    wrapper is the unit under test, so the environment only has to report
    exactly what the real one reports.
    """

    metadata = {"render_modes": []}
    observation_space = gymnasium.spaces.Box(0, 255, (2, 2, 3), np.uint8)
    action_space = gymnasium.spaces.Discrete(7)

    def __init__(self, reset_info: dict, steps: list[tuple], viewport: int | None = 1):
        self.reset_info = reset_info
        self.steps = list(steps)
        self.viewport = viewport
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self._observation(), dict(self.reset_info)

    def step(self, action):
        info, terminated, truncated, reward = self.steps[self.index]
        self.index += 1
        return self._observation(), reward, terminated, truncated, dict(info)

    def _observation(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)


def smb1_info(**overrides) -> dict:
    """A complete SMB1 info dict, with overrides applied."""
    info = {
        "coins": 0,
        "flag_get": False,
        "life": 2,
        "score": 0,
        "stage": 1,
        "status": "small",
        "time": 400,
        "world": 1,
        "x_pos": 40,
        "y_pos": 79,
    }
    info.update(overrides)
    return info
