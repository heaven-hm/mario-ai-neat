"""Observation pipeline: skip, max-pool, grayscale, resize, normalize, stack.

Each concern is its own wrapper so it can be tested in isolation, and the
pipeline is assembled in ``env.py``. Order matters and follows the standard
Atari recipe:

1. ``FrameSkipMaxPool``  repeat the action and max-pool consecutive frames,
   which removes the NES's alternating-frame flicker instead of teaching the
   CNN about it.
2. ``Grayscale``         colour carries no task information here.
3. ``Resize``            84x84, the size the CNN expects.
4. ``Normalize``         float32 in [0, 1].
5. ``FrameStack``        four consecutive frames, channel-first, so velocity
   and jump arcs are observable.

All observation wrappers produce Gymnasium observation spaces, and all of them
operate on whatever array the layer below returns, so they work identically on
the real ROM environment and on the synthetic one used by tests.
"""

from __future__ import annotations

from collections import deque

import cv2
import gymnasium
import numpy as np

FRAME_SIZE = 84
FRAME_STACK = 4
FRAME_SKIP = 4
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class FrameSkipMaxPool(gymnasium.Wrapper):
    """Repeat each action for ``skip`` frames and max-pool consecutive frames."""

    def __init__(self, env, skip: int = FRAME_SKIP):
        super().__init__(env)
        if skip < 1:
            raise ValueError(f"frame skip must be at least 1, got {skip}")
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info: dict[str, object] = {}
        observation = None
        for index in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if index >= self.skip - 2 and self._previous is not None:
                observation = np.maximum(observation, self._previous)
            self._previous = observation
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._previous = observation
        return observation, info

    _previous: np.ndarray | None = None


class Grayscale(gymnasium.ObservationWrapper):
    """Convert an HxWx3 RGB frame to an HxWx1 luma frame."""

    def __init__(self, env):
        super().__init__(env)
        shape = (*self.observation_space.shape[:2], 1)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, observation):
        frame = np.asarray(observation)
        if frame.ndim == 2:
            return frame[:, :, None]
        if frame.shape[-1] == 1:
            return frame
        luma = frame[..., :3].astype(np.float32) @ LUMA_WEIGHTS
        return np.clip(luma + 0.5, 0, 255).astype(np.uint8)[:, :, None]


class Resize(gymnasium.ObservationWrapper):
    """Resize a frame to ``size`` x ``size`` with area interpolation."""

    def __init__(self, env, size: int = FRAME_SIZE):
        super().__init__(env)
        if size < 1:
            raise ValueError(f"frame size must be positive, got {size}")
        self.size = size
        shape = (*self.observation_space.shape[:2], 1)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(size, size, shape[-1]), dtype=np.uint8
        )

    def observation(self, observation):
        frame = np.asarray(observation)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.shape[:2] == (self.size, self.size):
            return frame
        resized = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return resized[:, :, None] if resized.ndim == 2 else resized


class Normalize(gymnasium.ObservationWrapper):
    """Scale an 8-bit frame to float32 in [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)

    def observation(self, observation):
        frame = np.asarray(observation)
        if frame.dtype == np.uint8:
            return frame.astype(np.float32) / 255.0
        return np.clip(frame.astype(np.float32), 0.0, 1.0)


class FrameStack(gymnasium.ObservationWrapper):
    """Stack the last ``count`` frames on the channel axis (C, H, W)."""

    def __init__(self, env, count: int = FRAME_STACK):
        super().__init__(env)
        if count < 1:
            raise ValueError(f"frame stack must be at least 1, got {count}")
        self.count = count
        height, width = self.observation_space.shape[:2]
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(count, height, width), dtype=np.float32
        )
        self._frames: deque[np.ndarray] = deque(maxlen=count)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.count):
            self._frames.append(self._to_plane(observation))
        return self._stacked(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(self._to_plane(observation))
        return self._stacked(), reward, terminated, truncated, info

    def observation(self, observation):  # pragma: no cover - replaced above
        return observation

    def _to_plane(self, observation) -> np.ndarray:
        frame = np.asarray(observation, dtype=np.float32)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[:, :, 0]
        return frame

    def _stacked(self) -> np.ndarray:
        return np.stack(tuple(self._frames), axis=0).astype(np.float32)
