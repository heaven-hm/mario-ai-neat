"""Observation pipeline shape, dtype, and range, wrapper by wrapper."""

from __future__ import annotations

import unittest

import gymnasium
import numpy as np

from smb1_ppo.env import EnvConfig, NoProgressLimit, make_env
from smb1_ppo.wrappers import (
    FRAME_SIZE,
    FRAME_STACK,
    FrameSkipMaxPool,
    FrameStack,
    Grayscale,
    Normalize,
    Resize,
)
from support import FrameEnv


class FlickerEnv(FrameEnv):
    """Alternate a bright and a dark frame, the way an NES flickers sprites."""

    def _frame(self) -> np.ndarray:
        value = 220 if self.steps % 2 == 0 else 30
        return np.full((8, 8, 3), value, dtype=np.uint8)


class StalledEnv(FrameEnv):
    """Report a constant x position so progress never improves."""

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["x_pos"] = 100
        return observation, reward, terminated, truncated, info


class FrameSkipMaxPoolTests(unittest.TestCase):
    def test_accumulates_reward_across_the_skipped_frames(self) -> None:
        env = FrameSkipMaxPool(FrameEnv(reward=1.0), skip=4)
        env.reset()
        _, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual(reward, 4.0)
        self.assertFalse(terminated or truncated)

    def test_max_pools_the_last_two_frames_to_remove_flicker(self) -> None:
        env = FrameSkipMaxPool(FlickerEnv(), skip=3)
        env.reset()
        observation, _, _, _, _ = env.step(0)
        # Frames 1..3 are 30, 220, 30; pooling frames 2 and 3 keeps the sprite.
        self.assertEqual(int(observation[0, 0, 0]), 220)

    def test_stops_early_when_the_episode_ends_mid_skip(self) -> None:
        env = FrameSkipMaxPool(FrameEnv(done_at=2), skip=4)
        env.reset()
        _, reward, terminated, truncated, _ = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 2.0)

    def test_rejects_a_zero_skip(self) -> None:
        with self.assertRaises(ValueError):
            FrameSkipMaxPool(FrameEnv(), skip=0)


class GrayscaleTests(unittest.TestCase):
    def test_luma_weights_and_single_channel(self) -> None:
        env = Grayscale(FrameEnv())
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]
        luma = env.observation(frame)
        self.assertEqual(luma.shape, (2, 2, 1))
        self.assertEqual(luma.dtype, np.uint8)
        self.assertEqual(int(luma[0, 0, 0]), round(0.299 * 255))
        self.assertEqual(env.observation_space.shape, (8, 8, 1))
        self.assertEqual(env.observation_space.dtype, np.uint8)

    def test_passes_through_an_already_gray_frame(self) -> None:
        env = Grayscale(FrameEnv())
        frame = np.full((4, 4, 1), 7, dtype=np.uint8)
        self.assertEqual(env.observation(frame).tolist(), frame.tolist())
        flat = np.full((4, 4), 7, dtype=np.uint8)
        self.assertEqual(env.observation(flat).shape, (4, 4, 1))


class ResizeTests(unittest.TestCase):
    def test_resizes_to_84_and_keeps_uint8(self) -> None:
        env = Resize(Grayscale(FrameEnv()), size=FRAME_SIZE)
        observation, _ = env.reset()
        self.assertEqual(observation.shape, (FRAME_SIZE, FRAME_SIZE, 1))
        self.assertEqual(observation.dtype, np.uint8)
        self.assertEqual(env.observation_space.shape, (FRAME_SIZE, FRAME_SIZE, 1))

    def test_is_a_no_op_at_the_target_size(self) -> None:
        env = Resize(Grayscale(FrameEnv()), size=FRAME_SIZE)
        frame = np.full((FRAME_SIZE, FRAME_SIZE, 1), 3, dtype=np.uint8)
        self.assertTrue(np.array_equal(env.observation(frame), frame))


class NormalizeTests(unittest.TestCase):
    def test_scales_8_bit_frames_to_the_unit_interval(self) -> None:
        env = Normalize(Resize(Grayscale(FrameEnv(frame_value=235)), size=FRAME_SIZE))
        observation, _ = env.reset()
        self.assertEqual(observation.dtype, np.float32)
        self.assertGreaterEqual(float(observation.min()), 0.0)
        self.assertLessEqual(float(observation.max()), 1.0)
        # A plain 235 frame must not survive as 235; it has been scaled.
        self.assertGreater(float(observation.max()), 0.85)
        self.assertEqual(env.observation_space.dtype, np.float32)
        self.assertEqual(
            (float(env.observation_space.low.min()), float(env.observation_space.high.max())), (0.0, 1.0)
        )


class FrameStackTests(unittest.TestCase):
    def _environment(self, count: int):
        return FrameStack(Normalize(Resize(Grayscale(FrameEnv()), size=FRAME_SIZE)), count=count)

    def test_stacks_channel_first_and_repeats_the_first_frame_on_reset(self) -> None:
        env = self._environment(FRAME_STACK)
        observation, _ = env.reset()
        self.assertEqual(observation.shape, (FRAME_STACK, FRAME_SIZE, FRAME_SIZE))
        self.assertEqual(observation.dtype, np.float32)
        for index in range(1, FRAME_STACK):
            self.assertTrue(np.array_equal(observation[0], observation[index]))
        self.assertEqual(env.observation_space.shape, (FRAME_STACK, FRAME_SIZE, FRAME_SIZE))

    def test_newest_frame_is_last(self) -> None:
        env = self._environment(3)
        env.reset()
        observation, _, _, _, _ = env.step(0)
        self.assertEqual(observation.shape[0], 3)
        self.assertFalse(np.array_equal(observation[0], observation[-1]))

    def test_rejects_a_zero_stack(self) -> None:
        with self.assertRaises(ValueError):
            FrameStack(FrameEnv(), count=0)


class NoProgressLimitTests(unittest.TestCase):
    def test_truncates_a_stalled_episode(self) -> None:
        env = NoProgressLimit(StalledEnv(), frames=4)
        env.reset()
        truncated_seen = False
        for _ in range(6):
            _, _, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated)
            truncated_seen = truncated_seen or truncated
        self.assertTrue(truncated_seen)

    def test_does_not_truncate_while_progressing(self) -> None:
        env = NoProgressLimit(FrameEnv(), frames=4)
        env.reset()
        for _ in range(8):
            _, _, _, truncated, _ = env.step(0)
            self.assertFalse(truncated)

    def test_rejects_a_zero_window(self) -> None:
        with self.assertRaises(ValueError):
            NoProgressLimit(FrameEnv(), frames=0)


class FullPipelineTests(unittest.TestCase):
    def test_full_pipeline_shape_dtype_and_range(self) -> None:
        env = make_env(EnvConfig(environment="synthetic"))()
        try:
            observation, info = env.reset()
            self.assertEqual(observation.shape, (FRAME_STACK, FRAME_SIZE, FRAME_SIZE))
            self.assertEqual(observation.dtype, np.float32)
            self.assertGreaterEqual(float(observation.min()), 0.0)
            self.assertLessEqual(float(observation.max()), 1.0)
            self.assertIsInstance(env.observation_space, gymnasium.spaces.Box)
            self.assertIsInstance(env.action_space, gymnasium.spaces.Discrete)
            self.assertEqual(env.observation_space.shape, observation.shape)
            self.assertEqual(int(env.action_space.n), 7)
            self.assertIn("x_pos", info)
            for _ in range(3):
                observation, reward, terminated, truncated, info = env.step(1)
                self.assertEqual(observation.shape, (FRAME_STACK, FRAME_SIZE, FRAME_SIZE))
                self.assertEqual(observation.dtype, np.float32)
                self.assertIsInstance(reward, float)
                self.assertIn("reward_terms", info)
        finally:
            env.close()

    def test_frame_skip_defines_how_many_emulated_frames_a_decision_covers(self) -> None:
        env = make_env(EnvConfig(environment="synthetic", frame_skip=2))()
        try:
            env.reset()
            _, _, _, _, info = env.step(1)
            self.assertEqual(info["x_pos"], 40 + 3 * 2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
