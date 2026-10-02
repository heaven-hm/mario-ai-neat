"""The environment contract: legacy-API adaptation, spaces, config, and the loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium
import numpy as np

from smb1_ppo.compat import GymnasiumApiAdapter, to_gymnasium_space
from smb1_ppo.env import (
    ENVIRONMENT_SYNTHETIC,
    EnvConfig,
    LevelSpec,
    SeedOnReset,
    describe_environment,
    device_caveat,
    make_env,
    make_vec_env,
    observation_shape,
    raw_env_kwargs,
    resolve_device,
    start_method,
)
from smb1_ppo.rom import RomError
from smb1_ppo.synthetic import LEVEL_X, SyntheticScript
from support import (
    GymnasiumStyleGymEnv,
    LegacyGymEnv,
    environment_spaces,
    legacy_gym_available,
    legacy_spaces,
    smb1_info,
)


class LevelSpecTests(unittest.TestCase):
    def test_world_1_1_is_the_default_target(self) -> None:
        level = LevelSpec()
        self.assertEqual((level.world, level.stage), (1, 1))
        self.assertEqual(level.label, "1-1")
        self.assertEqual(level.target, (1, 1))

    def test_later_world_1_levels_are_configurable(self) -> None:
        for stage in range(1, 5):
            self.assertEqual(LevelSpec(world=1, stage=stage).label, f"1-{stage}")

    def test_out_of_range_levels_are_rejected(self) -> None:
        for world, stage in ((0, 1), (9, 1), (1, 0), (1, 5)):
            with self.assertRaises(ValueError):
                LevelSpec(world=world, stage=stage)


class EnvConfigTests(unittest.TestCase):
    def test_unknown_environment_kind_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnvConfig(environment="smw")

    def test_a_negative_no_progress_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnvConfig(no_progress_frames=-1)

    def test_observation_shape_matches_the_pipeline(self) -> None:
        config = EnvConfig(environment=ENVIRONMENT_SYNTHETIC, frame_stack=4, frame_size=84)
        self.assertEqual(observation_shape(config), (4, 84, 84))

    def test_a_smb1_environment_without_a_rom_explains_how_to_import_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = EnvConfig(rom=Path(directory) / "absent.nes")
            with self.assertRaises(RomError) as context:
                make_env(config)()
            message = str(context.exception)
            self.assertIn("python -m smb1_ppo.rom import", message)
            self.assertIn("SMB1_PPO_ROM", message)


class GymnasiumApiAdapterTests(unittest.TestCase):
    def _adapter(self):
        observation_space, action_space = environment_spaces()
        env = LegacyGymEnv(smb1_info(x_pos=100))
        env.observation_space = observation_space
        env.action_space = action_space
        return GymnasiumApiAdapter(env)

    def test_spaces_are_gymnasium_types_after_the_adapter(self) -> None:
        env = self._adapter()
        self.assertIsInstance(env.observation_space, gymnasium.spaces.Box)
        self.assertIsInstance(env.action_space, gymnasium.spaces.Discrete)
        self.assertEqual(env.observation_space.dtype, np.uint8)
        self.assertEqual(int(env.action_space.n), 7)

    @unittest.skipUnless(legacy_gym_available(), "the legacy gym package is not installed")
    def test_legacy_gym_spaces_are_converted(self) -> None:
        observation_space, action_space = legacy_spaces()
        converted_observation = to_gymnasium_space(observation_space)
        converted_action = to_gymnasium_space(action_space)
        self.assertIsInstance(converted_observation, gymnasium.spaces.Box)
        self.assertIsInstance(converted_action, gymnasium.spaces.Discrete)
        self.assertEqual(converted_observation.shape, (2, 2, 3))
        self.assertEqual(int(converted_action.n), 7)

    def test_reset_returns_an_observation_and_info_pair(self) -> None:
        env = self._adapter()
        result = env.reset(seed=7)
        self.assertIsInstance(result, tuple)
        observation, info = result
        self.assertEqual(observation.shape, (2, 2, 3))
        self.assertIsInstance(info, dict)

    def test_step_returns_the_gymnasium_five_tuple(self) -> None:
        env = self._adapter()
        env.reset()
        result = env.step(1)
        self.assertEqual(len(result), 5)
        observation, reward, terminated, truncated, info = result
        self.assertEqual(reward, 1.5)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["x_pos"], 110)
        _, _, terminated, _, info = env.step(1)
        self.assertTrue(terminated)
        self.assertEqual(info["x_pos"], 120)

    def test_truncation_is_taken_from_the_time_limit_info_key(self) -> None:
        observation_space, action_space = environment_spaces()
        inner = LegacyGymEnv(smb1_info())
        inner.observation_space = observation_space
        inner.action_space = action_space
        original = inner.step

        def fake_step(action):
            observation, reward, done, info = original(action)
            info["TimeLimit.truncated"] = True
            return observation, reward, done, info

        inner.step = fake_step
        env = GymnasiumApiAdapter(inner)
        env.reset()
        _, _, terminated, truncated, info = env.step(0)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertNotIn("TimeLimit.truncated", info)

    def test_viewport_is_unavailable_when_the_environment_has_no_ram(self) -> None:
        self.assertIsNone(self._adapter().viewport)

    def test_rendering_uses_the_signature_the_environment_exposes(self) -> None:
        self.assertEqual(self._adapter().render(), "rendered:human")
        gymnasium_style = GymnasiumStyleGymEnv()
        self.assertEqual(GymnasiumApiAdapter(gymnasium_style).render(), "rendered")

    def test_a_gymnasium_space_passes_through_unchanged(self) -> None:
        space = gymnasium.spaces.Box(0, 1, (2,), np.float32)
        self.assertIs(to_gymnasium_space(space), space)

    def test_an_unconvertible_space_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            to_gymnasium_space([1, 2, 3])


class SyntheticContractTests(unittest.TestCase):
    """The stand-in must report exactly what the real environment reports."""

    def test_info_keys_match_the_real_environment(self) -> None:
        env = make_env(EnvConfig(environment=ENVIRONMENT_SYNTHETIC))()
        try:
            _, info = env.reset()
            for key in smb1_info():
                self.assertIn(key, info, key)
        finally:
            env.close()

    def test_raw_reward_matches_the_upstream_formula(self) -> None:
        from smb1_ppo.synthetic import SyntheticSmb1Env

        environment = SyntheticSmb1Env()
        environment.reset()
        _, reward, _, _, _ = environment.step(1)
        # Walking 3 px pays 3, and one clock tick costs 1 every second decision.
        self.assertEqual(reward, 3.0)

    def test_episodes_can_end_on_flag_death_and_timeout(self) -> None:
        from smb1_ppo.synthetic import SyntheticSmb1Env

        flag = SyntheticSmb1Env(SyntheticScript(flag_x=100, max_decisions=200))
        flag.reset()
        for _ in range(20):
            _, _, terminated, _, info = flag.step(1)
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertTrue(info["flag_get"])

        death = SyntheticSmb1Env(SyntheticScript().with_death(death_x=100))
        death.reset()
        for _ in range(20):
            _, _, terminated, _, _ = death.step(1)
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertEqual(death.viewport, 2)

        timeout = SyntheticSmb1Env(SyntheticScript(flag_x=None, time_units=2, decisions_per_time_unit=1))
        timeout.reset()
        for _ in range(5):
            _, _, terminated, _, info = timeout.step(1)
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertEqual(info["time"], 0)

    def test_the_synthetic_level_is_long_enough_to_measure(self) -> None:
        self.assertGreater(LEVEL_X, 3000)


class VectorizationTests(unittest.TestCase):
    def test_single_worker_uses_a_dummy_vector_environment(self) -> None:
        environment = make_vec_env(EnvConfig(environment=ENVIRONMENT_SYNTHETIC), workers=1)
        try:
            self.assertEqual(environment.num_envs, 1)
            self.assertEqual(environment.reset().shape, (1, 4, 84, 84))
        finally:
            environment.close()

    def test_two_workers_use_independent_processes(self) -> None:
        environment = make_vec_env(EnvConfig(environment=ENVIRONMENT_SYNTHETIC), workers=2)
        try:
            self.assertEqual(environment.num_envs, 2)
            observations = environment.reset()
            self.assertEqual(observations.shape, (2, 4, 84, 84))
            observations, rewards, dones, infos = environment.step(np.array([1, 1]))
            self.assertEqual(observations.shape, (2, 4, 84, 84))
            self.assertEqual(len(rewards), 2)
            self.assertEqual(len(infos), 2)
        finally:
            environment.close()

    def test_zero_workers_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_vec_env(EnvConfig(environment=ENVIRONMENT_SYNTHETIC), workers=0)

    def test_start_method_is_safe_on_every_platform(self) -> None:
        self.assertIsNone(start_method(1))
        with mock.patch("smb1_ppo.env.platform.system", return_value="Windows"):
            self.assertEqual(start_method(8), "spawn")
        with mock.patch("smb1_ppo.env.platform.system", return_value="Darwin"):
            self.assertEqual(start_method(8), "spawn")
        with mock.patch("smb1_ppo.env.platform.system", return_value="Linux"):
            self.assertEqual(start_method(8), "forkserver")


class ConstructorArgumentTests(unittest.TestCase):
    """The two gym-super-mario-bros layouts take different keyword arguments."""

    def test_the_legacy_layout_takes_rom_mode_and_no_render_mode(self) -> None:
        legacy = {"self", "rom_mode", "lost_levels", "target"}
        self.assertEqual(
            raw_env_kwargs(legacy, LevelSpec(1, 2)),
            {"rom_mode": "vanilla", "lost_levels": False, "target": (1, 2)},
        )

    def test_the_gymnasium_layout_takes_render_mode(self) -> None:
        modern = {"self", "lost_levels", "target", "render_mode"}
        self.assertEqual(
            raw_env_kwargs(modern, LevelSpec(1, 3), "human"),
            {"lost_levels": False, "target": (1, 3), "render_mode": "human"},
        )

    def test_render_mode_is_omitted_when_the_environment_has_no_such_parameter(self) -> None:
        self.assertNotIn("render_mode", raw_env_kwargs({"self", "lost_levels", "target"}, LevelSpec()))

    def test_lost_levels_is_never_requested(self) -> None:
        for parameters in (
            {"self", "rom_mode", "lost_levels", "target"},
            {"self", "lost_levels", "target", "render_mode"},
        ):
            self.assertFalse(raw_env_kwargs(parameters, LevelSpec())["lost_levels"])

    def test_seed_on_reset_pins_the_episode_seed(self) -> None:
        environment = SeedOnReset(make_env(EnvConfig(environment=ENVIRONMENT_SYNTHETIC))(), seed=99)
        try:
            environment.reset()
            self.assertEqual(environment.seed_value, 99)
        finally:
            environment.close()


class DeviceTests(unittest.TestCase):
    def test_cpu_is_always_available(self) -> None:
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_auto_resolves_to_an_available_device(self) -> None:
        self.assertIn(resolve_device("auto"), ("cpu", "cuda", "mps"))
        self.assertEqual(resolve_device(None), resolve_device("auto"))

    def test_requesting_an_absent_accelerator_fails_loudly(self) -> None:
        import torch

        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")

    def test_mps_carries_a_documented_caveat(self) -> None:
        self.assertIsNotNone(device_caveat("mps"))
        self.assertIsNone(device_caveat("cpu"))

    def test_environment_description_records_the_versions(self) -> None:
        description = describe_environment()
        for key in (
            "python",
            "platform",
            "torch",
            "stable_baselines3",
            "gymnasium",
            "gym_super_mario_bros",
            "nes_py",
            "numpy",
            "opencv",
        ):
            self.assertIn(key, description)


class PpoIntegrationTests(unittest.TestCase):
    """A real PPO update, rollout logging, checkpoint, and evaluation."""

    def test_ppo_trains_and_evaluates_on_the_synthetic_contract(self) -> None:
        from stable_baselines3 import PPO

        from smb1_ppo.checkpoints import mark_best_model, read_best_model_evaluation
        from smb1_ppo.env import make_vec_env
        from smb1_ppo.evaluate import build_evaluation_environment, collect_episodes
        from smb1_ppo.stats import EpisodeOutcomeCallback, RewardStatsVecEnv

        config = EnvConfig(environment=ENVIRONMENT_SYNTHETIC, max_episode_steps=60)
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "run"
            run_directory.mkdir()
            stats_env = RewardStatsVecEnv(
                make_vec_env(config, workers=1), config.level, frame_skip=config.frame_skip
            )
            model = PPO(
                "CnnPolicy",
                stats_env,
                n_steps=64,
                batch_size=64,
                n_epochs=1,
                policy_kwargs={"normalize_images": False, "features_extractor_kwargs": {"features_dim": 32}},
                seed=3,
                verbose=0,
            )
            model.learn(total_timesteps=128, callback=EpisodeOutcomeCallback(stats_env), progress_bar=False)
            self.assertGreaterEqual(model.num_timesteps, 128)
            stats = stats_env.drain()
            self.assertGreater(stats.episodes_total, 0)
            self.assertGreater(stats.mean_max_x_total, 0.0)

            evaluation = {"win_rate": 0.0, "x_position": {"mean": 12.0}, "episodes": 2, "wins": 0}
            self.assertIsNotNone(mark_best_model(model, run_directory, evaluation))
            self.assertEqual(read_best_model_evaluation(run_directory)["step"], model.num_timesteps)

            evaluation_environment = build_evaluation_environment(config, record_trace=True)
            records = collect_episodes(model, evaluation_environment, config, 2, seed=1)
            evaluation_environment.close()
            stats_env.close()
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record.steps > 0 for record in records))
            self.assertTrue(all(record.reason for record in records))
            self.assertTrue(all(len(record.action_counts) == 7 for record in records))

    def test_collect_episodes_gives_up_instead_of_looping_forever(self) -> None:
        from smb1_ppo.evaluate import collect_episodes

        class _NeverEnds:
            num_timesteps = 0

            def __init__(self):
                self.observation = np.zeros((1, 4, 84, 84), dtype=np.float32)

            def predict(self, observations, deterministic=True):
                return np.zeros(len(observations), dtype=np.int64), None

        class _StubEnvironment:
            def reset(self):
                return np.zeros((1, 4, 84, 84), dtype=np.float32)

            def step(self, actions):
                return (
                    np.zeros((1, 4, 84, 84), dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                    np.zeros(1, dtype=bool),
                    [{}],
                )

            def stream_records(self):
                return []

            def close(self):
                pass

        config = EnvConfig(environment=ENVIRONMENT_SYNTHETIC, max_episode_steps=1)
        with self.assertRaises(RuntimeError):
            collect_episodes(_NeverEnds(), _StubEnvironment(), config, 3, seed=1)


if __name__ == "__main__":
    unittest.main()
