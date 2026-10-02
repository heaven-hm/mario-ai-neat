"""Reward transitions: mushroom, fire flower, power loss, death, flag.

Each test scripts the exact info sequence the real environment reports, so the
reward wrapper is checked against events rather than against the emulator.
"""

from __future__ import annotations

import unittest

from smb1_ppo.rewards import (
    CAUSE_HAZARD,
    CAUSE_PIT,
    CAUSE_UNCLASSIFIED,
    EVENT_DEATH,
    EVENT_FLAG,
    EVENT_TIMEOUT,
    EVENT_TRUNCATED,
    MarioRewardWrapper,
    RewardConfig,
    classify_death_cause,
    detect_signals,
)
from support import ScriptedEnv, smb1_info


def wrap(reset_info: dict, steps: list[tuple], viewport: int | None = 1, **config):
    env = MarioRewardWrapper(ScriptedEnv(reset_info, steps, viewport=viewport), RewardConfig(**config))
    env.reset()
    return env


def step(env, action: int = 1):
    return env.step(action)


class MushroomTests(unittest.TestCase):
    def test_a_rise_from_small_to_tall_is_one_mushroom(self) -> None:
        env = wrap(
            smb1_info(),
            [
                (smb1_info(x_pos=100, status="tall", score=1000), False, False, 0.0),
                (smb1_info(x_pos=120, status="tall", score=1000), False, False, 0.0),
            ],
        )
        _, reward, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["mushroom"], 8.0)
        self.assertEqual(env.power_ups, (1, 0, 0))
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["mushroom"], 0.0)
        self.assertEqual(env.power_ups, (1, 0, 0))
        self.assertGreater(reward, 0.0)


class FireFlowerTests(unittest.TestCase):
    def test_a_rise_to_fireball_is_one_fire_flower_and_beats_a_mushroom(self) -> None:
        env = wrap(
            smb1_info(status="tall"),
            [(smb1_info(x_pos=100, status="fireball", score=2000), False, False, 0.0)],
        )
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["fire_flower"], 15.0)
        self.assertEqual(info["reward_terms"]["mushroom"], 0.0)
        self.assertGreater(RewardConfig().fire_flower_reward, RewardConfig().mushroom_reward)
        self.assertEqual(env.power_ups, (0, 1, 0))

    def test_small_to_fireball_counts_as_a_fire_flower(self) -> None:
        env = wrap(
            smb1_info(),
            [(smb1_info(x_pos=100, status="fireball"), False, False, 0.0)],
        )
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["fire_flower"], 15.0)


class PowerLossTests(unittest.TestCase):
    def test_fireball_to_small_is_a_penalty(self) -> None:
        env = wrap(
            smb1_info(status="fireball"),
            [(smb1_info(x_pos=100, status="small"), False, False, 0.0)],
        )
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["power_loss"], -8.0)
        self.assertEqual(env.power_ups, (0, 0, 1))

    def test_tall_to_small_is_a_penalty(self) -> None:
        env = wrap(
            smb1_info(status="tall"),
            [(smb1_info(x_pos=100, status="small"), False, False, 0.0)],
        )
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["power_loss"], -8.0)

    def test_power_loss_is_not_charged_twice_on_a_death(self) -> None:
        env = wrap(
            smb1_info(status="fireball"),
            [(smb1_info(x_pos=100, status="small"), True, False, -25.0)],
        )
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["power_loss"], 0.0)
        self.assertEqual(info["reward_terms"]["death"], -15.0)


class DeathTests(unittest.TestCase):
    def test_a_terminal_step_without_a_flag_is_a_death(self) -> None:
        env = wrap(smb1_info(), [(smb1_info(x_pos=120), True, False, -25.0)])
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["death"], -15.0)
        self.assertEqual(info["mario_event"], EVENT_DEATH)
        self.assertEqual(info["death_cause"], CAUSE_HAZARD)

    def test_a_pit_death_is_named_from_the_viewport_byte(self) -> None:
        env = wrap(smb1_info(), [(smb1_info(x_pos=120), True, False, -25.0)], viewport=3)
        _, _, _, _, info = step(env)
        self.assertEqual(info["death_cause"], CAUSE_PIT)

    def test_losing_a_life_without_terminating_still_costs_a_death(self) -> None:
        env = wrap(smb1_info(), [(smb1_info(x_pos=120, life=1), False, False, 0.0)])
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["death"], -15.0)

    def test_an_in_game_time_out_is_reported_as_a_time_out(self) -> None:
        env = wrap(smb1_info(), [(smb1_info(x_pos=120, time=0), True, False, -25.0)])
        _, _, _, _, info = step(env)
        self.assertEqual(info["mario_event"], EVENT_TIMEOUT)
        self.assertEqual(info["reward_terms"]["timeout"], -15.0)
        self.assertEqual(info["reward_terms"]["death"], 0.0)

    def test_a_truncated_episode_is_not_a_death(self) -> None:
        env = wrap(smb1_info(), [(smb1_info(x_pos=120), False, True, 0.0)])
        _, _, _, _, info = step(env)
        self.assertEqual(info["mario_event"], EVENT_TRUNCATED)
        self.assertEqual(info["reward_terms"]["death"], 0.0)
        self.assertEqual(info["reward_terms"]["timeout"], 0.0)

    def test_death_outweighs_every_power_up(self) -> None:
        config = RewardConfig()
        self.assertLess(config.death_penalty, 0.0)
        self.assertGreater(abs(config.death_penalty), config.mushroom_reward)


class FlagTests(unittest.TestCase):
    def test_flag_capture_is_the_largest_single_reward(self) -> None:
        env = wrap(
            smb1_info(),
            [(smb1_info(x_pos=3150, flag_get=True), True, False, 0.0)],
        )
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["flag"], 50.0)
        self.assertEqual(info["mario_event"], EVENT_FLAG)
        self.assertEqual(info["reward_terms"]["death"], 0.0)
        config = RewardConfig()
        for other in (config.mushroom_reward, config.fire_flower_reward):
            self.assertGreater(config.flag_reward, other)


class ForwardProgressTests(unittest.TestCase):
    def test_pays_for_new_ground_only(self) -> None:
        env = wrap(
            smb1_info(),
            [
                (smb1_info(x_pos=140), False, False, 0.0),
                (smb1_info(x_pos=240), False, False, 0.0),
            ],
        )
        _, _, _, _, info = step(env)
        self.assertAlmostEqual(info["reward_terms"]["forward"], 0.25, places=6)
        _, _, _, _, info = step(env)
        self.assertAlmostEqual(info["reward_terms"]["forward"], 0.25, places=6)

    def test_walking_backwards_earns_nothing(self) -> None:
        env = wrap(
            smb1_info(),
            [(smb1_info(x_pos=200), False, False, 0.0), (smb1_info(x_pos=100), False, False, 0.0)],
        )
        step(env)
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["forward"], 0.0)

    def test_oscillating_over_old_ground_earns_nothing(self) -> None:
        env = wrap(
            smb1_info(),
            [
                (smb1_info(x_pos=300), False, False, 0.0),
                (smb1_info(x_pos=150), False, False, 0.0),
                (smb1_info(x_pos=290), False, False, 0.0),
                (smb1_info(x_pos=299), False, False, 0.0),
            ],
        )
        step(env)
        for _ in range(3):
            _, _, _, _, info = step(env)
            self.assertEqual(info["reward_terms"]["forward"], 0.0)

    def test_total_forward_shaping_is_capped_per_episode(self) -> None:
        env = wrap(
            smb1_info(),
            [
                (smb1_info(x_pos=1000), False, False, 0.0),
                (smb1_info(x_pos=2000), False, False, 0.0),
                (smb1_info(x_pos=3000), False, False, 0.0),
            ],
            forward_episode_cap=0.3,
            forward_step_cap=0.25,
        )
        first = step(env)[4]["reward_terms"]["forward"]
        second = step(env)[4]["reward_terms"]["forward"]
        third = step(env)[4]["reward_terms"]["forward"]
        self.assertAlmostEqual(first, 0.25, places=6)
        self.assertAlmostEqual(second, 0.05, places=6)
        self.assertEqual(third, 0.0)

    def test_standing_still_costs_time_and_earns_nothing(self) -> None:
        env = wrap(
            smb1_info(),
            [
                (smb1_info(x_pos=40), False, False, 0.0),
                (smb1_info(x_pos=40), False, False, 0.0),
            ],
        )
        for _ in range(2):
            _, reward, _, _, info = step(env)
            self.assertEqual(info["reward_terms"]["forward"], 0.0)
            self.assertLess(reward, 0.0)


class SignalHonestyTests(unittest.TestCase):
    def test_a_missing_status_field_disables_power_rewards(self) -> None:
        reset = smb1_info()
        reset.pop("status")
        info = smb1_info(x_pos=120)
        info.pop("status")
        env = wrap(reset, [(info, False, False, 0.0)])
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["mushroom"], 0.0)
        self.assertEqual(info["reward_terms"]["fire_flower"], 0.0)
        self.assertIn("status", env.signals.missing)
        self.assertIn("power-up", env.signals.consequences()["status"])

    def test_a_missing_x_position_disables_forward_reward(self) -> None:
        reset = smb1_info()
        reset.pop("x_pos")
        info = smb1_info()
        info.pop("x_pos")
        env = wrap(reset, [(info, False, False, 0.0)])
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["forward"], 0.0)
        self.assertIn("x_pos", env.signals.missing)

    def test_no_viewport_probe_means_an_unclassified_death(self) -> None:
        env = wrap(smb1_info(), [(smb1_info(x_pos=120), True, False, -25.0)], viewport=None)
        _, _, _, _, info = step(env)
        self.assertEqual(info["reward_terms"]["death"], -15.0)
        self.assertEqual(info["death_cause"], CAUSE_UNCLASSIFIED)
        self.assertIn("viewport", env.signals.missing)

    def test_detect_signals_reports_every_present_field(self) -> None:
        signals = detect_signals(smb1_info(), viewport=1)
        self.assertEqual(signals.missing, ())
        self.assertEqual(
            detect_signals({}, viewport=None).missing,
            ("x_pos", "status", "life", "time", "flag_get", "viewport"),
        )

    def test_classify_death_cause_only_names_what_it_can_see(self) -> None:
        self.assertEqual(classify_death_cause(None), CAUSE_UNCLASSIFIED)
        self.assertEqual(classify_death_cause(1), CAUSE_HAZARD)
        self.assertEqual(classify_death_cause(2), CAUSE_PIT)


class RawRewardPreservationTests(unittest.TestCase):
    def test_the_environment_reward_is_preserved_alongside_the_shaped_one(self) -> None:
        env = wrap(smb1_info(), [(smb1_info(x_pos=140), False, False, 7.5)])
        _, shaped, _, _, info = step(env)
        self.assertEqual(info["raw_reward"], 7.5)
        self.assertEqual(info["shaped_reward"], shaped)
        self.assertNotEqual(shaped, 7.5)
        self.assertEqual(env.raw_reward_total, 7.5)
        self.assertAlmostEqual(env.shaped_reward_total, shaped)

    def test_the_first_observation_establishes_the_baseline_without_reward(self) -> None:
        env = MarioRewardWrapper(
            ScriptedEnv({}, [(smb1_info(x_pos=140, status="tall"), False, False, 0.0)], viewport=1),
            RewardConfig(),
        )
        env.reset()
        _, reward, _, _, info = step(env)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["reward_terms"]["mushroom"], 0.0)
        self.assertEqual(info["reward_terms"]["forward"], 0.0)

    def test_shaped_reward_stays_negative_for_a_run_of_stalls(self) -> None:
        env = wrap(
            smb1_info(),
            [(smb1_info(x_pos=40), False, False, 0.0) for _ in range(5)],
        )
        total = sum(step(env)[1] for _ in range(5))
        self.assertLess(total, 0.0)


class RewardConfigTests(unittest.TestCase):
    def test_validate_rejects_a_flag_worth_less_than_a_power_up(self) -> None:
        with self.assertRaises(ValueError):
            RewardConfig(flag_reward=5.0).validate()

    def test_validate_rejects_positive_penalties_and_negative_scales(self) -> None:
        with self.assertRaises(ValueError):
            RewardConfig(death_penalty=1.0).validate()
        with self.assertRaises(ValueError):
            RewardConfig(time_penalty=0.5).validate()
        with self.assertRaises(ValueError):
            RewardConfig(forward_scale=-1.0).validate()
        RewardConfig().validate()


if __name__ == "__main__":
    unittest.main()
