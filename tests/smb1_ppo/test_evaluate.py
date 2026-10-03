"""Evaluation report metrics, promotion verdict, and written evidence."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from smb1_ppo.env import ENVIRONMENT_SMB1, ENVIRONMENT_SYNTHETIC, EnvConfig, LevelSpec
from smb1_ppo.episodes import (
    REASON_DEATH,
    REASON_FLAG,
    REASON_TIMEOUT,
    REPORT_FIELDS,
    EpisodeRecord,
    EpisodeTracker,
    promotion_verdict,
    summarize,
)
from smb1_ppo.evaluate import (
    build_report,
    check_environment_match,
    distinct_traces,
    resolve_run_directory,
    write_action_traces,
    write_episode_csv,
)
from smb1_ppo.rewards import EVENT_DEATH, EVENT_FLAG, EVENT_TIMEOUT
from support import smb1_info


def record(
    index: int,
    *,
    won: bool = False,
    max_x: int = 100,
    reason: str = REASON_DEATH,
    cause: str = "hazard",
    steps: int = 50,
    duration: float = 1.0,
    counts: list[int] | None = None,
    trace: list[dict] | None = None,
) -> EpisodeRecord:
    payload = EpisodeRecord(episode=index, world=1, stage=1)
    payload.won = won
    payload.reason = reason
    payload.death_cause = cause
    payload.max_x = max_x
    payload.terminal_x = max_x
    payload.steps = steps
    payload.duration_seconds = duration
    payload.action_counts = counts or [1] * 7
    payload.trace = trace or []
    return payload


class TrackerTests(unittest.TestCase):
    def test_a_flag_episode_is_recorded_as_a_win(self) -> None:
        tracker = EpisodeTracker(1, 1)
        tracker.reset()
        tracker.observe(smb1_info(x_pos=200), False, False, action=1, step_reward=1.0)
        finished = tracker.observe(
            smb1_info(x_pos=3150, flag_get=True, mario_event=EVENT_FLAG),
            True,
            False,
            action=2,
            step_reward=50.0,
        )
        self.assertIsNotNone(finished)
        self.assertTrue(finished.won)
        self.assertEqual(finished.reason, REASON_FLAG)
        self.assertEqual(finished.max_x, 3150)
        self.assertEqual(finished.steps, 2)
        self.assertEqual(finished.action_counts[1], 1)
        self.assertEqual(finished.action_counts[2], 1)
        self.assertEqual(len(finished.trace), 2)
        self.assertEqual(finished.trace[0]["action_name"], "right")
        self.assertEqual(tracker.episodes, [finished])

    def test_a_death_and_a_time_out_are_told_apart(self) -> None:
        tracker = EpisodeTracker(1, 1, index=1)
        tracker.reset()
        death = tracker.observe(
            smb1_info(x_pos=500, mario_event=EVENT_DEATH, death_cause="pit"),
            True,
            False,
            action=1,
            step_reward=-15.0,
        )
        self.assertEqual(death.reason, REASON_DEATH)
        self.assertEqual(death.death_cause, "pit")
        self.assertFalse(death.won)

        tracker.observe(
            smb1_info(time=0, mario_event=EVENT_TIMEOUT), True, False, action=0, step_reward=-15.0
        )
        self.assertEqual(tracker.episodes[-1].reason, REASON_TIMEOUT)
        self.assertEqual(tracker.episodes[-1].episode, 2)

    def test_a_truncated_episode_is_not_a_death(self) -> None:
        tracker = EpisodeTracker(1, 1)
        tracker.reset()
        finished = tracker.observe(smb1_info(mario_event="truncated"), False, True, action=0, step_reward=0.0)
        self.assertEqual(finished.reason, "truncated")
        self.assertEqual(finished.deaths, 0)

    def test_frame_skip_and_power_ups_are_counted(self) -> None:
        tracker = EpisodeTracker(1, 1, frame_skip=4)
        tracker.reset()
        tracker.observe(
            smb1_info(status="tall", reward_terms={"mushroom": 8.0}),
            False,
            False,
            action=2,
            step_reward=8.0,
        )
        tracker.observe(
            smb1_info(status="fireball", reward_terms={"fire_flower": 15.0}),
            False,
            False,
            action=3,
            step_reward=15.0,
        )
        finished = tracker.observe(
            smb1_info(status="small", reward_terms={"power_loss": -8.0}, mario_event=EVENT_DEATH),
            True,
            False,
            action=1,
            step_reward=-15.0,
        )
        self.assertEqual(finished.mushrooms, 1)
        self.assertEqual(finished.fire_flowers, 1)
        self.assertEqual(finished.power_losses, 1)
        self.assertEqual(finished.frames, 12)
        self.assertEqual(finished.deaths, 1)
        self.assertGreaterEqual(finished.duration_seconds, 0.0)

    def test_trace_can_be_switched_off_for_training(self) -> None:
        tracker = EpisodeTracker(1, 1, record_trace=False)
        tracker.reset()
        self.assertIsNone(tracker.observe(smb1_info(), False, False, action=1, step_reward=0.0))
        finished = tracker.observe(smb1_info(), True, False, action=2, step_reward=0.0)
        self.assertEqual(finished.trace, [])
        self.assertEqual(finished.steps, 2)


class SummaryTests(unittest.TestCase):
    def test_summary_counts_wins_positions_and_reasons(self) -> None:
        records = [
            record(1, won=True, max_x=3150, reason=REASON_FLAG, steps=500, duration=30.0),
            record(2, max_x=900, steps=200, duration=12.0),
            record(3, max_x=300, reason=REASON_TIMEOUT, steps=100, duration=8.0),
        ]
        summary = summarize(records, "1-1")
        self.assertEqual((summary["episodes"], summary["wins"], summary["losses"]), (3, 1, 2))
        self.assertAlmostEqual(summary["win_rate"], 1 / 3)
        self.assertEqual(summary["x_position"]["mean"], (3150 + 900 + 300) / 3)
        self.assertEqual(summary["x_position"]["median"], 900)
        self.assertEqual(summary["x_position"]["max"], 3150)
        self.assertEqual(summary["death_reasons"], {REASON_FLAG: 1, REASON_DEATH: 1, REASON_TIMEOUT: 1})
        self.assertEqual(summary["death_causes"], {"hazard": 1})
        self.assertEqual(summary["duration_seconds"]["mean"], (30 + 12 + 8) / 3)
        self.assertEqual(summary["steps"]["max"], 500)
        self.assertEqual(summary["action_counts"]["NOOP"], 3)

    def test_an_empty_summary_does_not_divide_by_zero(self) -> None:
        summary = summarize([], "1-1")
        self.assertEqual(summary["episodes"], 0)
        self.assertEqual(summary["win_rate"], 0.0)
        self.assertEqual(summary["x_position"]["max"], 0)

    def test_power_ups_are_summed_across_episodes(self) -> None:
        first = record(1)
        first.mushrooms = 2
        first.fire_flowers = 1
        first.power_losses = 1
        second = record(2, won=True, reason=REASON_FLAG)
        second.mushrooms = 1
        summary = summarize([first, second], "1-1")
        self.assertEqual(summary["power_ups"], {"mushrooms": 3, "fire_flowers": 1, "power_losses": 1})


class PromotionTests(unittest.TestCase):
    def test_twenty_greedy_wins_promote(self) -> None:
        verdict = promotion_verdict(20, 1.0)
        self.assertTrue(verdict["promoted"])
        self.assertEqual(
            verdict["criterion"], "20 deterministic no-cheat World 1-1 episodes with a 100% win rate"
        )

    def test_nineteen_wins_out_of_twenty_does_not_promote(self) -> None:
        self.assertFalse(promotion_verdict(20, 0.95)["promoted"])

    def test_five_wins_out_of_five_does_not_promote_without_twenty_episodes(self) -> None:
        self.assertFalse(promotion_verdict(5, 1.0)["promoted"])

    def test_a_synthetic_or_sampled_run_can_never_promote(self) -> None:
        verdict = promotion_verdict(20, 1.0, promotable=False)
        self.assertFalse(verdict["promoted"])
        self.assertIn("not a real World 1-1", verdict["verdict"])

    def test_summary_carries_the_promotion_verdict(self) -> None:
        records = [record(index, won=True, reason=REASON_FLAG) for index in range(1, 21)]
        summary = summarize(records, "1-1")
        self.assertTrue(summary["promotion"]["promoted"])
        self.assertEqual(summary["promotion"]["measured_episodes"], 20)


class EvidenceTests(unittest.TestCase):
    def test_an_episode_row_holds_every_reported_field(self) -> None:
        row = record(1).row()
        for field in REPORT_FIELDS:
            self.assertIn(field, row)
        self.assertIn("action_right+A", row)
        self.assertEqual(row["action_right+A"], 1)
        self.assertEqual(row["action_counts"], "1 1 1 1 1 1 1")

    def test_episode_csv_and_action_traces_round_trip(self) -> None:
        trace = [
            {
                "step": 1,
                "action": 1,
                "action_name": "right",
                "x_pos": 40,
                "y_pos": 79,
                "time": 400,
                "status": "small",
                "score": 0,
                "raw_reward": 1.0,
                "shaped_reward": 0.01,
                "event": "",
            },
            {
                "step": 2,
                "action": 2,
                "action_name": "right+A",
                "x_pos": 90,
                "y_pos": 79,
                "time": 400,
                "status": "small",
                "score": 0,
                "raw_reward": 1.0,
                "shaped_reward": 0.5,
                "event": EVENT_FLAG,
            },
        ]
        records = [record(1, won=True, reason=REASON_FLAG, trace=trace)]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_episode_csv(directory / "episodes.csv", records)
            write_action_traces(directory, records)
            with (directory / "episodes.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["won"], "True")
            self.assertEqual(rows[0]["reason"], REASON_FLAG)
            with (directory / "action_trace.csv").open() as handle:
                trace_rows = list(csv.DictReader(handle))
            self.assertEqual(len(trace_rows), 2)
            self.assertEqual(trace_rows[1]["action_name"], "right+A")
            self.assertEqual(trace_rows[0]["episode"], "1")
            payload = json.loads((directory / "action_trace.jsonl").read_text().strip())
            self.assertEqual(payload["actions"], [1, 2])
            self.assertTrue(payload["won"])

    def test_distinct_traces_separates_repeats_from_variation(self) -> None:
        one = [record(1, trace=[{"action": 1}, {"action": 2}])]
        same = [
            record(1, trace=[{"action": 1}, {"action": 2}]),
            record(2, trace=[{"action": 1}, {"action": 2}]),
        ]
        different = [
            record(1, trace=[{"action": 1}, {"action": 2}]),
            record(2, trace=[{"action": 1}, {"action": 3}]),
        ]
        self.assertEqual(distinct_traces(one), 1)
        self.assertEqual(distinct_traces(same), 1)
        self.assertEqual(distinct_traces(different), 2)


class EnvironmentMatchTests(unittest.TestCase):
    def test_a_different_observation_pipeline_is_refused(self) -> None:
        config = EnvConfig(environment=ENVIRONMENT_SMB1, frame_skip=8)
        metadata = {
            "run_directory": "/tmp/run",
            "environment": {"frame_skip": 4, "frame_size": 84, "frame_stack": 4},
        }
        with self.assertRaises(SystemExit) as context:
            check_environment_match(config, metadata)
        self.assertIn("--frame-skip", str(context.exception))

    def test_a_matching_pipeline_passes_and_reports_reward_drift(self) -> None:
        config = EnvConfig(environment=ENVIRONMENT_SMB1, frame_skip=4)
        metadata = {
            "environment": {
                "frame_skip": 4,
                "frame_size": 84,
                "frame_stack": 4,
                "reward": {"flag_reward": 99.0},
            }
        }
        result = check_environment_match(config, metadata)
        self.assertTrue(result["checked"])
        self.assertEqual(result["reward_drift"]["flag_reward"]["trained"], 99.0)

    def test_missing_metadata_is_not_a_mismatch(self) -> None:
        result = check_environment_match(EnvConfig(environment=ENVIRONMENT_SMB1), {})
        self.assertEqual(result["reward_drift"], {})


class RunDirectoryResolutionTests(unittest.TestCase):
    def test_the_directory_owning_the_model_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "smb1-ppo-w1-1"
            (run / "checkpoints").mkdir(parents=True)
            (run / "run.json").write_text("{}")
            model = run / "checkpoints" / "latest.zip"
            model.write_bytes(b"x")
            self.assertEqual(resolve_run_directory(model, None), run.resolve())
            self.assertEqual(resolve_run_directory(run / "best_model.zip", None), run.resolve())

    def test_an_explicit_run_directory_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory) / "other"
            other.mkdir()
            self.assertEqual(resolve_run_directory(Path("model.zip"), other), other.resolve())


class ReportTests(unittest.TestCase):
    def test_a_synthetic_report_records_why_it_cannot_promote(self) -> None:
        records = [record(index, won=True, reason=REASON_FLAG) for index in range(1, 21)]
        report = build_report(
            records,
            EnvConfig(environment=ENVIRONMENT_SYNTHETIC, level=LevelSpec(1, 1)),
            Path("model.zip"),
            {},
            deterministic=True,
            match={"checked": True, "reward_drift": {}},
            environment_config={"kind": ENVIRONMENT_SYNTHETIC, "level": "1-1", "rom": None},
        )
        self.assertEqual(report["environment"], ENVIRONMENT_SYNTHETIC)
        self.assertFalse(report["promotion"]["promoted"])
        self.assertIn("synthetic", report["promotion"]["blocked_by"])
        self.assertEqual(report["cheats"], "none: stock environment, stock ROM mode 'vanilla'")

    def test_a_sampled_run_records_why_it_cannot_promote(self) -> None:
        records = [record(index, won=True, reason=REASON_FLAG) for index in range(1, 21)]
        report = build_report(
            records,
            EnvConfig(environment=ENVIRONMENT_SMB1),
            Path("model.zip"),
            {},
            deterministic=False,
            match={"checked": True, "reward_drift": {}},
            environment_config={"kind": ENVIRONMENT_SMB1, "level": "1-1", "rom": {"verdict": "known-good"}},
        )
        self.assertFalse(report["promotion"]["promoted"])
        self.assertIn("sampled", report["promotion"]["blocked_by"])

    def test_a_real_deterministic_sweep_records_distinct_traces(self) -> None:
        trace = [{"action": 1}, {"action": 2}]
        records = [
            record(1, won=True, reason=REASON_FLAG, trace=trace),
            record(2, won=True, reason=REASON_FLAG, trace=trace),
        ]
        report = build_report(
            records,
            EnvConfig(environment=ENVIRONMENT_SMB1),
            Path("model.zip"),
            {},
            deterministic=True,
            match={"checked": True, "reward_drift": {}},
            environment_config={"kind": ENVIRONMENT_SMB1, "level": "1-1", "rom": {"verdict": "known-good"}},
        )
        self.assertEqual(report["distinct_action_traces"], 1)
        self.assertTrue(report["promotion"]["promoted"] is False)  # only two episodes were run
        self.assertEqual(report["deaths"], 0)


if __name__ == "__main__":
    unittest.main()
