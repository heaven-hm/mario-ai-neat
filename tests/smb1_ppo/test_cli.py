"""End-to-end checks of the documented commands.

These run the real command lines as subprocesses, which is the only way to
cover the pieces that only exist across a process boundary: the subprocess
vectorized environments, TensorBoard logging, checkpoint writing, and the
evaluation CLI. They use the ROM-free synthetic stand-in, so they run anywhere.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAIN_STEPS = 512


class CommandLineTests(unittest.TestCase):
    def run_module(self, *arguments: str) -> subprocess.CompletedProcess:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        return subprocess.run(
            [sys.executable, "-m", *arguments],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
        )

    def test_train_then_evaluate_writes_the_documented_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "smb1-ppo-w1-1"
            training = self.run_module(
                "smb1_ppo.train",
                "--env",
                "synthetic",
                "--workers",
                "2",
                "--total-timesteps",
                str(TRAIN_STEPS),
                "--n-steps",
                "64",
                "--batch-size",
                "64",
                "--n-epochs",
                "1",
                "--max-episode-steps",
                "200",
                "--checkpoint-freq",
                "256",
                "--keep-checkpoints",
                "2",
                "--eval-freq",
                "256",
                "--eval-episodes",
                "1",
                "--run-dir",
                str(run_directory),
                "--no-progress-bar",
            )
            self.assertEqual(training.returncode, 0, training.stderr[-4000:])
            self.assertIn("training PPO on SMB1 1-1 (synthetic)", training.stdout)

            metadata = json.loads((run_directory / "run.json").read_text())
            self.assertEqual(metadata["environment"]["kind"], "synthetic")
            self.assertEqual(metadata["environment"]["level"], "1-1")
            self.assertEqual(metadata["training"]["workers"], 2)
            self.assertEqual(metadata["total_timesteps"], TRAIN_STEPS)
            self.assertIn("stable_baselines3", metadata["libraries"])
            self.assertIsNone(metadata["environment"]["rom"])

            checkpoints = run_directory / "checkpoints"
            self.assertTrue((checkpoints / "latest.zip").is_file())
            self.assertTrue((checkpoints / "latest.zip.json").is_file())
            periodic = sorted(checkpoints.glob("checkpoint_*_steps.zip"))
            self.assertEqual(len(periodic), 2)
            self.assertTrue(periodic[0].with_suffix(".zip.json").is_file())
            state = json.loads((run_directory / "training_state.json").read_text())
            self.assertGreaterEqual(state["step"], 1)

            tensorboard = list((run_directory / "tensorboard").rglob("events.out.tfevents*"))
            self.assertTrue(tensorboard, "no TensorBoard event file was written")

            self.assertTrue((run_directory / "best_model.zip").is_file())
            best = json.loads((run_directory / "best_model_eval.json").read_text())
            self.assertEqual(best["selected_by"], "deterministic greedy evaluation only")
            self.assertTrue(best["deterministic"])
            self.assertIn("episodes", best)

            evaluations = sorted((run_directory / "evaluations").glob("step-*.json"))
            self.assertTrue(evaluations, "no during-training evaluation was recorded")

            output_directory = Path(directory) / "evaluation"
            evaluation = self.run_module(
                "smb1_ppo.evaluate",
                "--model",
                str(run_directory / "best_model.zip"),
                "--env",
                "synthetic",
                "--episodes",
                "3",
                "--max-episode-steps",
                "200",
                "--out",
                str(output_directory),
            )
            self.assertEqual(evaluation.returncode, 0, evaluation.stderr[-4000:])
            results = json.loads((output_directory / "results.json").read_text())
            self.assertEqual(results["episodes"], 3)
            self.assertEqual(results["environment"], "synthetic")
            self.assertFalse(results["promotion"]["promoted"])
            self.assertIn("synthetic", results["promotion"]["blocked_by"])
            self.assertTrue(results["deterministic"])
            self.assertEqual(results["environment_config"]["rom"], None)

            with (output_directory / "episodes.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            for field in ("won", "reason", "max_x", "duration_seconds", "action_counts"):
                self.assertIn(field, rows[0])
            self.assertTrue((output_directory / "action_trace.csv").is_file())
            traces = [
                json.loads(line)
                for line in (output_directory / "action_trace.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(traces), 3)
            self.assertEqual(len(traces[0]["actions"]), int(rows[0]["steps"]))

    def test_evaluate_refuses_a_pipeline_that_does_not_match_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "smb1-ppo-w1-1"
            self.run_module(
                "smb1_ppo.train",
                "--env",
                "synthetic",
                "--workers",
                "1",
                "--total-timesteps",
                "64",
                "--n-steps",
                "64",
                "--batch-size",
                "64",
                "--n-epochs",
                "1",
                "--max-episode-steps",
                "50",
                "--no-eval",
                "--run-dir",
                str(run_directory),
                "--no-progress-bar",
            )
            evaluation = self.run_module(
                "smb1_ppo.evaluate",
                "--model",
                str(run_directory / "checkpoints" / "latest.zip"),
                "--env",
                "synthetic",
                "--episodes",
                "1",
                "--frame-skip",
                "8",
            )
            self.assertNotEqual(evaluation.returncode, 0)
            self.assertIn("observation pipeline mismatch", evaluation.stderr)

    def test_the_rom_import_command_reports_a_usable_image(self) -> None:
        results = self.run_module("smb1_ppo.rom", "check", "--path", "/nonexistent/rom.nes")
        self.assertEqual(results.returncode, 2)
        self.assertIn("no Super Mario Bros. ROM", results.stderr)
        self.assertIn("smb1_ppo.rom import", results.stderr)

    def test_the_trainer_refuses_a_protected_rainbow_run_directory(self) -> None:
        training = self.run_module(
            "smb1_ppo.train",
            "--env",
            "synthetic",
            "--total-timesteps",
            "64",
            "--run-dir",
            "runs/full-rainbow-input-fixed",
            "--no-progress-bar",
        )
        self.assertNotEqual(training.returncode, 0)
        self.assertIn("must not write there", training.stderr)


if __name__ == "__main__":
    unittest.main()
