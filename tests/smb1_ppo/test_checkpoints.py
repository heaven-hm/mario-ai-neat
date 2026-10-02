"""Checkpoint metadata, resume resolution, and the protected-directory guard."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smb1_ppo import checkpoints


class FakeModel:
    """The two things the checkpoint helpers need from an SB3 model."""

    def __init__(self, payload: bytes = b"weights", steps: int = 0):
        self.payload = payload
        self.num_timesteps = steps

    def save(self, path) -> None:
        Path(path).write_bytes(self.payload)


class RunDirectoryGuardTests(unittest.TestCase):
    def test_the_rainbow_experiment_directories_are_protected(self) -> None:
        for protected in checkpoints.PROTECTED_RUN_DIRECTORIES:
            with self.assertRaises(checkpoints.RunDirectoryError):
                checkpoints.guard_run_directory(protected)
            with self.assertRaises(checkpoints.RunDirectoryError):
                checkpoints.guard_run_directory(Path(protected) / "checkpoints")

    def test_the_ppo_run_directory_is_allowed(self) -> None:
        self.assertTrue(str(checkpoints.guard_run_directory("runs/smb1-ppo-w1-1")).endswith("smb1-ppo-w1-1"))
        self.assertEqual(checkpoints.DEFAULT_RUN_DIRECTORY, Path("runs/smb1-ppo-w1-1"))

    def test_a_run_directory_outside_the_repository_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(checkpoints.guard_run_directory(directory).is_dir())


class CheckpointTests(unittest.TestCase):
    def _run_directory(self, directory: str) -> Path:
        run = Path(directory) / "runs" / "smb1-ppo-w1-1"
        run.mkdir(parents=True)
        return run

    def test_checkpoint_names_are_zero_padded_and_sortable(self) -> None:
        self.assertEqual(checkpoints.checkpoint_filename(100), "checkpoint_000000000100_steps.zip")
        self.assertLess(checkpoints.checkpoint_filename(99), checkpoints.checkpoint_filename(100))

    def test_saving_writes_the_model_and_its_resume_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_directory(directory)
            model = FakeModel(payload=b"policy-bytes", steps=1024)
            path = checkpoints.save_checkpoint(model, run, 1024, {"n_steps": 256})
            self.assertTrue(path.is_file())
            metadata = json.loads(path.with_suffix(".zip.json").read_text())
            self.assertEqual(metadata["step"], 1024)
            self.assertEqual(metadata["algorithm"], "PPO")
            self.assertEqual(metadata["config"], {"n_steps": 256})
            self.assertEqual(metadata["size_bytes"], len(b"policy-bytes"))
            self.assertEqual(len(metadata["sha256"]), 64)
            self.assertIn("created_at", metadata)

    def test_the_latest_alias_is_written_alongside_periodic_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_directory(directory)
            model = FakeModel()
            for step in (100, 200, 300):
                checkpoints.save_checkpoint(model, run, step, {})
                checkpoints.save_checkpoint(model, run, step, {}, label=checkpoints.LATEST_CHECKPOINT)
            self.assertTrue((run / "checkpoints" / checkpoints.LATEST_CHECKPOINT).is_file())
            self.assertEqual(
                [path.name for path in checkpoints.list_checkpoints(run)],
                [
                    "checkpoint_000000000300_steps.zip",
                    "checkpoint_000000000200_steps.zip",
                    "checkpoint_000000000100_steps.zip",
                ],
            )

    def test_pruning_keeps_the_newest_and_removes_their_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_directory(directory)
            model = FakeModel()
            for step in (100, 200, 300, 400):
                checkpoints.save_checkpoint(model, run, step, {})
            removed = checkpoints.prune_checkpoints(run, keep=2)
            self.assertEqual(len(removed), 2)
            self.assertEqual(len(checkpoints.list_checkpoints(run)), 2)
            for path in removed:
                self.assertFalse(path.exists())
                self.assertFalse(path.with_suffix(".zip.json").exists())

    def test_resume_prefers_latest_then_best_then_the_newest_periodic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_directory(directory)
            with self.assertRaises(FileNotFoundError):
                checkpoints.resolve_resume(run, "latest")
            model = FakeModel()
            checkpoints.save_checkpoint(model, run, 500, {})
            checkpoints.save_checkpoint(model, run, 500, {}, label=checkpoints.LATEST_CHECKPOINT)
            self.assertEqual(checkpoints.resolve_resume(run, "latest").name, checkpoints.LATEST_CHECKPOINT)
            checkpoints.mark_best_model(model, run, {"win_rate": 0.5, "x_position": {"mean": 10.0}})
            self.assertEqual(checkpoints.resolve_resume(run, "best").name, checkpoints.BEST_MODEL)

    def test_resume_accepts_an_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "somewhere.zip"
            explicit.write_bytes(b"x")
            self.assertEqual(checkpoints.resolve_resume(directory, str(explicit)), explicit)
            with self.assertRaises(FileNotFoundError):
                checkpoints.resolve_resume(directory, str(Path(directory) / "absent.zip"))

    def test_training_state_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_directory(directory)
            self.assertEqual(checkpoints.read_training_state(run), {})
            checkpoints.write_training_state(run, {"step": 4096, "checkpoint": "latest.zip"})
            self.assertEqual(checkpoints.read_training_state(run)["step"], 4096)


class BestModelTests(unittest.TestCase):
    def _run_directory(self, directory: str) -> Path:
        run = Path(directory) / "runs" / "smb1-ppo-w1-1"
        run.mkdir(parents=True)
        return run

    def test_a_first_measured_evaluation_becomes_the_best_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_directory(directory)
            model = FakeModel(steps=1024)
            evaluation = {"win_rate": 0.0, "x_position": {"mean": 250.0}, "episodes": 5}
            path = checkpoints.mark_best_model(model, run, evaluation, {"device": "cpu"})
            self.assertIsNotNone(path)
            payload = checkpoints.read_best_model_evaluation(run)
            self.assertEqual(payload["step"], 1024)
            self.assertEqual(payload["selected_by"], "deterministic greedy evaluation only")
            self.assertEqual(payload["previous"], None)

    def test_a_worse_or_equal_evaluation_never_replaces_the_best_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_directory(directory)
            model = FakeModel(payload=b"first")
            checkpoints.mark_best_model(model, run, {"win_rate": 0.0, "x_position": {"mean": 250.0}})
            original = (run / checkpoints.BEST_MODEL).read_bytes()

            model.payload = b"second"
            self.assertIsNone(
                checkpoints.mark_best_model(model, run, {"win_rate": 0.0, "x_position": {"mean": 250.0}})
            )
            self.assertIsNone(
                checkpoints.mark_best_model(model, run, {"win_rate": 0.0, "x_position": {"mean": 100.0}})
            )
            self.assertEqual((run / checkpoints.BEST_MODEL).read_bytes(), original)

    def test_more_wins_beat_a_further_x_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_directory(directory)
            model = FakeModel(payload=b"measured")
            checkpoints.mark_best_model(model, run, {"win_rate": 0.0, "x_position": {"mean": 3200.0}})
            model.payload = b"winner"
            path = checkpoints.mark_best_model(model, run, {"win_rate": 0.2, "x_position": {"mean": 100.0}})
            self.assertIsNotNone(path)
            self.assertEqual((run / checkpoints.BEST_MODEL).read_bytes(), b"winner")
            self.assertEqual(checkpoints.read_best_model_evaluation(run)["win_rate"], 0.2)

    def test_ranking_prefers_win_rate_then_position(self) -> None:
        self.assertEqual(checkpoints.evaluation_rank(None), (-1.0, -1.0))
        self.assertLess(
            checkpoints.evaluation_rank({"win_rate": 0.0, "x_position": {"mean": 999.0}}),
            checkpoints.evaluation_rank({"win_rate": 1.0, "x_position": {"mean": 1.0}}),
        )
        self.assertLess(
            checkpoints.evaluation_rank({"win_rate": 1.0, "x_position": {"mean": 1.0}}),
            checkpoints.evaluation_rank({"win_rate": 1.0, "x_position": {"mean": 2.0}}),
        )


if __name__ == "__main__":
    unittest.main()
