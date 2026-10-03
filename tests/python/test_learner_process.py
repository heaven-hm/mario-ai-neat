from __future__ import annotations

import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.agent import AgentConfig
from mario_ai_fceux.learner import learner_main


class LearnerProcessTests(unittest.TestCase):
    def test_collector_and_learner_are_separate_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            inbox, outbox = context.Queue(), context.Queue()
            config = AgentConfig(observation_size=4, action_count=2, atom_count=11,
                                 batch_size=2, learning_starts=2, seed=3)
            process = context.Process(target=learner_main,
                                      args=(inbox, outbox, directory, vars(config), False, "cpu"))
            process.start()
            try:
                states = np.zeros((2, 4), dtype=np.float32)
                inbox.put(("act", 1, states, False))
                kind, request_id, actions, values, hidden, status = outbox.get(timeout=20)
                self.assertEqual((kind, request_id), ("act", 1))
                self.assertEqual(actions.shape, (2,))
                self.assertEqual(values.shape, (2, 2))
                self.assertEqual(hidden.shape, (2, 16))
                self.assertEqual(status["steps"], 0)
                inbox.put(("transition", "worker", states[0], 0, 1.0, states[0], False))
                inbox.put(("transition", "worker", states[1], 1, -1.0, states[1], True))
                inbox.put(("save",))
                inbox.put(("stop",))
                process.join(timeout=30)
                self.assertFalse(process.is_alive())
                self.assertTrue((Path(directory) / "model.pt").exists())
                self.assertTrue((Path(directory) / "replay.npz").exists())
            finally:
                if process.is_alive():
                    process.kill()


if __name__ == "__main__":
    unittest.main()
