from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.agent import AgentConfig, RainbowLiteAgent
from mario_ai_fceux.protocol import atomic_write_json, read_json
from mario_ai_fceux.replay import ReplayDatabase, Transition


class PythonTrainerTests(unittest.TestCase):
    def test_protocol_round_trip_is_atomic_and_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "message.json"
            atomic_write_json(path, {"sequence": 4, "action": 2})
            self.assertEqual(read_json(path), {"sequence": 4, "action": 2})

    def test_replay_round_trip_and_priority_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay = ReplayDatabase(Path(directory) / "replay.sqlite3", observation_size=4, capacity=10)
            for index in range(6):
                replay.add(Transition(np.full(4, index), index % 2, float(index), np.full(4, index + 1),
                                      index == 5, 0.99, priority=index + 1))
            identifiers, transitions, weights = replay.sample(4)
            self.assertEqual(len(replay), 6)
            self.assertEqual(len(identifiers), 4)
            self.assertEqual(transitions[0].state.shape, (4,))
            self.assertEqual(weights.shape, (4,))
            replay.update_priorities(identifiers, np.full(4, 3.0))
            replay.close()

    def test_agent_trains_and_checkpoint_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay = ReplayDatabase(Path(directory) / "replay.sqlite3", observation_size=4)
            config = AgentConfig(observation_size=4, action_count=2, batch_size=4, learning_starts=4,
                                 n_step=2, target_sync_steps=1)
            agent = RainbowLiteAgent(replay, config=config, device="cpu")
            for index in range(8):
                state = np.asarray([index, 0, 0, 1], dtype=np.float32)
                next_state = state + 0.1
                agent.observe("worker-a", state, index % 2, 0.1, next_state, index == 7)
            self.assertGreaterEqual(len(replay), 4)
            self.assertIsNotNone(agent.learn())
            checkpoint = Path(directory) / "model.pt"
            agent.save(checkpoint)
            resumed = RainbowLiteAgent(replay, config=config, device="cpu")
            resumed.load(checkpoint)
            self.assertEqual(resumed.steps, agent.steps)
            self.assertEqual(resumed.select_actions(np.zeros((1, 4), dtype=np.float32), explore=False).shape, (1,))
            replay.close()


if __name__ == "__main__":
    unittest.main()
