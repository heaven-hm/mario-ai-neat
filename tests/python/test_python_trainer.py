from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.agent import AgentConfig, RainbowAgent
from mario_ai_fceux.model import RainbowNetwork
from mario_ai_fceux.protocol import atomic_write_json, read_json
from mario_ai_fceux.replay import PrioritizedReplayBuffer, Transition


class PythonTrainerTests(unittest.TestCase):
    def test_protocol_round_trip_is_atomic_and_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "message.json"
            atomic_write_json(path, {"sequence": 4, "action": 2})
            self.assertEqual(read_json(path), {"sequence": 4, "action": 2})

    def test_global_prioritized_replay_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay = PrioritizedReplayBuffer(observation_size=4, capacity=16, seed=4)
            for index in range(6):
                replay.add(Transition(np.full(4, index), index % 2, float(index), np.full(4, index + 1),
                                      index == 5, 0.99, priority=index + 1))
            identifiers, transitions, weights = replay.sample(4, beta=0.4)
            self.assertEqual(len(replay), 6)
            self.assertEqual(len(identifiers), 4)
            self.assertEqual(transitions[0].state.shape, (4,))
            self.assertEqual(weights.shape, (4,))
            replay.update_priorities(identifiers, np.full(4, 3.0))
            snapshot = Path(directory) / "replay.npz"
            replay.save(snapshot)
            resumed = PrioritizedReplayBuffer.load(snapshot)
            self.assertEqual(len(resumed), 6)
            self.assertGreater(resumed.tree.total, 0)

    def test_agent_trains_and_checkpoint_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay = PrioritizedReplayBuffer(observation_size=4, capacity=32, seed=9)
            config = AgentConfig(observation_size=4, action_count=2, batch_size=4, learning_starts=4,
                                 n_step=2, target_sync_steps=1, atom_count=11, value_min=-5, value_max=5, seed=9)
            agent = RainbowAgent(replay, config=config, device="cpu")
            for index in range(12):
                state = np.asarray([index, 0, 0, 1], dtype=np.float32)
                next_state = state + 0.1
                agent.observe("worker-a", state, index % 2, 0.1, next_state, index == 11)
            self.assertGreaterEqual(len(replay), 4)
            self.assertIsNotNone(agent.learn())
            checkpoint = Path(directory) / "model.pt"
            replay_snapshot = Path(directory) / "replay.npz"
            agent.save(checkpoint, replay_snapshot)
            resumed_replay = PrioritizedReplayBuffer.load(replay_snapshot)
            resumed = RainbowAgent(resumed_replay, config=config, device="cpu")
            resumed.load(checkpoint)
            self.assertEqual(resumed.steps, agent.steps)
            self.assertEqual(resumed.select_actions(np.zeros((1, 4), dtype=np.float32), explore=False).shape, (1,))
            self.assertEqual(len(resumed_replay), len(replay))

    def test_c51_distribution_and_noisynet_evaluation_are_well_formed(self) -> None:
        network = RainbowNetwork(observation_size=4, action_count=2, atom_count=11)
        states = __import__("torch").zeros((3, 4))
        network.eval()
        first = network.distribution(states)
        second = network.distribution(states)
        self.assertTrue(__import__("torch").allclose(first, second))
        self.assertTrue(__import__("torch").allclose(first.sum(dim=-1), __import__("torch").ones((3, 2))))
        network.train()
        network.reset_noise()
        noisy_first = network.distribution(states)
        network.reset_noise()
        noisy_second = network.distribution(states)
        self.assertFalse(__import__("torch").allclose(noisy_first, noisy_second))


if __name__ == "__main__":
    unittest.main()
