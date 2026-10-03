from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            expected_next_sample = replay.sample(4, beta=0.4)[0]
            resumed = PrioritizedReplayBuffer.load(snapshot)
            self.assertEqual(len(resumed), 6)
            self.assertGreater(resumed.tree.total, 0)
            self.assertEqual(resumed.tree.minimum, replay.tree.minimum)
            np.testing.assert_array_equal(resumed.sample(4, beta=0.4)[0], expected_next_sample)

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
            evaluation_agent = RainbowAgent(
                PrioritizedReplayBuffer(observation_size=4, capacity=1, seed=9),
                config=config, device="cpu",
            )
            evaluation_agent.load(checkpoint, validate_replay=False, restore_rng=False)

    def test_rainbow_learns_preferred_action_in_a_fixed_reward_task(self) -> None:
        replay = PrioritizedReplayBuffer(observation_size=4, capacity=128, seed=13)
        config = AgentConfig(observation_size=4, action_count=2, gamma=0.0,
                             learning_rate=1e-3, batch_size=16, learning_starts=16,
                             target_sync_steps=20, n_step=1, atom_count=11,
                             value_min=-2, value_max=2, seed=13)
        agent = RainbowAgent(replay, config=config, device="cpu")
        state = np.zeros(4, dtype=np.float32)
        for _ in range(32):
            replay.add(Transition(state, 0, -1.0, state, True, 0.0))
            replay.add(Transition(state, 1, 1.0, state, True, 0.0))
        before = agent.select_actions(np.asarray([state]), explore=False)[0]
        for _ in range(120):
            self.assertIsNotNone(agent.learn())
        after = agent.select_actions(np.asarray([state]), explore=False)[0]
        self.assertEqual(after, 1)
        self.assertGreater(agent.optimizer_steps, 0)
        # The seeded initial policy can already choose correctly; training must
        # at minimum produce a clear preference in expected action value.
        values, _ = agent.inspect(np.asarray([state]))
        self.assertGreater(values[0, 1], values[0, 0])

    def test_checkpoint_pair_falls_back_to_matching_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_path, checkpoint = root / "replay.npz", root / "model.pt"
            config = AgentConfig(observation_size=4, action_count=2, batch_size=2,
                                 learning_starts=2, atom_count=5, value_min=-2,
                                 value_max=2, seed=5)
            replay = PrioritizedReplayBuffer(observation_size=4, capacity=16, seed=5)
            for index in range(4):
                state = np.full(4, index, dtype=np.float32)
                replay.add(Transition(state, index % 2, 0.1, state + 1, False, 0.99))
            agent = RainbowAgent(replay, config=config, device="cpu")
            agent.save(checkpoint, replay_path)
            first_snapshot = replay.snapshot_id
            replay.add(Transition(np.ones(4), 1, 1.0, np.zeros(4), True, 0.0))
            agent.save(checkpoint, replay_path)
            self.assertNotEqual(replay.snapshot_id, first_snapshot)

            unrelated = PrioritizedReplayBuffer(observation_size=4, capacity=16, seed=99)
            unrelated.add(Transition(np.zeros(4), 0, 0.0, np.ones(4), False, 0.99))
            unrelated.save(replay_path)
            recovered_checkpoint, recovered_replay = RainbowAgent.load_checkpoint_pair(
                checkpoint, replay_path, observation_size=4, seed=5,
            )
            self.assertEqual(recovered_checkpoint, checkpoint.with_suffix(".pt.bak"))
            self.assertEqual(recovered_replay.snapshot_id, first_snapshot)
            self.assertEqual(len(recovered_replay), 4)

    @unittest.skipUnless(__import__("torch").backends.mps.is_available(),
                         "This regression test exercises an MPS-mapped checkpoint")
    def test_mps_resume_keeps_rng_state_on_cpu(self) -> None:
        import torch

        replay = PrioritizedReplayBuffer(observation_size=4, capacity=8, seed=3)
        config = AgentConfig(observation_size=4, action_count=2, batch_size=2,
                             learning_starts=2, atom_count=5, value_min=-2,
                             value_max=2, seed=3)
        source = RainbowAgent(replay, config=config, device="cpu")
        payload = {
            "online": source.online.state_dict(),
            "target": source.target.state_dict(),
            "optimizer": source.optimizer.state_dict(),
            "steps": 17,
            "optimizer_steps": 4,
            "python_random": __import__("random").getstate(),
            "numpy_random": np.random.get_state(),
            "torch_random": torch.get_rng_state().to("mps"),
        }
        resumed = RainbowAgent(replay, config=config, device="cpu")
        with patch("mario_ai_fceux.agent.torch.load", return_value=payload):
            resumed.load("unused.pt")
        self.assertEqual(resumed.steps, 17)
        self.assertEqual(resumed.optimizer_steps, 4)

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
