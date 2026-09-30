from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.baselines import BaselineConfig, DDQNAgent, PPOAgent, load_baseline_policy
from mario_ai_fceux.environment import FileWorker, Observation
from mario_ai_fceux.import_lua_evaluation import champion_episodes


class BaselineAlgorithmTests(unittest.TestCase):
    def config(self) -> BaselineConfig:
        return BaselineConfig(observation_size=8, hidden_size=16, batch_size=8,
                              replay_capacity=32, target_sync_steps=2,
                              ppo_epochs=2, ppo_minibatch_size=4, seed=17)

    def test_ddqn_learns_a_preferred_action_on_fixed_terminal_rewards(self) -> None:
        configuration = self.config()
        configuration.learning_rate = 0.01
        agent = DDQNAgent(configuration, device="cpu")
        state = np.zeros(8, dtype=np.float32)
        for _ in range(120):
            agent.observe(state, 2, 1.0, state, True)
            agent.learn()
        with torch.no_grad():
            values = agent.online(torch.zeros(1, 8)).numpy()[0]
        self.assertEqual(int(values.argmax()), 2)
        self.assertGreater(agent.updates, 100)

    def test_ppo_uses_worker_local_gae_and_updates(self) -> None:
        agent = PPOAgent(self.config(), device="cpu")
        for worker_id, reward in (("left", 1.0), ("right", -1.0)):
            state = np.full(8, 0.1 if worker_id == "left" else -0.1, dtype=np.float32)
            action, log_probability, value = agent.act_with_statistics(state)
            agent.observe(worker_id, state, action, reward, log_probability, value, True)
        loss = agent.learn()
        self.assertIsNotNone(loss)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertEqual(agent.rollout, {})
        self.assertEqual(agent.updates, 1)

    def test_ppo_learns_the_rewarded_action_in_a_contextual_bandit(self) -> None:
        configuration = self.config()
        configuration.learning_rate = 0.01
        configuration.ppo_epochs = 4
        configuration.ppo_minibatch_size = 16
        agent = PPOAgent(configuration, device="cpu")
        state = np.zeros(8, dtype=np.float32)
        for _ in range(12):
            for _ in range(64):
                action, old_log_probability, old_value = agent.act_with_statistics(state)
                reward = float(action == 1)
                agent.observe("worker", state, action, reward, old_log_probability,
                              old_value, terminated=True)
            agent.learn()
        with torch.no_grad():
            logits, _ = agent.policy(torch.zeros((1, 8)))
            probability_of_rewarded_action = torch.softmax(logits, dim=-1)[0, 1].item()
        self.assertGreater(probability_of_rewarded_action, 0.9)
        self.assertEqual(agent.updates, 12)

    def test_both_baseline_checkpoints_load_as_deterministic_policies(self) -> None:
        sample_states = np.zeros((2, 8), dtype=np.float32)
        for agent_class in (DDQNAgent, PPOAgent):
            agent = agent_class(self.config(), device="cpu")
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "model.pt"
                agent.save(checkpoint)
                expected_algorithm = agent.algorithm
                label, policy = load_baseline_policy(checkpoint, "cpu")
                self.assertEqual(label, expected_algorithm)
                first = policy.select_actions(sample_states)
                second = policy.select_actions(sample_states)
                np.testing.assert_array_equal(first, second)
                self.assertEqual(first.shape, (2,))

    def test_checkpoint_resume_rejects_a_different_algorithm_config(self) -> None:
        agent = DDQNAgent(self.config(), device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ddqn.pt"
            agent.save(checkpoint)
            incompatible = DDQNAgent(BaselineConfig(observation_size=8, hidden_size=32,
                                                    batch_size=8, seed=17), device="cpu")
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                incompatible.load(checkpoint)

    def test_worker_hold_command_is_sequence_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = FileWorker("worker-00", directory)
            observation = Observation(12, np.zeros(184, dtype=np.float32), 32, 0, False, "")
            worker.hold(observation)
            command = json.loads((Path(directory) / "command.json").read_text(encoding="utf-8"))
        self.assertEqual(command, {"sequence": 12, "action": 3, "reset": False, "hold": True})

    def test_lua_import_uses_latest_champion_session_and_requested_level(self) -> None:
        log = """[2026-09-29 12:00:00] started | mode=training |
[2026-09-29 12:00:01] episode start | generation=4 | genome=1/100 | world=1 | level=1 | x=40 | power=0 | action_repeat=12
[2026-09-29 12:00:05] episode end | reason=death | fitness=1 | max_x=300 | frames=240 | decisions=20 | elapsed_seconds=4
[2026-09-29 12:01:00] started | mode=champion |
[2026-09-29 12:01:01] episode start | generation=4 | genome=1/100 | world=1 | level=1 | x=40 | power=0 | action_repeat=12
[2026-09-29 12:01:05] episode end | reason=stuck | fitness=1 | max_x=320 | frames=240 | decisions=20 | elapsed_seconds=4
[2026-09-29 12:01:06] episode start | generation=4 | genome=1/100 | world=1 | level=2 | x=40 | power=0 | action_repeat=12
[2026-09-29 12:01:10] episode end | reason=victory | fitness=1 | max_x=400 | frames=240 | decisions=20 | elapsed_seconds=4
"""
        episodes = champion_episodes(log, 1, 1)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["reason"], "timeout")
        self.assertEqual(episodes[0]["max_x"], 320)

    def test_lua_import_requires_a_champion_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "no Champion session"):
            champion_episodes("[x] started | mode=training |", 1, 1)


if __name__ == "__main__":
    unittest.main()
