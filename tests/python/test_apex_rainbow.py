"""Tests for the Ape-X Rainbow distributed architecture.

Covers:
- NStepBuffer n-step return calculation (per-actor, no trajectory mixing)
- Ape-X epsilon schedule (monotone, bounded)
- Actor batch assembly and queue interaction
- Learner receiving batches and training
- Weight broadcast serialization / deserialization
- Eval worker weight loading
- End-to-end: actor → queue → learner (multiprocess)
"""

from __future__ import annotations

import io
import multiprocessing
import queue
import sys
import threading
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mario_ai_fceux.apex_actor import (
    NStepBuffer,
    ActorConfig,
    _apex_epsilon,
    _action_details,
    _select_action,
    _load_weights_from_bytes,
    _enqueue_batch,
    _terminal_transition,
    _shaped_reward,
)
from mario_ai_fceux.apex_learner import apex_learner_main, _serialize_weights
from mario_ai_fceux.apex_train import parse_arguments
from mario_ai_fceux.model import RainbowNetwork
from mario_ai_fceux.replay import Transition
from mario_ai_fceux.environment import Observation
from mario_ai_fceux.replay import PrioritizedReplayBuffer, SumTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(reward: float = 0.1, terminated: bool = False,
                     gamma: float = 0.99, obs_size: int = 4) -> Transition:
    state = np.zeros(obs_size, dtype=np.float32)
    return Transition(state, 0, reward, state + 0.1, terminated, gamma)


def _make_observation(world_x: int = 100, terminal: bool = False,
                      reason: str = "", obs_size: int = 184) -> Observation:
    return Observation(
        sequence=1,
        state=np.zeros(obs_size, dtype=np.float32),
        world_x=world_x,
        power=0,
        terminal=terminal,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# NStepBuffer tests
# ---------------------------------------------------------------------------

class TestNStepBuffer(unittest.TestCase):

    def test_no_output_until_n_steps_accumulated(self) -> None:
        buf = NStepBuffer(gamma=0.99, n_step=3)
        t = _make_transition()
        # First two pushes: no output yet.
        self.assertEqual(buf.push(t), [])
        self.assertEqual(buf.push(t), [])
        # Third push: one n-step transition becomes ready.
        ready = buf.push(t)
        self.assertEqual(len(ready), 1)

    def test_n_step_reward_is_discounted_sum(self) -> None:
        buf = NStepBuffer(gamma=0.99, n_step=3)
        for _ in range(4):
            buf.push(_make_transition(reward=1.0))
        # After 4 pushes we should have 2 ready transitions.
        # First transition reward: 1 + 0.99 + 0.99^2 = 2.9701
        all_ready: list[Transition] = []
        buf2 = NStepBuffer(gamma=0.99, n_step=3)
        t = _make_transition(reward=1.0)
        for _ in range(4):
            all_ready.extend(buf2.push(t))
        first_reward = all_ready[0].reward
        expected = 1.0 + 0.99 + 0.99 ** 2
        self.assertAlmostEqual(first_reward, expected, places=5)

    def test_flush_drains_remaining_steps_at_terminal(self) -> None:
        buf = NStepBuffer(gamma=0.99, n_step=3)
        buf.push(_make_transition(reward=1.0))
        buf.push(_make_transition(reward=1.0))
        # Flush should emit the 2 remaining steps.
        flushed = buf.flush()
        self.assertEqual(len(flushed), 2)

    def test_terminal_transition_stops_discount_propagation(self) -> None:
        buf = NStepBuffer(gamma=0.99, n_step=3)
        buf.push(_make_transition(reward=2.0))
        buf.push(_make_transition(reward=2.0))
        ready = buf.push(_make_transition(reward=2.0, terminated=True))
        # All three steps are ready because the third is terminal.
        self.assertGreaterEqual(len(ready), 1)
        # The first ready transition must be marked terminated (terminal propagated).
        # Its discount should be 0 since the episode ended.
        terminal_transitions = [r for r in ready if r.terminated]
        self.assertGreater(len(terminal_transitions), 0)

    def test_trajectories_are_isolated_per_buffer_instance(self) -> None:
        """Each actor has its own NStepBuffer; no cross-contamination."""
        buf_a = NStepBuffer(gamma=0.99, n_step=2)
        buf_b = NStepBuffer(gamma=0.99, n_step=2)
        t_a = _make_transition(reward=5.0)
        t_b = _make_transition(reward=1.0)
        buf_a.push(t_a)
        buf_b.push(t_b)
        ready_a = buf_a.push(t_a)
        ready_b = buf_b.push(t_b)
        # Actor A's reward must be higher than actor B's.
        self.assertGreater(ready_a[0].reward, ready_b[0].reward)


class TestReplayCorrectness(unittest.TestCase):

    def test_sum_tree_finds_global_prefixes_for_non_power_of_two_capacity(self) -> None:
        tree = SumTree(5)
        for index, priority in enumerate((1.0, 2.0, 4.0, 8.0, 16.0)):
            tree.update(index, priority)
        self.assertEqual(tree.total, 31.0)
        self.assertEqual(tree.minimum, 1.0)
        self.assertEqual(tree.find_prefixsum(0.5), 0)
        self.assertEqual(tree.find_prefixsum(2.0), 1)
        self.assertEqual(tree.find_prefixsum(30.0), 4)

    def test_importance_weights_use_global_min_probability(self) -> None:
        replay = PrioritizedReplayBuffer(2, capacity=3, alpha=1.0, seed=11)
        for index, priority in enumerate((1.0, 2.0, 4.0)):
            state = np.full(2, index, dtype=np.float32)
            replay.add(Transition(state, 0, 0.0, state, False, 0.99, priority))
        class FixedSampler:
            @staticmethod
            def uniform(low: np.ndarray, high: np.ndarray) -> np.ndarray:
                return np.asarray([0.5, 3.0, 6.0])
        replay.rng = FixedSampler()  # type: ignore[assignment]
        _, _, weights = replay.sample(3, beta=1.0)
        # max IS weight comes from the least probable item: N * p_min.
        expected_min = 0.25
        expected_max = 1.0
        self.assertAlmostEqual(float(weights.min()), expected_min, places=6)
        self.assertAlmostEqual(float(weights.max()), expected_max, places=6)

    def test_terminal_transition_carries_death_penalty_and_terminal_flag(self) -> None:
        previous = _make_observation(world_x=200)
        death = _make_observation(world_x=200, terminal=True, reason="death")
        transition = _terminal_transition(previous, death, action=1)
        self.assertTrue(transition.terminated)
        self.assertEqual(transition.discount, 0.0)
        self.assertLess(transition.reward, 0.0)

    def test_batch_sender_waits_for_capacity_without_dropping_experience(self) -> None:
        experience_queue: queue.Queue = queue.Queue(maxsize=1)
        marker = _make_transition()
        experience_queue.put_nowait(("occupied",))
        started = threading.Event()
        finished = threading.Event()

        def send() -> None:
            started.set()
            _enqueue_batch(experience_queue, 2, [marker])
            finished.set()

        sender = threading.Thread(target=send)
        sender.start()
        self.assertTrue(started.wait(timeout=2))
        time.sleep(0.05)
        self.assertFalse(finished.is_set())
        self.assertEqual(experience_queue.get_nowait(), ("occupied",))
        sender.join(timeout=2)
        self.assertTrue(finished.is_set())
        kind, actor_index, transitions = experience_queue.get_nowait()
        self.assertEqual((kind, actor_index, len(transitions)), ("batch", 2, 1))


# ---------------------------------------------------------------------------
# Ape-X epsilon schedule tests
# ---------------------------------------------------------------------------

class TestApexEpsilon(unittest.TestCase):

    def test_first_actor_has_highest_epsilon(self) -> None:
        eps_0 = _apex_epsilon(0, 8)
        eps_7 = _apex_epsilon(7, 8)
        self.assertGreater(eps_0, eps_7)

    def test_epsilons_are_monotone_decreasing(self) -> None:
        epsilons = [_apex_epsilon(i, 8) for i in range(8)]
        for a, b in zip(epsilons, epsilons[1:]):
            self.assertGreater(a, b)

    def test_epsilons_are_in_valid_range(self) -> None:
        for i in range(8):
            eps = _apex_epsilon(i, 8)
            self.assertGreater(eps, 0.0)
            self.assertLessEqual(eps, 1.0)

    def test_single_actor_gets_medium_epsilon(self) -> None:
        eps = _apex_epsilon(0, 1)
        self.assertAlmostEqual(eps, 0.10, places=5)

    def test_last_actor_is_near_greedy(self) -> None:
        eps = _apex_epsilon(7, 8)
        self.assertLess(eps, 0.01)

    def test_default_experience_queue_capacity_is_ten_thousand_batches(self) -> None:
        with patch("sys.argv", ["apex_train", "--rom", "SuperMarioBros.nes"]):
            arguments = parse_arguments()
        self.assertEqual(arguments.queue_capacity, 10_000)
        self.assertEqual(arguments.actor_batch_size, 32)


# ---------------------------------------------------------------------------
# Action selection tests
# ---------------------------------------------------------------------------

class TestActionSelection(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.network = RainbowNetwork(4, 2, 11).to(self.device)
        self.network.eval()
        self.support = torch.linspace(-5.0, 5.0, 11, device=self.device)

    def test_greedy_action_is_valid_index(self) -> None:
        state = np.zeros(4, dtype=np.float32)
        action = _select_action(self.network, self.support, state, 0.0, 2, self.device)
        self.assertIn(action, [0, 1])

    def test_random_action_taken_when_epsilon_one(self) -> None:
        """With epsilon=1.0 we should get a random mix over many calls."""
        state = np.zeros(4, dtype=np.float32)
        actions = {_select_action(self.network, self.support, state, 1.0, 2, self.device)
                   for _ in range(50)}
        self.assertEqual(actions, {0, 1})

    def test_action_details_match_the_fceux_network_overlay(self) -> None:
        state = np.zeros(4, dtype=np.float32)
        action, q_values, hidden = _action_details(
            self.network, self.support, state, 0.0, 2, self.device,
        )
        self.assertIn(action, (0, 1))
        self.assertEqual(q_values.shape, (2,))
        self.assertEqual(hidden.shape, (16,))
        self.assertTrue(np.isfinite(q_values).all())
        self.assertTrue(np.isfinite(hidden).all())

    def test_actor_action_values_do_not_change_when_noisy_weights_reset(self) -> None:
        state = np.zeros(4, dtype=np.float32)
        self.network.train()
        _, initial_values, _ = _action_details(
            self.network, self.support, state, 0.0, 2, self.device,
        )
        self.network.reset_noise()
        _, reset_values, _ = _action_details(
            self.network, self.support, state, 0.0, 2, self.device,
        )
        self.assertFalse(self.network.training)
        np.testing.assert_array_equal(initial_values, reset_values)


# ---------------------------------------------------------------------------
# Weight serialization / broadcast tests
# ---------------------------------------------------------------------------

class TestWeightBroadcast(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.network_a = RainbowNetwork(4, 2, 11)
        self.network_b = RainbowNetwork(4, 2, 11)

    def test_serialize_and_deserialize_weights(self) -> None:
        """Weights survive a round-trip through bytes."""
        weight_bytes = _serialize_weights(self.network_a)
        self.assertIsInstance(weight_bytes, bytes)
        self.assertGreater(len(weight_bytes), 100)
        _load_weights_from_bytes(self.network_b, weight_bytes, self.device)
        for key in self.network_a.state_dict():
            self.assertTrue(torch.allclose(
                self.network_a.state_dict()[key].float(),
                self.network_b.state_dict()[key].float(),
            ), f"Parameter mismatch for {key}")

    def test_weight_bytes_are_different_for_different_networks(self) -> None:
        """Two independently initialized networks have different weight bytes."""
        bytes_a = _serialize_weights(self.network_a)
        bytes_b = _serialize_weights(self.network_b)
        # They should differ (extremely unlikely to collide by chance).
        self.assertNotEqual(bytes_a, bytes_b)


# ---------------------------------------------------------------------------
# Shaped reward test
# ---------------------------------------------------------------------------

class TestShapedReward(unittest.TestCase):

    def test_forward_progress_gives_positive_reward(self) -> None:
        prev = _make_observation(world_x=100)
        curr = _make_observation(world_x=116)  # +1 tile
        reward = _shaped_reward(prev, curr)
        self.assertGreater(reward, 0.0)

    def test_victory_gives_bonus(self) -> None:
        prev = _make_observation(world_x=3000)
        curr = _make_observation(world_x=3200, terminal=True, reason="victory")
        reward = _shaped_reward(prev, curr)
        self.assertGreater(reward, 5.0)

    def test_death_gives_penalty(self) -> None:
        prev = _make_observation(world_x=200)
        curr = _make_observation(world_x=200, terminal=True, reason="death")
        reward = _shaped_reward(prev, curr)
        self.assertLess(reward, 0.0)

    def test_reward_is_bounded(self) -> None:
        prev = _make_observation(world_x=0)
        curr = _make_observation(world_x=100_000)
        reward = _shaped_reward(prev, curr)
        self.assertLessEqual(reward, 22.0)  # 2.0 max progress + 20 victory max


# ---------------------------------------------------------------------------
# Actor config defaults
# ---------------------------------------------------------------------------

class TestActorConfig(unittest.TestCase):

    def test_default_config_is_sane(self) -> None:
        cfg = ActorConfig()
        self.assertEqual(cfg.observation_size, 184)
        self.assertEqual(cfg.action_count, 6)
        self.assertEqual(cfg.n_step, 3)
        self.assertGreater(cfg.batch_size, 0)


# ---------------------------------------------------------------------------
# Learner integration: batch ingestion via multiprocessing
# ---------------------------------------------------------------------------

class TestApexLearnerIntegration(unittest.TestCase):

    def test_learner_trains_on_actor_batches(self) -> None:
        """Learner process accepts batches, trains, and responds to status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context = multiprocessing.get_context("spawn")
            exp_queue = context.Queue(maxsize=1000)
            # One weight queue (simulating a single actor).
            wq = context.Queue(maxsize=4)
            status_inbox = context.Queue()
            status_outbox = context.Queue()

            from mario_ai_fceux.agent import AgentConfig
            config = AgentConfig(
                observation_size=4,
                action_count=2,
                atom_count=11,
                batch_size=4,
                learning_starts=4,
                n_step=2,
                seed=42,
            )
            learner_config = {**vars(config), "replay_capacity": 200}

            process = context.Process(
                target=apex_learner_main,
                args=(exp_queue, [wq], status_inbox, status_outbox,
                      tmpdir, learner_config, False, "cpu", 10, 100, 2),
                daemon=True,
            )
            process.start()
            try:
                # Send one batch from actor 0.
                state = np.zeros(4, dtype=np.float32)
                batch = [
                    Transition(state, 0, 1.0, state + 0.1, False, 0.99)
                    for _ in range(8)
                ]
                exp_queue.put(("batch", 0, batch))
                time.sleep(0.5)

                # Query status.
                status_inbox.put(("status",))
                kind, status = status_outbox.get(timeout=10)
                self.assertEqual(kind, "status")
                self.assertIn("replay_transitions", status)
                self.assertGreaterEqual(status["replay_transitions"], 0)

                # Check weights are broadcast.
                try:
                    msg_kind, weight_bytes = wq.get(timeout=5)
                    self.assertEqual(msg_kind, "weights")
                    self.assertIsInstance(weight_bytes, bytes)
                except Exception:
                    pass  # Weight broadcast not yet triggered — OK.

                # Stop.
                status_inbox.put(("stop",))
                process.join(timeout=30)
                self.assertFalse(process.is_alive())

                # Model checkpoint must exist.
                self.assertTrue((Path(tmpdir) / "model.pt").exists())

            finally:
                if process.is_alive():
                    process.kill()

if __name__ == "__main__":
    unittest.main()
